"""Behavioral parity tests against sortedcontainers 2.4."""

from __future__ import annotations

import random
from collections.abc import MutableSequence, MutableSet

import numpy as np
import pytest
from sortedcontainers import (
    SortedDict as RefSortedDict,
    SortedKeyList as RefSortedKeyList,
    SortedList as RefSortedList,
    SortedSet as RefSortedSet,
)

from mojo_sortedcontainers import SortedDict, SortedKeyList, SortedList, SortedSet
from mojo_sortedcontainers import _lib


@pytest.mark.parametrize(
    "values",
    [
        [],
        [5, 1, 5, -2, 10],
        ["pear", "apple", "fig", "banana"],
        [(2, "b"), (1, "z"), (2, "a")],
    ],
)
def test_sorted_list_construction(values):
    ours, ref = SortedList(values), RefSortedList(values)
    assert list(ours) == list(ref)
    assert len(ours) == len(ref)
    assert repr(ours) == repr(ref)
    ours._check()
    assert isinstance(ours, MutableSequence)


def test_native_integer_and_float_construction_matches_upstream():
    rng = np.random.default_rng(1)
    integers = rng.integers(-10_000, 10_000, 20_000).tolist()
    floats = rng.normal(size=20_000).tolist()
    assert list(SortedList(integers)) == list(RefSortedList(integers))
    assert list(SortedList(floats)) == list(RefSortedList(floats))


@pytest.mark.parametrize("dtype, symbol", [(np.int64, "msc_sort_i64"), (np.float64, "msc_sort_f64")])
def test_parallel_native_sort_handles_uneven_chunks(dtype, symbol):
    rng = np.random.default_rng(11)
    values = rng.integers(-1_000_000, 1_000_000, 262_147).astype(dtype)
    expected = np.sort(values.copy())
    scratch = np.empty_like(values)
    assert getattr(_lib.lib(), symbol)(
        _lib.addr(values),
        len(values),
        _lib.addr(scratch),
    ) == 0
    np.testing.assert_array_equal(values, expected)


def test_parallel_sort_threshold_keeps_small_inputs_serial(monkeypatch):
    calls = []

    class FakeLibrary:
        @staticmethod
        def msc_sort_i64(address, length, scratch):
            calls.append((length, scratch))
            return 0

    monkeypatch.setattr(_lib, "_library", FakeLibrary())
    monkeypatch.setattr(_lib, "_PARALLEL_SORT_THRESHOLD", 8)
    _lib.sort_numeric(list(range(7)))
    _lib.sort_numeric(list(range(9)))
    assert calls[0] == (7, None)
    assert calls[1][0] == 9 and calls[1][1] is not None


def test_native_buffer_validation_and_kernel_failure_propagation(monkeypatch):
    with pytest.raises(TypeError):
        _lib.addr(np.arange(4, dtype=np.int32), np.int64)
    with pytest.raises(ValueError):
        _lib.addr(np.arange(8, dtype=np.int64)[::2], np.int64)
    readonly = np.arange(4, dtype=np.int64)
    readonly.flags.writeable = False
    with pytest.raises(ValueError):
        _lib.addr(readonly, np.int64, writable=True)

    class RejectingLibrary:
        @staticmethod
        def msc_sort_i64(address, length, scratch):
            return -1

    monkeypatch.setattr(_lib, "_library", RejectingLibrary())
    with pytest.raises(RuntimeError, match="rejected"):
        _lib.sort_numeric(list(range(10)))


@pytest.mark.parametrize(
    "dtype,symbol",
    [(np.int64, "msc_merge_i64"), (np.float64, "msc_merge_f64")],
)
def test_native_merge_all_short_simd_tails(dtype, symbol):
    kernel = getattr(_lib.lib(), symbol)
    for left_size in range(18):
        for right_size in range(18):
            left = np.arange(0, left_size * 2, 2, dtype=dtype)
            right = np.arange(1, right_size * 2 + 1, 2, dtype=dtype)
            result = np.empty(left_size + right_size, dtype=dtype)
            status = kernel(
                _lib.addr(left, dtype),
                left_size,
                _lib.addr(right, dtype),
                right_size,
                _lib.addr(result, dtype, writable=True),
            )
            assert status == 0
            np.testing.assert_array_equal(result, np.sort(np.concatenate((left, right))))


def test_native_exports_reject_invalid_lengths_and_null_buffers():
    native = _lib.lib()
    assert native.msc_sort_i64(None, -1, None) != 0
    assert native.msc_sort_i64(None, 1, None) != 0
    assert native.msc_merge_f64(None, 1, None, 0, None) != 0
    assert native.msc_bisect_i64(None, 1, None, 1, None, 0) != 0
    assert native.msc_sort_i64(None, 0, None) == 0


@pytest.mark.parametrize(
    "left,incoming",
    [
        (list(range(2_055)), list(range(10_000, 12_049))),
        (
            [float(value) for value in range(2_055)],
            [float(value) for value in range(10_000, 12_049)],
        ),
    ],
)
def test_native_merge_simd_tail_matches_upstream(left, incoming):
    ours, ref = SortedList(left), RefSortedList(left)
    ours.update(incoming)
    ref.update(incoming)
    assert list(ours) == list(ref)


def test_numeric_edge_cases_fall_back_without_losing_python_semantics():
    huge = [1 << 100, -(1 << 100)]
    incoming = list(range(3_000))
    ours, ref = SortedList(huge), RefSortedList(huge)
    ours.update(incoming)
    ref.update(incoming)
    assert list(ours) == list(ref)

    floats = [0.0, -0.0, float("nan")] * 1_000
    ours, ref = SortedList(floats), RefSortedList(floats)
    assert all(
        (left != left and right != right) or left.hex() == right.hex()
        for left, right in zip(ours, ref)
    )


def test_sorted_list_random_mutations():
    ours, ref = SortedList(), RefSortedList()
    rng = random.Random(2)
    for _ in range(600):
        value = rng.randrange(-30, 31)
        operation = rng.choice(("add", "add", "discard", "remove", "pop"))
        if operation == "add":
            ours.add(value)
            ref.add(value)
        elif operation == "discard":
            ours.discard(value)
            ref.discard(value)
        elif operation == "remove":
            if value in ref:
                ours.remove(value)
                ref.remove(value)
            else:
                with pytest.raises(ValueError):
                    ours.remove(value)
        elif ours:
            index = rng.randrange(-len(ours), len(ours))
            assert ours.pop(index) == ref.pop(index)
        assert list(ours) == list(ref)
    ours._check()


def test_large_sorted_list_chunked_insertions_and_deletions():
    values = list(range(10_000))
    ours, ref = SortedList(values), RefSortedList(values)
    for value in range(-75, 10_075, 7):
        ours.add(value)
        ref.add(value)
    del ours[100:4_000:19]
    del ref[100:4_000:19]
    assert list(ours) == list(ref)
    assert ours.bisect_left(5_001) == ref.bisect_left(5_001)
    assert ours.bisect_right(5_001) == ref.bisect_right(5_001)
    ours.clear()
    ref.clear()
    ours.add(3)
    ref.add(3)
    assert list(ours) == list(ref)
    ours._check()


def test_sorted_list_update_delete_and_slice():
    ours, ref = SortedList(range(0, 100, 2)), RefSortedList(range(0, 100, 2))
    ours.update(range(1, 100, 2))
    ref.update(range(1, 100, 2))
    assert ours[10:30:3] == ref[10:30:3]
    del ours[5:75:4]
    del ref[5:75:4]
    assert list(ours) == list(ref)
    ours.clear()
    ref.clear()
    assert list(ours) == list(ref)


@pytest.mark.parametrize("inclusive", [(True, True), (True, False), (False, True), (False, False)])
@pytest.mark.parametrize("reverse", [False, True])
def test_sorted_list_ranges(inclusive, reverse):
    values = [1, 2, 2, 3, 4, 6, 6, 8]
    ours, ref = SortedList(values), RefSortedList(values)
    assert list(ours.irange(2, 6, inclusive, reverse)) == list(
        ref.irange(2, 6, inclusive, reverse)
    )
    assert list(ours.islice(1, 7, reverse)) == list(ref.islice(1, 7, reverse))


def test_sorted_list_search_count_index_and_comparison():
    values = [1, 2, 2, 2, 5, 9]
    ours, ref = SortedList(values), RefSortedList(values)
    for value in range(11):
        assert ours.bisect_left(value) == ref.bisect_left(value)
        assert ours.bisect_right(value) == ref.bisect_right(value)
        assert ours.count(value) == ref.count(value)
        assert (value in ours) == (value in ref)
    assert ours.index(2, 2) == ref.index(2, 2)
    assert ours == tuple(ref)
    assert ours < list(ref) + [10]
    assert list(ours + [3, 0]) == list(ref + [3, 0])
    assert list(ours * 2) == list(ref * 2)


def test_sorted_list_copy_and_remaining_sequence_arithmetic():
    ours, ref = SortedList([3, 1, 2]), RefSortedList([3, 1, 2])
    assert list(ours.copy()) == list(ref.copy())
    assert list([4, 0] + ours) == list([4, 0] + ref)
    assert list(2 * ours) == list(2 * ref)
    ours += [5, -1]
    ref += [5, -1]
    assert list(ours) == list(ref)
    ours *= 2
    ref *= 2
    assert list(ours) == list(ref)


def test_sorted_list_rejected_positional_mutations():
    ours = SortedList([1, 2])
    with pytest.raises(NotImplementedError):
        ours.append(3)
    with pytest.raises(NotImplementedError):
        ours.extend([3])
    with pytest.raises(NotImplementedError):
        ours.insert(0, 3)
    with pytest.raises(NotImplementedError):
        ours.reverse()
    with pytest.raises(NotImplementedError):
        ours[0] = 3


def test_mojo_bisect_many_matches_repeated_upstream_search():
    rng = np.random.default_rng(3)
    values = rng.integers(-1_000_000, 1_000_000, 50_000).tolist()
    queries = rng.integers(-1_200_000, 1_200_000, 10_000).tolist()
    ours, ref = SortedList(values), RefSortedList(values)
    assert ours.bisect_many(queries) == [ref.bisect_left(value) for value in queries]
    assert ours.bisect_many(queries, right=True) == [
        ref.bisect_right(value) for value in queries
    ]


def test_sorted_key_list_and_factory():
    key = lambda value: (len(value), value[-1])
    values = ["bbb", "a", "cc", "ddd", "bb", "z"]
    ours, ref = SortedList(values, key=key), RefSortedList(values, key=key)
    assert isinstance(ours, SortedKeyList)
    assert list(ours) == list(ref)
    assert ours.key is key
    ours.add("xx")
    ref.add("xx")
    ours.update(["q", "eeee"])
    ref.update(["q", "eeee"])
    assert list(ours) == list(ref)
    assert ours.bisect_key_left((2, "b")) == ref.bisect_key_left((2, "b"))
    assert ours.bisect_key_right((2, "x")) == ref.bisect_key_right((2, "x"))
    assert list(ours.irange_key((2, "a"), (3, "z"))) == list(
        ref.irange_key((2, "a"), (3, "z"))
    )
    ours._check()


def test_sorted_key_list_equal_keys_preserve_values_and_membership():
    key = abs
    ours, ref = SortedKeyList([2, -2, 1, -1], key), RefSortedKeyList([2, -2, 1, -1], key)
    for value in (-2, -1, 1, 2, 3):
        assert (value in ours) == (value in ref)
        assert ours.count(value) == ref.count(value)
    ours.discard(-2)
    ref.discard(-2)
    assert list(ours) == list(ref)


def test_sorted_dict_order_mutation_and_lookup():
    ours = SortedDict({3: "c", 1: "a"})
    ref = RefSortedDict({3: "c", 1: "a"})
    for key, value in [(2, "b"), (0, "z"), (3, "C")]:
        ours[key] = value
        ref[key] = value
    ours.update({5: "e", 4: "d"})
    ref.update({5: "e", 4: "d"})
    assert list(ours) == list(ref)
    assert list(reversed(ours)) == list(reversed(ref))
    assert list(ours.items()) == list(ref.items())
    assert ours.peekitem(1) == ref.peekitem(1)
    assert ours.popitem(2) == ref.popitem(2)
    assert ours.pop(99, "missing") == ref.pop(99, "missing")
    assert ours.setdefault(7, "g") == ref.setdefault(7, "g")
    del ours[1]
    del ref[1]
    assert dict(ours) == dict(ref)
    ours._check()


def test_sorted_dict_views_are_indexable():
    ours, ref = SortedDict({3: "c", 1: "a", 2: "b"}), RefSortedDict({3: "c", 1: "a", 2: "b"})
    for method in ("keys", "items", "values"):
        ours_view, ref_view = getattr(ours, method)(), getattr(ref, method)()
        assert ours_view[0] == ref_view[0]
        assert ours_view[-1] == ref_view[-1]
        assert ours_view[::2] == ref_view[::2]
        assert list(reversed(ours_view)) == list(reversed(ref_view))


def test_sorted_dict_key_function_and_ranges():
    key = lambda value: (len(value), value)
    data = {"bbb": 3, "a": 1, "cc": 2, "z": 0}
    ours, ref = SortedDict(key, data), RefSortedDict(key, data)
    assert list(ours) == list(ref)
    assert ours.key is key
    assert ours.bisect_left("bb") == ref.bisect_left("bb")
    assert ours.bisect_key_left((2, "bb")) == ref.bisect_key_left((2, "bb"))
    assert list(ours.islice(1, 3)) == list(ref.islice(1, 3))
    copied = ours.copy()
    assert copied.key is key and list(copied.items()) == list(ours.items())


def test_sorted_dict_union_and_fromkeys():
    ours = SortedDict({2: "b", 1: "a"})
    ref = RefSortedDict({2: "b", 1: "a"})
    assert list((ours | {0: "z"}).items()) == list((ref | {0: "z"}).items())
    assert list(({4: "d"} | ours).items()) == list(({4: "d"} | ref).items())
    assert list(SortedDict.fromkeys([3, 1, 2], 0).items()) == list(
        RefSortedDict.fromkeys([3, 1, 2], 0).items()
    )


def test_sorted_set_construction_indexing_and_mutation():
    ours, ref = SortedSet([5, 1, 3, 1]), RefSortedSet([5, 1, 3, 1])
    assert list(ours) == list(ref)
    assert isinstance(ours, MutableSet)
    assert ours[1:] == ref[1:]
    ours.add(2)
    ref.add(2)
    ours.discard(5)
    ref.discard(5)
    assert ours.pop(1) == ref.pop(1)
    ours.update([8, 0], [9, 3])
    ref.update([8, 0], [9, 3])
    del ours[1:3]
    del ref[1:3]
    assert list(ours) == list(ref)
    ours._check()


@pytest.mark.parametrize(
    "method",
    ["difference", "intersection", "symmetric_difference", "union"],
)
def test_sorted_set_algebra(method):
    ours, ref = SortedSet(range(8)), RefSortedSet(range(8))
    other = {0, 2, 8, 10}
    assert list(getattr(ours, method)(other)) == list(getattr(ref, method)(other))


@pytest.mark.parametrize(
    "method",
    ["difference_update", "intersection_update", "symmetric_difference_update", "update"],
)
def test_sorted_set_inplace_algebra(method):
    ours, ref = SortedSet(range(8)), RefSortedSet(range(8))
    other = {0, 2, 8, 10}
    getattr(ours, method)(other)
    getattr(ref, method)(other)
    assert list(ours) == list(ref)


def test_sorted_set_key_ranges_and_relations():
    key = abs
    ours, ref = SortedSet([-5, 3, -2, 1], key), RefSortedSet([-5, 3, -2, 1], key)
    assert list(ours) == list(ref)
    assert list(ours.irange(-2, 3)) == list(ref.irange(-2, 3))
    assert ours == ref
    assert ours < set(ours) | {20}
    assert ours.count(-2) == ref.count(-2)
    assert ours.index(3) == ref.index(3)
    assert ours.bisect_key_left(2) == ref.bisect_key_left(2)
    assert (ours == list(ours)) == (ref == list(ref))


def test_sorted_set_algebra_operators_match_upstream():
    ours, ref = SortedSet(range(6)), RefSortedSet(range(6))
    other = {0, 2, 8}
    for ours_result, ref_result in (
        (ours - other, ref - other),
        (ours & other, ref & other),
        (ours | other, ref | other),
        (ours ^ other, ref ^ other),
        (other & ours, other & ref),
        (other | ours, other | ref),
        (other ^ ours, other ^ ref),
    ):
        assert list(ours_result) == list(ref_result)

    for operation in ("__isub__", "__iand__", "__ior__", "__ixor__"):
        ours_copy, ref_copy = ours.copy(), ref.copy()
        getattr(ours_copy, operation)(other)
        getattr(ref_copy, operation)(other)
        assert list(ours_copy) == list(ref_copy)
