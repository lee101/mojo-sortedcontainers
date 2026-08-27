"""Sorted list implementations with Mojo-backed homogeneous numeric bulk paths."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import MutableSequence
from operator import eq, ge, gt, le, lt, ne

from ._lib import bisect_many_numeric, merge_numeric, sort_numeric

_NATIVE_THRESHOLD = 2048
_BLOCK_THRESHOLD = 4096
_BLOCK_LOAD = 512


def identity(value):
    return value


def _sorted_values(values: list) -> tuple[list, str | None]:
    if len(values) >= _NATIVE_THRESHOLD:
        native = sort_numeric(values)
        if native is not None:
            return native
    return sorted(values), None


class _ChunkedList:
    def __init__(self, values):
        self._lists = [
            values[pos:pos + _BLOCK_LOAD]
            for pos in range(0, len(values), _BLOCK_LOAD)
        ]
        self._reindex()

    def _reindex(self):
        self._maxes = [block[-1] for block in self._lists]
        total = 0
        self._prefix = []
        for block in self._lists:
            total += len(block)
            self._prefix.append(total)
        self._length = total

    def _ensure_prefix(self):
        if self._prefix is not None:
            return
        total = 0
        self._prefix = []
        for block in self._lists:
            total += len(block)
            self._prefix.append(total)

    def _loc(self, index):
        self._ensure_prefix()
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("list index out of range")
        block = bisect_right(self._prefix, index)
        before = self._prefix[block - 1] if block else 0
        return block, index - before

    def add(self, value):
        if not self._lists:
            self._lists.append([value])
            self._maxes.append(value)
            self._prefix = [1]
            self._length = 1
            return
        block = bisect_right(self._maxes, value)
        if block == len(self._lists):
            block -= 1
        values = self._lists[block]
        values.insert(bisect_right(values, value), value)
        self._length += 1
        self._prefix = None
        if len(values) > _BLOCK_LOAD * 2:
            self._lists[block:block + 1] = [
                values[:_BLOCK_LOAD],
                values[_BLOCK_LOAD:],
            ]
            self._maxes[block:block + 1] = [
                self._lists[block][-1],
                self._lists[block + 1][-1],
            ]
        else:
            self._maxes[block] = values[-1]

    def bisect_left(self, value):
        self._ensure_prefix()
        block = bisect_left(self._maxes, value)
        if block == len(self._lists):
            return len(self)
        before = self._prefix[block - 1] if block else 0
        return before + bisect_left(self._lists[block], value)

    def bisect_right(self, value):
        self._ensure_prefix()
        block = bisect_right(self._maxes, value)
        if block == len(self._lists):
            return len(self)
        before = self._prefix[block - 1] if block else 0
        return before + bisect_right(self._lists[block], value)

    def to_list(self):
        return [value for block in self._lists for value in block]

    def clear(self):
        self._lists.clear()
        self._reindex()

    def __len__(self):
        return self._length

    def __iter__(self):
        for block in self._lists:
            yield from block

    def __reversed__(self):
        for block in reversed(self._lists):
            yield from reversed(block)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.to_list()[index]
        block, offset = self._loc(index)
        return self._lists[block][offset]

    def __delitem__(self, index):
        if isinstance(index, slice):
            values = self.to_list()
            del values[index]
            self.__init__(values)
            return
        block, offset = self._loc(index)
        del self._lists[block][offset]
        if not self._lists[block]:
            del self._lists[block]
        elif (
            block + 1 < len(self._lists)
            and len(self._lists[block]) + len(self._lists[block + 1]) <= _BLOCK_LOAD
        ):
            self._lists[block].extend(self._lists.pop(block + 1))
        self._reindex()

    def pop(self, index=-1):
        block, offset = self._loc(index)
        value = self._lists[block].pop(offset)
        if not self._lists[block]:
            del self._lists[block]
        self._reindex()
        return value

    def index(self, value, start=0, stop=None):
        values = self.to_list()
        return values.index(value, start, len(values) if stop is None else stop)


class SortedList(MutableSequence):
    """A mutable sequence that keeps its values in ascending order."""

    def __new__(cls, iterable=None, key=None):
        if cls is SortedList and key is not None:
            return object.__new__(SortedKeyList)
        return object.__new__(cls)

    def __init__(self, iterable=None, key=None):
        if key is not None:
            raise TypeError("inherit SortedKeyList for key argument")
        values = [] if iterable is None else list(iterable)
        self._values, self._native_kind = _sorted_values(values)

    @property
    def key(self):
        return None

    def _invalidate(self):
        self._native_kind = None

    def clear(self):
        self._values.clear()
        self._native_kind = None

    def add(self, value):
        if isinstance(self._values, _ChunkedList):
            self._values.add(value)
        elif len(self._values) >= _BLOCK_THRESHOLD:
            self._values = _ChunkedList(self._values)
            self._values.add(value)
        else:
            self._values.insert(bisect_right(self._values, value), value)
        self._invalidate()

    def update(self, iterable):
        incoming = list(iterable)
        if not incoming:
            return
        if len(incoming) >= _NATIVE_THRESHOLD:
            native = sort_numeric(incoming, as_array=True)
            if native is not None:
                ordered, kind = native
                if not self._values:
                    self._values = ordered.tolist()
                    self._native_kind = kind
                    return
                safe_left = kind == self._native_kind or (
                    kind == "i64"
                    and all(
                        type(value) is int and -(1 << 63) <= value < (1 << 63)
                        for value in self._values
                    )
                ) or (
                    kind == "f64"
                    and all(type(value) is float and value == value for value in self._values)
                    and not (
                        any(value == 0.0 and str(value).startswith("-") for value in self._values)
                        and any(value == 0.0 and not str(value).startswith("-") for value in self._values)
                    )
                )
                if safe_left:
                    self._values = merge_numeric(self._values, ordered, kind)
                    self._native_kind = kind
                    return
        if isinstance(self._values, _ChunkedList):
            self._values = self._values.to_list()
        self._values.extend(incoming)
        self._values.sort()
        self._invalidate()

    def __contains__(self, value):
        pos = self.bisect_left(value)
        return pos < len(self._values) and self._values[pos] == value

    def discard(self, value):
        pos = self.bisect_left(value)
        if pos < len(self._values) and self._values[pos] == value:
            del self._values[pos]
            self._invalidate()

    def remove(self, value):
        pos = self.bisect_left(value)
        if pos == len(self._values) or self._values[pos] != value:
            raise ValueError(f"{value!r} not in list")
        del self._values[pos]
        self._invalidate()

    def __delitem__(self, index):
        del self._values[index]
        self._invalidate()

    def __getitem__(self, index):
        return self._values[index]

    def __setitem__(self, index, value):
        raise NotImplementedError("use ``sl.add(value)`` instead")

    def __iter__(self):
        return iter(self._values)

    def __reversed__(self):
        return reversed(self._values)

    def reverse(self):
        raise NotImplementedError("use ``reversed(sl)`` instead")

    def islice(self, start=None, stop=None, reverse=False):
        length = len(self)
        start = 0 if start is None else start
        stop = length if stop is None else stop
        if start < 0:
            start = max(0, length + start)
        if stop < 0:
            stop = max(0, length + stop)
        values = self._values[start:stop]
        return iter(values[::-1] if reverse else values)

    def irange(self, minimum=None, maximum=None, inclusive=(True, True), reverse=False):
        left = 0 if minimum is None else (
            bisect_left(self._values, minimum) if inclusive[0]
            else bisect_right(self._values, minimum)
        )
        right = len(self) if maximum is None else (
            bisect_right(self._values, maximum) if inclusive[1]
            else bisect_left(self._values, maximum)
        )
        values = self._values[left:right]
        return iter(values[::-1] if reverse else values)

    def __len__(self):
        return len(self._values)

    def bisect_left(self, value):
        if isinstance(self._values, _ChunkedList):
            return self._values.bisect_left(value)
        return bisect_left(self._values, value)

    def bisect_right(self, value):
        if isinstance(self._values, _ChunkedList):
            return self._values.bisect_right(value)
        return bisect_right(self._values, value)

    bisect = bisect_right

    def bisect_many(self, values, right=False):
        """Return insertion indexes for many values, using Mojo for numeric data."""
        queries = list(values)
        source = (
            self._values.to_list()
            if isinstance(self._values, _ChunkedList)
            else self._values
        )
        native = bisect_many_numeric(source, queries, right)
        if native is not None:
            return native.tolist()
        fn = self.bisect_right if right else self.bisect_left
        return [fn(value) for value in queries]

    def count(self, value):
        return self.bisect_right(value) - self.bisect_left(value)

    def copy(self):
        return type(self)(list(self._values))

    __copy__ = copy

    def append(self, value):
        raise NotImplementedError("use ``sl.add(value)`` instead")

    def extend(self, values):
        raise NotImplementedError("use ``sl.update(values)`` instead")

    def insert(self, index, value):
        raise NotImplementedError("use ``sl.add(value)`` instead")

    def pop(self, index=-1):
        value = self._values.pop(index)
        self._invalidate()
        return value

    def index(self, value, start=None, stop=None):
        start = 0 if start is None else start
        stop = len(self) if stop is None else stop
        return self._values.index(value, start, stop)

    def __add__(self, other):
        return type(self)(list(self) + list(other))

    def __radd__(self, other):
        return type(self)(list(other) + list(self))

    def __iadd__(self, other):
        self.update(other)
        return self

    def __mul__(self, num):
        return type(self)(list(self._values) * num)

    __rmul__ = __mul__

    def __imul__(self, num):
        self._values = sorted(list(self._values) * num)
        self._invalidate()
        return self

    def _compare(self, other, operation):
        try:
            return operation(list(self._values), list(other))
        except TypeError:
            return NotImplemented

    def __eq__(self, other):
        return self._compare(other, eq)

    def __ne__(self, other):
        return self._compare(other, ne)

    def __lt__(self, other):
        return self._compare(other, lt)

    def __gt__(self, other):
        return self._compare(other, gt)

    def __le__(self, other):
        return self._compare(other, le)

    def __ge__(self, other):
        return self._compare(other, ge)

    def __repr__(self):
        return f"{type(self).__name__}({list(self._values)!r})"

    def _check(self):
        values = list(self._values)
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


class SortedKeyList(SortedList):
    """A sorted list ordered by the result of a key function."""

    def __new__(cls, iterable=None, key=identity):
        return object.__new__(cls)

    def __init__(self, iterable=None, key=identity):
        self._key = key
        self._values = [] if iterable is None else list(iterable)
        self._values.sort(key=key)
        self._keys = [key(value) for value in self._values]
        self._native_kind = None

    @property
    def key(self):
        return self._key

    def clear(self):
        self._values.clear()
        self._keys.clear()

    def add(self, value):
        key = self._key(value)
        pos = bisect_right(self._keys, key)
        self._keys.insert(pos, key)
        self._values.insert(pos, value)

    def update(self, iterable):
        incoming = list(iterable)
        if incoming:
            self._values.extend(incoming)
            self._values.sort(key=self._key)
            self._keys = [self._key(value) for value in self._values]

    def __contains__(self, value):
        key = self._key(value)
        left = bisect_left(self._keys, key)
        right = bisect_right(self._keys, key)
        return value in self._values[left:right]

    def discard(self, value):
        key = self._key(value)
        left = bisect_left(self._keys, key)
        right = bisect_right(self._keys, key)
        for pos in range(left, right):
            if self._values[pos] == value:
                del self._values[pos]
                del self._keys[pos]
                return

    def remove(self, value):
        before = len(self)
        self.discard(value)
        if len(self) == before:
            raise ValueError(f"{value!r} not in list")

    def __delitem__(self, index):
        del self._values[index]
        del self._keys[index]

    def pop(self, index=-1):
        del self._keys[index]
        return self._values.pop(index)

    def irange(self, minimum=None, maximum=None, inclusive=(True, True), reverse=False):
        min_key = None if minimum is None else self._key(minimum)
        max_key = None if maximum is None else self._key(maximum)
        return self.irange_key(min_key, max_key, inclusive, reverse)

    def irange_key(self, min_key=None, max_key=None, inclusive=(True, True), reverse=False):
        left = 0 if min_key is None else (
            bisect_left(self._keys, min_key) if inclusive[0]
            else bisect_right(self._keys, min_key)
        )
        right = len(self) if max_key is None else (
            bisect_right(self._keys, max_key) if inclusive[1]
            else bisect_left(self._keys, max_key)
        )
        values = self._values[left:right]
        return iter(values[::-1] if reverse else values)

    def bisect_left(self, value):
        return bisect_left(self._keys, self._key(value))

    def bisect_right(self, value):
        return bisect_right(self._keys, self._key(value))

    bisect = bisect_right

    def bisect_key_left(self, key):
        return bisect_left(self._keys, key)

    def bisect_key_right(self, key):
        return bisect_right(self._keys, key)

    bisect_key = bisect_key_right

    def count(self, value):
        key = self._key(value)
        return self._values[bisect_left(self._keys, key):bisect_right(self._keys, key)].count(value)

    def copy(self):
        return type(self)(self._values, key=self._key)

    __copy__ = copy

    def index(self, value, start=None, stop=None):
        start = 0 if start is None else start
        stop = len(self) if stop is None else stop
        return self._values.index(value, start, stop)

    def __add__(self, other):
        return type(self)(list(self) + list(other), key=self._key)

    def __radd__(self, other):
        return type(self)(list(other) + list(self), key=self._key)

    def __mul__(self, num):
        return type(self)(self._values * num, key=self._key)

    __rmul__ = __mul__

    def __iadd__(self, other):
        self.update(other)
        return self

    def __imul__(self, num):
        self._values *= num
        self._values.sort(key=self._key)
        self._keys = [self._key(value) for value in self._values]
        return self

    def __repr__(self):
        return f"{type(self).__name__}({self._values!r}, key={self._key!r})"

    def _check(self):
        assert self._keys == [self._key(value) for value in self._values]
        assert all(self._keys[i] <= self._keys[i + 1] for i in range(len(self) - 1))


SortedListWithKey = SortedKeyList
