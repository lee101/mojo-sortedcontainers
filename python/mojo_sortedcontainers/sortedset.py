"""Set with deterministic sorted iteration and sequence-style indexing."""

from __future__ import annotations

from collections.abc import MutableSet, Set

from .sortedlist import SortedList


class SortedSet(MutableSet):
    """A mutable set whose values are also kept in a sorted list."""

    def __init__(self, iterable=None, key=None):
        self._key = key
        self._set = set() if iterable is None else set(iterable)
        self._reset_list()

    def _reset_list(self):
        self._list = SortedList(self._set, key=self._key)
        if self._key is not None:
            self.bisect_key_left = self._list.bisect_key_left
            self.bisect_key_right = self._list.bisect_key_right
            self.bisect_key = self._list.bisect_key
            self.irange_key = self._list.irange_key

    @classmethod
    def _fromset(cls, values, key=None):
        return cls(values, key=key)

    @property
    def key(self):
        return self._key

    def __contains__(self, value):
        return value in self._set

    def __getitem__(self, index):
        return self._list[index]

    def __delitem__(self, index):
        values = self._list[index]
        if isinstance(index, slice):
            del self._list[index]
            self._set.difference_update(values)
        else:
            del self._list[index]
            self._set.remove(values)

    def __eq__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        return self._set == set(other)

    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        return self._set < set(other)

    def __gt__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        return self._set > set(other)

    def __le__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        return self._set <= set(other)

    def __ge__(self, other):
        if not isinstance(other, Set):
            return NotImplemented
        return self._set >= set(other)

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        return iter(self._list)

    def __reversed__(self):
        return reversed(self._list)

    def add(self, value):
        if value not in self._set:
            self._set.add(value)
            self._list.add(value)

    def clear(self):
        self._set.clear()
        self._list.clear()

    def copy(self):
        return type(self)(self, key=self._key)

    __copy__ = copy

    def count(self, value):
        return int(value in self._set)

    def discard(self, value):
        if value in self._set:
            self._set.remove(value)
            self._list.remove(value)

    def pop(self, index=-1):
        value = self._list.pop(index)
        self._set.remove(value)
        return value

    def remove(self, value):
        self._set.remove(value)
        self._list.remove(value)

    def difference(self, *iterables):
        return type(self)._fromset(self._set.difference(*iterables), self._key)

    def __sub__(self, *iterables):
        return self.difference(*iterables)

    def __isub__(self, other):
        self.difference_update(other)
        return self

    def difference_update(self, *iterables):
        self._set.difference_update(*iterables)
        self._reset_list()
        return self

    def intersection(self, *iterables):
        return type(self)._fromset(self._set.intersection(*iterables), self._key)

    def __and__(self, *iterables):
        return self.intersection(*iterables)

    __rand__ = __and__

    def __iand__(self, other):
        self.intersection_update(other)
        return self

    def intersection_update(self, *iterables):
        self._set.intersection_update(*iterables)
        self._reset_list()
        return self

    def symmetric_difference(self, other):
        return type(self)._fromset(self._set.symmetric_difference(other), self._key)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    __rxor__ = __xor__

    def __ixor__(self, other):
        self.symmetric_difference_update(other)
        return self

    def symmetric_difference_update(self, other):
        self._set.symmetric_difference_update(other)
        self._reset_list()
        return self

    def union(self, *iterables):
        values = self._set.union(*iterables)
        if type(self) is SortedSet and self._key is None and self._set:
            result = object.__new__(SortedSet)
            result._key = None
            result._set = values
            result._list = object.__new__(SortedList)
            result._list._values = sorted(values)
            result._list._native_kind = None
            return result
        return type(self)._fromset(values, self._key)

    def __or__(self, *iterables):
        return self.union(*iterables)

    __ror__ = __or__

    def __ior__(self, other):
        self.update(other)
        return self

    def update(self, *iterables):
        values = set()
        for iterable in iterables:
            values.update(iterable)
        new_values = values - self._set
        if new_values:
            self._set.update(new_values)
            self._list.update(new_values)
        return self

    def bisect_left(self, value):
        return self._list.bisect_left(value)

    def bisect_right(self, value):
        return self._list.bisect_right(value)

    bisect = bisect_right

    def index(self, value, start=None, stop=None):
        return self._list.index(value, start, stop)

    def irange(self, minimum=None, maximum=None, inclusive=(True, True), reverse=False):
        return self._list.irange(minimum, maximum, inclusive, reverse)

    def islice(self, start=None, stop=None, reverse=False):
        return self._list.islice(start, stop, reverse)

    def _check(self):
        assert self._set == set(self._list)
        self._list._check()

    def __repr__(self):
        if not self:
            body = "()"
        else:
            body = repr(list(self))
        if self._key is None:
            return f"{type(self).__name__}({body})"
        return f"{type(self).__name__}({body}, key={self._key!r})"
