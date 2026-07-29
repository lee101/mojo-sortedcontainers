"""Dictionary whose iteration order follows sorted keys."""

from __future__ import annotations

from collections.abc import ItemsView, KeysView, ValuesView

from .sortedlist import SortedList

_MISSING = object()


class SortedKeysView(KeysView):
    def __getitem__(self, index):
        return self._mapping._list[index]

    def __reversed__(self):
        return reversed(self._mapping)


class SortedItemsView(ItemsView):
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [(key, self._mapping[key]) for key in self._mapping._list[index]]
        key = self._mapping._list[index]
        return key, self._mapping[key]

    def __iter__(self):
        for key in self._mapping:
            yield key, self._mapping[key]

    def __reversed__(self):
        for key in reversed(self._mapping):
            yield key, self._mapping[key]


class SortedValuesView(ValuesView):
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._mapping[key] for key in self._mapping._list[index]]
        return self._mapping[self._mapping._list[index]]

    def __iter__(self):
        for key in self._mapping:
            yield self._mapping[key]

    def __reversed__(self):
        for key in reversed(self._mapping):
            yield self._mapping[key]


class SortedDict(dict):
    """A dict with keys maintained in sorted order."""

    def __init__(self, *args, **kwargs):
        key = None
        if args and callable(args[0]):
            key, args = args[0], args[1:]
        if len(args) > 1:
            raise TypeError(f"expected at most 1 arguments, got {len(args)}")
        dict.__init__(self)
        if args:
            dict.update(self, args[0])
        if kwargs:
            dict.update(self, kwargs)
        self._key = key
        self._list = SortedList(dict.keys(self), key=key)
        if key is not None:
            self.bisect_key_left = self._list.bisect_key_left
            self.bisect_key_right = self._list.bisect_key_right
            self.bisect_key = self._list.bisect_key
            self.irange_key = self._list.irange_key

    @property
    def key(self):
        return self._key

    @property
    def iloc(self):
        return self._list

    def clear(self):
        dict.clear(self)
        self._list.clear()

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self._list.remove(key)

    def __iter__(self):
        return iter(self._list)

    def __reversed__(self):
        return reversed(self._list)

    def __setitem__(self, key, value):
        if key not in self:
            self._list.add(key)
        dict.__setitem__(self, key, value)

    def __or__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        result = self.copy()
        result.update(other)
        return result

    def __ror__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        result = type(self)(self._key, other) if self._key else type(self)(other)
        result.update(self)
        return result

    def __ior__(self, other):
        self.update(other)
        return self

    def copy(self):
        return type(self)(self._key, self) if self._key else type(self)(self)

    __copy__ = copy

    @classmethod
    def fromkeys(cls, iterable, value=None):
        return cls((key, value) for key in iterable)

    def keys(self):
        return SortedKeysView(self)

    def items(self):
        return SortedItemsView(self)

    def values(self):
        return SortedValuesView(self)

    def pop(self, key, default=_MISSING):
        if key in self:
            self._list.remove(key)
            return dict.pop(self, key)
        if default is _MISSING:
            raise KeyError(key)
        return default

    def popitem(self, index=-1):
        if not self:
            raise KeyError("popitem(): dictionary is empty")
        key = self._list.pop(index)
        return key, dict.pop(self, key)

    def peekitem(self, index=-1):
        if not self:
            raise IndexError("list index out of range")
        key = self._list[index]
        return key, self[key]

    def setdefault(self, key, default=None):
        if key not in self:
            self._list.add(key)
            dict.__setitem__(self, key, default)
            return default
        return dict.__getitem__(self, key)

    def update(self, *args, **kwargs):
        if len(args) > 1:
            raise TypeError(f"update expected at most 1 argument, got {len(args)}")
        incoming = dict(args[0]) if args else {}
        incoming.update(kwargs)
        new_keys = incoming.keys() - dict.keys(self)
        if new_keys:
            self._list.update(new_keys)
        dict.update(self, incoming)

    def bisect_left(self, key):
        return self._list.bisect_left(key)

    def bisect_right(self, key):
        return self._list.bisect_right(key)

    bisect = bisect_right

    def index(self, key, start=None, stop=None):
        return self._list.index(key, start, stop)

    def irange(self, minimum=None, maximum=None, inclusive=(True, True), reverse=False):
        return self._list.irange(minimum, maximum, inclusive, reverse)

    def islice(self, start=None, stop=None, reverse=False):
        return self._list.islice(start, stop, reverse)

    def _check(self):
        assert set(self._list) == set(dict.keys(self))
        self._list._check()

    def __repr__(self):
        body = "{" + ", ".join(f"{key!r}: {self[key]!r}" for key in self) + "}"
        if self._key is None:
            return f"{type(self).__name__}({body})"
        return f"{type(self).__name__}({self._key!r}, {body})"
