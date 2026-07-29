# mojo-sortedcontainers

Sorted lists, dictionaries, and sets with Mojo kernels for numeric bulk work.

This is a tested subset of the public `sortedcontainers` API, not a drop-in
replacement. It keeps normal Python object semantics for the covered
`SortedList`, `SortedKeyList`, `SortedDict`, and `SortedSet` operations.
Homogeneous machine-sized integers and finite Python floats use compiled Mojo
for large sorts, sorted merges, and batched binary searches. Other comparable
Python objects and key functions use the Python path.

## Install

Installation is currently from a source checkout. A compiler is required;
there is no published wheel or PyPI release yet. The repository pins the Mojo
toolchain used to build the shared library:

```bash
pixi install
pixi run build
pixi run test
```

Use the checkout through the `pixi` environment; its `PYTHONPATH` already
includes `python/`.

## Usage

```python
from mojo_sortedcontainers import SortedDict, SortedList, SortedSet

scores = SortedList([8, 3, 5, 3])
scores.add(4)
assert list(scores) == [3, 3, 4, 5, 8]
assert list(scores.irange(3, 5)) == [3, 3, 4, 5]

# An extension for high-throughput search; one Mojo call handles every query.
assert scores.bisect_many([2, 4, 9]) == [0, 2, 5]

prices = SortedDict({"pear": 4, "apple": 2})
assert list(prices.items()) == [("apple", 2), ("pear", 4)]

tags = SortedSet(["release", "bug", "docs"])
assert list(tags) == ["bug", "docs", "release"]
```

## Covered API

The parity suite proves the following surface against upstream
`sortedcontainers`:

- `SortedList` construction, indexed/sliced access and deletion, add/update,
  discard/remove/pop, bisection, count/index, `islice`, `irange`, copying,
  comparisons, and sequence arithmetic.
- `SortedKeyList`, including key-based bisection and ranges.
- `SortedDict`, including sorted iteration, indexed keys/items/values views,
  `peekitem`, indexed `popitem`, updates, unions, and delegated range/search
  methods.
- `SortedSet`, including indexed access/deletion, sorted ranges, set
  comparisons, and all standard set-algebra/update operations.

Not covered: APIs absent from the list above, private attributes, load-factor
controls, internal node-index details, upstream performance characteristics,
and upstream's exact pickle representation. The implementation uses a simpler
internal index rather than porting upstream's block tree. `bisect_many` is an
intentional `SortedList` extension.

Large lists switch lazily to a block index for repeated insertion, while small
lists keep the lower-overhead contiguous representation.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz using
Python 3.13.14. Each cell is the best measured repeat from that single run;
lower is better.

| case | mojo-sortedcontainers | sortedcontainers | relative |
| --- | ---: | ---: | ---: |
| SortedList build, 400k ints | 105.0 ms | 113.7 ms | 1.08x faster |
| SortedList build + update, 300k + 200k | 180.4 ms | 241.1 ms | 1.34x faster |
| 100k lower bounds in 400k values | 123.2 ms | 276.8 ms | 2.25x faster |
| SortedList build + 5k adds | 135.8 ms | 247.3 ms | 1.82x faster |
| SortedDict build + ordered iteration, 150k | 154.1 ms | 187.1 ms | 1.21x faster |
| SortedSet union, 300k + 200k | 353.9 ms | 366.7 ms | 1.04x faster |

There is no GPU path. Sorting, merging, and binary search are comparison- and
memory-bound, with too little arithmetic per byte moved to justify device
transfer and launch overhead.

## How it works

Python owns all storage and object lifetime. Numeric bulk paths copy exact,
homogeneous Python integers or floats into one-dimensional, C-contiguous NumPy
arrays with an exact native dtype. The wrapper validates dtype, shape, stride,
writeability, and non-null addresses before each call and keeps every array
referenced until the call returns. The single Mojo compilation unit reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values and performs in-place span
sorting, SIMD remainder copies during linear merges, or a batch of binary
searches. Exports reject invalid lengths and required null pointers; the Python
wrapper turns a nonzero status into an exception. Large sorts use bounded CPU
workers and a NumPy-owned scratch buffer. No Mojo allocation crosses the ABI.

Large repeated insertions use the block index so each operation moves a small
block rather than the entire Python list. Set union sorts the already
deduplicated result directly, avoiding temporary NumPy buffers and redundant
full-result validation.

Generic values remain Python objects. A `SortedList` stores them in a sorted
sequence index; `SortedDict` combines the built-in hash table with a sorted key
list; `SortedSet` combines a built-in set with the same sorted index.
Values that cannot be represented without changing Python behavior—large
integers, NaNs, mixed types, keyed objects, and signed-zero mixtures—stay on
the Python path.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

`pixi run bench` holds a machine-wide lock so concurrent benchmark jobs do not
distort the measurements. The shared library is emitted at
`dist/libmojo-sortedcontainers.so`.

MIT licensed.
