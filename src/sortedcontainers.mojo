"""Numeric bulk kernels for mojo-sortedcontainers.

Python owns every buffer. The exported functions only sort, merge, or search
contiguous int64 and float64 arrays supplied by the wrapper.
"""

from std.runtime.asyncrt import TaskGroup, initialize_runtime, parallelism_level
from std.sys import simd_width_of

comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]


@always_inline
def parallelize[
    origins: OriginSet, //, func: def(Int) capturing[origins] -> None
](num_work_items: Int, max_workers: Int):
    var num_workers = min(num_work_items, min(max_workers, parallelism_level()))
    if num_workers <= 1:
        for i in range(num_work_items):
            func(i)
        return
    var chunk_size, extra_items = divmod(num_work_items, num_workers)

    async def work_chunk(worker: Int, chunk_size: Int, extra_items: Int):
        var begin = worker * chunk_size + min(worker, extra_items)
        var count = chunk_size + Int(worker < extra_items)
        for i in range(begin, begin + count):
            func(i)

    var tasks = TaskGroup()
    for worker in range(num_workers):
        tasks.create_task(work_chunk(worker, chunk_size, extra_items))
    tasks.wait()


def sort_i64(values: IPtr, n: Int):
    if n < 2:
        return
    sort(Span(unsafe_ptr=values, length=n))


def sort_f64(values: FPtr, n: Int):
    if n < 2:
        return
    sort(Span(unsafe_ptr=values, length=n))


def copy_i64(src: IPtr, src_pos: Int, count: Int, dst: IPtr, dst_pos: Int):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= count:
        var values = src.load[width=W](src_pos + i)
        dst.store(dst_pos + i, values)
        i += W
    while i < count:
        dst[dst_pos + i] = src[src_pos + i]
        i += 1


def copy_f64(src: FPtr, src_pos: Int, count: Int, dst: FPtr, dst_pos: Int):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= count:
        var values = src.load[width=W](src_pos + i)
        dst.store(dst_pos + i, values)
        i += W
    while i < count:
        dst[dst_pos + i] = src[src_pos + i]
        i += 1


def merge_i64(a: IPtr, na: Int, b: IPtr, nb: Int, dst: IPtr):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var j = 0
    var k = 0
    while i < na and j < nb:
        if i + W <= na and a[i + W - 1] <= b[j]:
            dst.store(k, a.load[width=W](i))
            i += W
            k += W
            continue
        if j + W <= nb and b[j + W - 1] < a[i]:
            dst.store(k, b.load[width=W](j))
            j += W
            k += W
            continue
        if b[j] < a[i]:
            dst[k] = b[j]
            j += 1
        else:
            dst[k] = a[i]
            i += 1
        k += 1
    copy_i64(a, i, na - i, dst, k)
    k += na - i
    copy_i64(b, j, nb - j, dst, k)


def merge_f64(a: FPtr, na: Int, b: FPtr, nb: Int, dst: FPtr):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    var j = 0
    var k = 0
    while i < na and j < nb:
        if i + W <= na and a[i + W - 1] <= b[j]:
            dst.store(k, a.load[width=W](i))
            i += W
            k += W
            continue
        if j + W <= nb and b[j + W - 1] < a[i]:
            dst.store(k, b.load[width=W](j))
            j += W
            k += W
            continue
        if b[j] < a[i]:
            dst[k] = b[j]
            j += 1
        else:
            dst[k] = a[i]
            i += 1
        k += 1
    copy_f64(a, i, na - i, dst, k)
    k += na - i
    copy_f64(b, j, nb - j, dst, k)


def parallel_sort_i64(values: IPtr, n: Int, scratch: IPtr):
    comptime TASKS = 4

    @parameter
    def sort_chunk(task: Int):
        var start = task * n // TASKS
        var end = (task + 1) * n // TASKS
        sort_i64(values + start, end - start)

    parallelize[sort_chunk](TASKS, TASKS)
    var middle = n // 2
    var first_split = n // TASKS
    var second_split = 3 * n // TASKS

    @parameter
    def merge_pair(pair: Int):
        if pair == 0:
            merge_i64(
                values, first_split,
                values + first_split, middle - first_split,
                scratch,
            )
        else:
            merge_i64(
                values + middle, second_split - middle,
                values + second_split, n - second_split,
                scratch + middle,
            )

    parallelize[merge_pair](2, 2)
    merge_i64(scratch, middle, scratch + middle, n - middle, values)


def parallel_sort_f64(values: FPtr, n: Int, scratch: FPtr):
    comptime TASKS = 4

    @parameter
    def sort_chunk(task: Int):
        var start = task * n // TASKS
        var end = (task + 1) * n // TASKS
        sort_f64(values + start, end - start)

    parallelize[sort_chunk](TASKS, TASKS)
    var middle = n // 2
    var first_split = n // TASKS
    var second_split = 3 * n // TASKS

    @parameter
    def merge_pair(pair: Int):
        if pair == 0:
            merge_f64(
                values, first_split,
                values + first_split, middle - first_split,
                scratch,
            )
        else:
            merge_f64(
                values + middle, second_split - middle,
                values + second_split, n - second_split,
                scratch + middle,
            )

    parallelize[merge_pair](2, 2)
    merge_f64(scratch, middle, scratch + middle, n - middle, values)


def lower_i64(values: IPtr, n: Int, needle: Int64) -> Int64:
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if values[mid] < needle:
            lo = mid + 1
        else:
            hi = mid
    return Int64(lo)


def upper_i64(values: IPtr, n: Int, needle: Int64) -> Int64:
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if needle < values[mid]:
            hi = mid
        else:
            lo = mid + 1
    return Int64(lo)


def lower_f64(values: FPtr, n: Int, needle: Float64) -> Int64:
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if values[mid] < needle:
            lo = mid + 1
        else:
            hi = mid
    return Int64(lo)


def upper_f64(values: FPtr, n: Int, needle: Float64) -> Int64:
    var lo = 0
    var hi = n
    while lo < hi:
        var mid = lo + (hi - lo) // 2
        if needle < values[mid]:
            hi = mid
        else:
            lo = mid + 1
    return Int64(lo)


@export("msc_sort_i64")
def msc_sort_i64(addr: Int, n: Int, scratch: Int) abi("C") -> Int:
    if n < 0 or (n > 0 and addr == 0):
        return -1
    if n == 0:
        return 0
    var values = IPtr(unsafe_from_address=addr)
    if scratch != 0:
        initialize_runtime()
        parallel_sort_i64(values, n, IPtr(unsafe_from_address=scratch))
    else:
        sort_i64(values, n)
    return 0


@export("msc_sort_f64")
def msc_sort_f64(addr: Int, n: Int, scratch: Int) abi("C") -> Int:
    if n < 0 or (n > 0 and addr == 0):
        return -1
    if n == 0:
        return 0
    var values = FPtr(unsafe_from_address=addr)
    if scratch != 0:
        initialize_runtime()
        parallel_sort_f64(values, n, FPtr(unsafe_from_address=scratch))
    else:
        sort_f64(values, n)
    return 0


@export("msc_merge_i64")
def msc_merge_i64(a: Int, na: Int, b: Int, nb: Int, dst: Int) abi("C") -> Int:
    if na < 0 or nb < 0:
        return -1
    if (na > 0 and a == 0) or (nb > 0 and b == 0):
        return -1
    if (na > 0 or nb > 0) and dst == 0:
        return -1
    if na == 0 and nb == 0:
        return 0
    merge_i64(
        IPtr(unsafe_from_address=a), na,
        IPtr(unsafe_from_address=b), nb,
        IPtr(unsafe_from_address=dst),
    )
    return 0


@export("msc_merge_f64")
def msc_merge_f64(a: Int, na: Int, b: Int, nb: Int, dst: Int) abi("C") -> Int:
    if na < 0 or nb < 0:
        return -1
    if (na > 0 and a == 0) or (nb > 0 and b == 0):
        return -1
    if (na > 0 or nb > 0) and dst == 0:
        return -1
    if na == 0 and nb == 0:
        return 0
    merge_f64(
        FPtr(unsafe_from_address=a), na,
        FPtr(unsafe_from_address=b), nb,
        FPtr(unsafe_from_address=dst),
    )
    return 0


@export("msc_bisect_i64")
def msc_bisect_i64(
    values: Int, n: Int, queries: Int, m: Int, dst: Int, right: Int
) abi("C") -> Int:
    if n < 0 or m < 0:
        return -1
    if (n > 0 and values == 0) or (m > 0 and (queries == 0 or dst == 0)):
        return -1
    if m == 0:
        return 0
    if n == 0:
        var empty_result = IPtr(unsafe_from_address=dst)
        for i in range(m):
            empty_result[i] = 0
        return 0
    var v = IPtr(unsafe_from_address=values)
    var q = IPtr(unsafe_from_address=queries)
    var result = IPtr(unsafe_from_address=dst)
    for i in range(m):
        if right != 0:
            result[i] = upper_i64(v, n, q[i])
        else:
            result[i] = lower_i64(v, n, q[i])
    return 0


@export("msc_bisect_f64")
def msc_bisect_f64(
    values: Int, n: Int, queries: Int, m: Int, dst: Int, right: Int
) abi("C") -> Int:
    if n < 0 or m < 0:
        return -1
    if (n > 0 and values == 0) or (m > 0 and (queries == 0 or dst == 0)):
        return -1
    if m == 0:
        return 0
    if n == 0:
        var empty_result = IPtr(unsafe_from_address=dst)
        for i in range(m):
            empty_result[i] = 0
        return 0
    var v = FPtr(unsafe_from_address=values)
    var q = FPtr(unsafe_from_address=queries)
    var result = IPtr(unsafe_from_address=dst)
    for i in range(m):
        if right != 0:
            result[i] = upper_f64(v, n, q[i])
        else:
            result[i] = lower_f64(v, n, q[i])
    return 0
