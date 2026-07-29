"""ctypes bridge to the Mojo numeric bulk kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_SORTEDCONTAINERS_LIB") or os.path.join(
    ROOT, "dist", "libmojo-sortedcontainers.so"
)
I = ctypes.c_int64
P = ctypes.c_void_p
_PARALLEL_SORT_THRESHOLD = 4_194_304

_SIGNATURES = {
    "msc_sort_i64": ([P, I, P], I),
    "msc_sort_f64": ([P, I, P], I),
    "msc_merge_i64": ([P, I, P, I, P], I),
    "msc_merge_f64": ([P, I, P, I, P], I),
    "msc_bisect_i64": ([P, I, P, I, P, I], I),
    "msc_bisect_f64": ([P, I, P, I, P, I], I),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    if not os.path.exists(LIB):
        subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            check=True,
            cwd=ROOT,
        )
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray, dtype=None, *, writable=False) -> int:
    """Return an address only for an ABI-compatible one-dimensional array."""
    if not isinstance(array, np.ndarray) or array.ndim != 1:
        raise TypeError("native buffers must be one-dimensional NumPy arrays")
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise TypeError(f"expected dtype {np.dtype(dtype)}, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError("native buffers must be C-contiguous")
    if writable and not array.flags.writeable:
        raise ValueError("native output buffer must be writable")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise ValueError("non-empty native buffer has a null address")
    return address


def _call(fn, *args) -> None:
    status = fn(*args)
    if status != 0:
        raise RuntimeError(f"Mojo kernel rejected its buffers (status {status})")


def _all_i64(values) -> bool:
    if not all(type(value) is int for value in values):
        return False
    return min(values) >= -(1 << 63) and max(values) < (1 << 63)


def sort_numeric(values: list) -> tuple[list, str] | None:
    """Sort exact homogeneous int/float values in Mojo, or decline safely."""
    if not values:
        return [], "i64"
    kind = type(values[0])
    if kind is int and _all_i64(values):
        array = np.asarray(values, dtype=np.int64)
        scratch = np.empty_like(array) if len(array) >= _PARALLEL_SORT_THRESHOLD else None
        _call(lib().msc_sort_i64, addr(array, np.int64, writable=True), len(array),
              None if scratch is None else addr(scratch, np.int64, writable=True))
        return array.tolist(), "i64"
    if kind is float and all(type(value) is float for value in values):
        array = np.asarray(values, dtype=np.float64)
        if np.isnan(array).any():
            return None
        zeros = array == 0.0
        if zeros.any() and np.unique(np.signbit(array[zeros])).size > 1:
            return None
        scratch = np.empty_like(array) if len(array) >= _PARALLEL_SORT_THRESHOLD else None
        _call(lib().msc_sort_f64, addr(array, np.float64, writable=True), len(array),
              None if scratch is None else addr(scratch, np.float64, writable=True))
        return array.tolist(), "f64"
    return None


def merge_numeric(left: list, right: list, kind: str) -> list:
    dtype = np.int64 if kind == "i64" else np.float64
    a = np.asarray(left, dtype=dtype)
    b = np.asarray(right, dtype=dtype)
    result = np.empty(len(a) + len(b), dtype=dtype)
    fn = lib().msc_merge_i64 if kind == "i64" else lib().msc_merge_f64
    _call(fn, addr(a, dtype), len(a), addr(b, dtype), len(b),
          addr(result, dtype, writable=True))
    return result.tolist()


def bisect_many_numeric(values: list, queries, right: bool) -> np.ndarray | None:
    if not values:
        return np.zeros(len(queries), dtype=np.int64)
    kind = type(values[0])
    if kind is int:
        if not _all_i64(values):
            return None
        q = list(queries)
        if q and not _all_i64(q):
            return None
        array = np.asarray(values, dtype=np.int64)
        needles = np.asarray(q, dtype=np.int64)
        fn = lib().msc_bisect_i64
    elif kind is float:
        q = list(queries)
        if not all(type(value) is float for value in values) or not all(type(value) is float for value in q):
            return None
        array = np.asarray(values, dtype=np.float64)
        needles = np.asarray(q, dtype=np.float64)
        if np.isnan(array).any() or np.isnan(needles).any():
            return None
        fn = lib().msc_bisect_f64
    else:
        return None
    result = np.empty(len(needles), dtype=np.int64)
    if len(needles):
        _call(fn, addr(array, array.dtype), len(array),
              addr(needles, needles.dtype), len(needles),
              addr(result, np.int64, writable=True), int(right))
    return result
