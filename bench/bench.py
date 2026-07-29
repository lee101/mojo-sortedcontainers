"""Honest timings against sortedcontainers on identical inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
from sortedcontainers import (
    SortedDict as RefSortedDict,
    SortedList as RefSortedList,
    SortedSet as RefSortedSet,
)

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from mojo_sortedcontainers import SortedDict, SortedList, SortedSet  # noqa: E402


def timeit(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    rng = np.random.default_rng(42)
    build_values = rng.integers(-10_000_000, 10_000_000, 400_000).tolist()
    base_values = rng.integers(-5_000_000, 5_000_000, 300_000).tolist()
    new_values = rng.integers(-5_000_000, 5_000_000, 200_000).tolist()
    queries = rng.integers(-12_000_000, 12_000_000, 100_000).tolist()
    add_values = rng.integers(-10_000_000, 10_000_000, 5_000).tolist()
    mapping = {int(value): index for index, value in enumerate(build_values[:150_000])}

    mojo_search = SortedList(build_values)
    ref_search = RefSortedList(build_values)
    assert mojo_search.bisect_many(queries) == [
        ref_search.bisect_left(value) for value in queries
    ]

    cases = [
        (
            "SortedList build, 400k ints",
            lambda: SortedList(build_values),
            lambda: RefSortedList(build_values),
            3,
        ),
        (
            "SortedList build + update, 300k + 200k",
            lambda: SortedList(base_values).update(new_values),
            lambda: RefSortedList(base_values).update(new_values),
            3,
        ),
        (
            "100k lower bounds in 400k values",
            lambda: mojo_search.bisect_many(queries),
            lambda: [ref_search.bisect_left(value) for value in queries],
            5,
        ),
        (
            "SortedList build + 5k adds",
            lambda: _add_many(SortedList(build_values), add_values),
            lambda: _add_many(RefSortedList(build_values), add_values),
            3,
        ),
        (
            "SortedDict build + ordered iteration, 150k",
            lambda: sum(SortedDict(mapping).values()),
            lambda: sum(RefSortedDict(mapping).values()),
            3,
        ),
        (
            "SortedSet union, 300k + 200k",
            lambda: SortedSet(base_values).union(new_values),
            lambda: RefSortedSet(base_values).union(new_values),
            3,
        ),
    ]

    print(f"Machine: {cpu_name()}; Python {platform.python_version()}")
    print()
    print("| case | mojo-sortedcontainers | sortedcontainers | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, reference, repeat in cases:
        ours()
        a = timeit(ours, repeat)
        b = timeit(reference, repeat)
        word = "faster" if a < b else "slower"
        factor = max(a, b) / min(a, b)
        print(
            f"| {name} | {a * 1e3:.1f} ms | {b * 1e3:.1f} ms | "
            f"{factor:.2f}x {word} |"
        )


def _add_many(container, values):
    for value in values:
        container.add(value)
    return container


if __name__ == "__main__":
    main()
