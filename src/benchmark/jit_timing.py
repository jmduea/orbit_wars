"""Shared JIT microbench timing for ``src.benchmark`` modules."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import jax


@dataclass(frozen=True, slots=True)
class TimingStats:
    compile_seconds: float
    mean_seconds: float
    std_seconds: float
    min_seconds: float
    max_seconds: float


@dataclass(frozen=True, slots=True)
class EncodeTimingStats(TimingStats):
    encodes_per_call: int
    mean_seconds_per_encode: float
    encodes_per_second: float


def _sync_output(out: object) -> None:
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)


def measure_jitted(fn, *, warmup: int, repeats: int) -> TimingStats:
    compile_start = time.perf_counter()
    out = fn()
    _sync_output(out)
    compile_seconds = time.perf_counter() - compile_start

    for _ in range(warmup):
        out = fn()
        _sync_output(out)

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        _sync_output(out)
        timings.append(time.perf_counter() - start)

    mean_s = statistics.mean(timings)
    std_s = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    return TimingStats(
        compile_seconds=compile_seconds,
        mean_seconds=mean_s,
        std_seconds=std_s,
        min_seconds=min(timings),
        max_seconds=max(timings),
    )


def measure_jitted_encodes(
    fn,
    *,
    warmup: int,
    repeats: int,
    encodes_per_call: int,
) -> EncodeTimingStats:
    base = measure_jitted(fn, warmup=warmup, repeats=repeats)
    per_encode = base.mean_seconds / encodes_per_call
    return EncodeTimingStats(
        compile_seconds=base.compile_seconds,
        mean_seconds=base.mean_seconds,
        std_seconds=base.std_seconds,
        min_seconds=base.min_seconds,
        max_seconds=base.max_seconds,
        encodes_per_call=encodes_per_call,
        mean_seconds_per_encode=per_encode,
        encodes_per_second=(encodes_per_call / base.mean_seconds)
        if base.mean_seconds > 0
        else 0.0,
    )
