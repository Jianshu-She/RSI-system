#!/usr/bin/env python
"""pause_on_slow_step_ms: SlowStepGuard pauses the child while a step is late.

GPU (one device). A fake backward process records set_pause calls; a "step" is a
torch.cuda._sleep kernel with an event behind it. Checks: a fast step never
pauses; a slow step pauses after the threshold and releases once it completes;
the hold blocks the coordinator's other resumes; a busy owner keeps the child
paused on release; a dispatch with no event (failed forward) is dropped.

    python tests/test_slow_step_guard.py
"""
import time

import torch

from vllm.deltaserve.coordinator import FinetuneCoordinator
from vllm.deltaserve.slow_step_guard import SlowStepGuard

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    PASS += bool(ok)
    FAIL += not ok
    print(f"  [{'ok' if ok else 'FAIL'}] {name} {detail}")


class FakeBwd:
    def __init__(self):
        self.paused = False
        self.calls = []

    def set_pause(self, p):
        self.paused = p
        self.calls.append((time.perf_counter(), p))


def cycles_per_ms():
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    torch.cuda._sleep(10_000_000)
    e.record()
    e.synchronize()
    return 10_000_000 / s.elapsed_time(e)


def step(guard, ms, cpm):
    tok = guard.begin()
    torch.cuda._sleep(int(ms * cpm))
    ev = torch.cuda.Event()
    ev.record()
    guard.end(tok, ev)
    return ev


def setup(busy=lambda: False):
    coord = FinetuneCoordinator(16)
    coord.backward_process = FakeBwd()
    return coord, SlowStepGuard(coord, 0.05, owner_busy=busy)


def main():
    cycles_per_ms()  # warm up
    cpm = cycles_per_ms()
    print("fast step (10 ms, threshold 50 ms):")
    coord, g = setup()
    step(g, 10, cpm).synchronize()
    time.sleep(0.03)
    check("never paused", not coord.backward_process.calls and g.triggers == 0)

    print("slow step (300 ms):")
    coord, g = setup()
    t0 = time.perf_counter()
    ev = step(g, 300, cpm)
    time.sleep(0.15)
    check("step still running at 150 ms", not ev.query())
    bwd = coord.backward_process
    first = next((t for t, p in bwd.calls if p), None)
    check("paused after the threshold", first is not None and 0.045 < first - t0 < 0.08,
          f"at {1e3 * (first - t0):.0f} ms" if first else "")
    check("hold set", coord.guard_hold and bwd.paused and g.triggers == 1)
    coord.gpu_resume_backward()
    check("other resume blocked while held", bwd.paused)
    ev.synchronize()
    time.sleep(0.02)
    check("released after completion", not coord.guard_hold and not bwd.paused)
    coord.gpu_resume_backward()
    check("resume works again", not bwd.paused)

    print("release with a busy owner:")
    busy = [True]
    coord, g = setup(lambda: busy[0])
    step(g, 150, cpm).synchronize()
    time.sleep(0.02)
    check("hold dropped, child stays paused", not coord.guard_hold and coord.backward_process.paused)

    print("failed forward (no event):")
    coord, g = setup()
    tok = g.begin()
    g.end(tok, None)
    time.sleep(0.08)
    check("dropped, never paused", not coord.backward_process.calls)

    print("in-flight dispatch counts as late:")
    coord, g = setup()
    tok = g.begin()
    time.sleep(0.08)
    check("pending dispatch pauses", coord.backward_process.paused)
    g.end(tok, None)
    time.sleep(0.02)
    check("released", not coord.backward_process.paused)
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
