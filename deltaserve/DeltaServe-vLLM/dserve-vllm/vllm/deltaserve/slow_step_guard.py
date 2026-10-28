"""[pause_on_slow_step_ms] Pause the backward child while an inference step is late.

The runner brackets every inference-bearing forward issued while a backward is
outstanding with ``begin()`` (before its pause decision) and ``end(tok, event)``
(an event recorded behind the forward). A daemon thread polls the in-flight
steps; when the oldest has been in flight longer than the threshold it clears
the child's grant and sets ``coord.guard_hold``, which turns every other
``gpu_resume_backward`` into a no-op. Once no step is in flight it drops the hold
and resumes the child unless another pause owner still holds it (the
prefill-completion event or the forward-token throttle; ``owner_busy``).

Under TP each rank runs its own guard; the child's ``_maybe_pause`` agreement
(MAX-all-reduce of "any grant cleared") stops every rank at the same boundary,
so a trigger on one rank pauses them all.
"""
import collections
import threading
import time

import torch


class SlowStepGuard:

    def __init__(self, coord, threshold_s, owner_busy, poll_s=0.002):
        self.coord = coord
        self.threshold_s = float(threshold_s)
        self.owner_busy = owner_busy   # () -> True while another pause owner holds
        self.poll_s = poll_s
        self.triggers = 0
        self._lock = threading.Lock()
        self._inflight = collections.OrderedDict()   # tok -> [t0, event | None]
        self._next = 0
        self._device = torch.cuda.current_device()
        threading.Thread(target=self._run, name="dserve-slow-step-guard",
                         daemon=True).start()

    def begin(self):
        with self._lock:
            tok = self._next
            self._next += 1
            self._inflight[tok] = [time.perf_counter(), None]
        return tok

    def end(self, tok, event):
        """``event`` recorded behind the forward; None = nothing reached the GPU."""
        with self._lock:
            if event is None:
                self._inflight.pop(tok, None)
            elif tok in self._inflight:
                self._inflight[tok][1] = event

    def _oldest_pending(self):
        with self._lock:
            while self._inflight:
                tok, (t0, ev) = next(iter(self._inflight.items()))
                if ev is not None and ev.query():
                    del self._inflight[tok]
                    continue
                return t0
        return None

    def _run(self):
        torch.cuda.set_device(self._device)
        coord = self.coord
        while True:
            time.sleep(self.poll_s)
            try:
                t0 = self._oldest_pending()
                if t0 is not None and time.perf_counter() - t0 > self.threshold_s:
                    if not coord.guard_hold:
                        coord.guard_hold = True
                        self.triggers += 1
                    # Re-assert every poll: a resume that raced the hold is undone.
                    coord.backward_process.set_pause(True)
                elif t0 is None and coord.guard_hold:
                    coord.guard_hold = False
                    if not self.owner_busy():
                        coord.backward_process.set_pause(False)
            except Exception:  # never take the worker down; the child's 5 s cap bounds a stuck pause
                pass
