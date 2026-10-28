"""Average-TBT gate (slo.ft_decode_stall_budget) in FinetuneScheduler.admit_ft_to_step.

CPU only. A stub estimator predicts a decode step at 10 ms and adds 0.1 ms per FT
token, so one 500-token sample on a decode step stalls the decodes 50 ms. With a
0.1 budget over a 1 s window FT may add 100 ms of stall per second: two such
steps are admitted, the third waits until the window slides. Budget 0 is the old
behaviour (no gate); prefill-only and idle steps (no decodes) are never gated.

    python tests/test_ft_stall_gate.py
"""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vllm.config.finetune import FinetuneConfig
from vllm.deltaserve.coordinator import FinetuneCoordinator
from vllm.deltaserve.estimator import StepFeatures
from vllm.deltaserve.ft_scheduler import FinetuneScheduler


class _Store:
    def __init__(self, n, length=500):
        self.pool = [SimpleNamespace(input_len=length, request_id=f"s{i}") for i in range(n)]
        self.current_epoch, self.total_epochs = 0, 1

    def has_next(self):
        return bool(self.pool)

    def advance_epoch(self):
        return False

    def pop_next(self, exclude=()):
        ids = {id(s) for s in exclude}
        return next((s for s in self.pool if id(s) not in ids), None)

    def claim(self, samples):
        ids = {id(s) for s in samples}
        self.pool = [s for s in self.pool if id(s) not in ids]


class _Est:
    is_ready = True

    def predict(self, f, regime=None, apply_margin=True):
        return 0.01 + 1e-4 * f.t_ft


def make(budget, window=1.0, b_d=4, t_in=0):
    s = object.__new__(FinetuneScheduler)
    cfg = FinetuneConfig(enable_finetuning=True, coserving_admission_phase="both",
                         max_tbt_slo=0.2, decode_only_ft_safety_margin=1.0,
                         ft_decode_stall_budget=budget, ft_stall_window_s=window)
    s.vllm_config = SimpleNamespace(finetune_config=cfg)
    coord = FinetuneCoordinator(16384)
    coord.ft_started = True
    s._coord = coord
    s._ft_injector = SimpleNamespace(store=_Store(50), _make_request=lambda smp: smp)
    s._estimator = _Est()
    s._admit_trace = None
    s._async = False
    s._last_step_pred_s = 0.0
    s._ttft_slo, s._max_tbt_slo = 0.5, 0.2
    s._stall_budget, s._stall_window = budget, window
    from collections import deque
    s._stalls = deque()
    s.max_num_scheduled_tokens = 16384
    s._unspent_prefill = 0
    s._profiling_mode = False
    feats = StepFeatures(t_in=t_in, p=1 if t_in else 0, b_d=b_d, k=4000.0,
                         prefill_lens=[t_in] if t_in else None)
    s._current_step_features = lambda: (feats, None)
    return s


def admit(s, at):
    with patch("vllm.deltaserve.ft_scheduler.time.time", return_value=at):
        n, _ = s.admit_ft_to_step()
    s._coord.fill_count = 0  # the test only exercises admission, not the buffer
    return n


class StallGateTest(unittest.TestCase):
    def test_budget_limits_stall_per_window(self):
        s = make(budget=0.1)
        # Step 1 admits two samples (100 ms of stall, the whole budget); step 2,
        # 10 ms later, admits nothing.
        self.assertEqual(admit(s, 100.0), 1000)
        self.assertAlmostEqual(sum(x for _, x in s._stalls), 0.1)
        self.assertEqual(admit(s, 100.01), 0, "budget spent")
        # The window slides: 1.5 s later FT is admitted again.
        self.assertEqual(admit(s, 101.5), 1000)

    def test_zero_budget_is_ungated(self):
        s = make(budget=0.0)
        got = [admit(s, 100.0 + 0.01 * i) for i in range(5)]
        self.assertTrue(all(n > 0 for n in got))
        self.assertEqual(len(s._stalls), 0)

    def test_no_decodes_not_gated(self):
        s = make(budget=0.01, b_d=0, t_in=300)
        # A 10 ms budget would stop at the first sample on a decode step; with no
        # decodes the step fills the token budget and records no stall.
        self.assertEqual(admit(s, 100.0), 16000)
        self.assertEqual(len(s._stalls), 0)

    def test_single_step_bounded_by_budget(self):
        # 0.1 * 1 s = 100 ms: at most two 500-token samples (50 ms each) in one step.
        s = make(budget=0.1)
        self.assertEqual(admit(s, 100.0), 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
