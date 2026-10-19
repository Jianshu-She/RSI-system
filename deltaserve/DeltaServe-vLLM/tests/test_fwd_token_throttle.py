"""The shared pause/release predicate must protect a one-token decode."""
import unittest

from vllm.config.finetune import FinetuneConfig


class ForwardThrottleTest(unittest.TestCase):
    def test_zero_threshold_protects_single_decode(self):
        cfg = FinetuneConfig(fwd_token_throttle_enable=True,
                             fwd_token_throttle=0)
        self.assertTrue(cfg.should_throttle_backward(1))
        self.assertTrue(cfg.should_throttle_backward(512))
        # Both an empty step and pure FT have zero inference tokens: release.
        self.assertFalse(cfg.should_throttle_backward(0))

    def test_positive_threshold_remains_strict(self):
        cfg = FinetuneConfig(fwd_token_throttle_enable=True,
                             fwd_token_throttle=32)
        self.assertFalse(cfg.should_throttle_backward(32))
        self.assertTrue(cfg.should_throttle_backward(33))

    def test_disabled_and_negative_threshold(self):
        cfg = FinetuneConfig(fwd_token_throttle_enable=False,
                             fwd_token_throttle=0)
        self.assertFalse(cfg.should_throttle_backward(512))
        cfg.fwd_token_throttle_enable = True
        cfg.fwd_token_throttle = -1
        self.assertFalse(cfg.should_throttle_backward(512))


if __name__ == '__main__':
    unittest.main()
