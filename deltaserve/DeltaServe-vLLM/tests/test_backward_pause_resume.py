"""Backward resumes during idle without cancelling a newer throttle pause."""

from contextlib import nullcontext
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm.config.finetune import FinetuneConfig
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class PauseResumeTest(unittest.TestCase):
    def make_idle_runner(self, event, relay_mode):
        coord = Mock(relay_mode=relay_mode)
        runner = SimpleNamespace(
            execute_model_state=None,
            routed_experts_initialized=False,
            speculative_config=None,
            parallel_config=SimpleNamespace(distributed_executor_backend="mp"),
            vllm_config=SimpleNamespace(
                finetune_config=FinetuneConfig(
                    fwd_token_throttle_enable=True, fwd_token_throttle=0
                )
            ),
            synchronize_input_prep=nullcontext,
            _update_states=Mock(),
            _ft_fill_relay_fields=Mock(),
            _ft_coordinator=coord,
            _ft_resume_evt=event,
            _throttle_held=False,
        )
        runner._ft_maybe_resume_backward = lambda c: (
            GPUModelRunner._ft_maybe_resume_backward(runner, c)
        )
        return runner, coord

    def execute_idle(self, runner):
        output = SimpleNamespace(total_num_scheduled_tokens=0)
        module = "vllm.v1.worker.gpu_model_runner"
        with (
            patch(module + ".has_kv_transfer_group", return_value=False),
            patch(module + ".has_ec_transfer", return_value=False),
        ):
            return GPUModelRunner.execute_model(runner, output)

    def test_idle_step_resumes_completed_ft_forward_without_throttle(self):
        for relay in (False, True):
            with self.subTest(relay_mode=relay):
                event = Mock(query=Mock(return_value=True))
                runner, coord = self.make_idle_runner(event, relay)
                self.execute_idle(runner)
                coord.gpu_resume_backward.assert_called_once_with()
                self.assertIsNone(runner._ft_resume_evt)
                # A later idle poll must not release the grant a second time.
                self.execute_idle(runner)
                coord.gpu_resume_backward.assert_called_once_with()

    def test_idle_step_retries_event_after_sample_tokens_misses_completion(self):
        for relay in (False, True):
            with self.subTest(relay_mode=relay):
                event = Mock(query=Mock(side_effect=[False, False, True]))
                runner, coord = self.make_idle_runner(event, relay)
                # The same event poll used by sample_tokens runs before the
                # GPU completes. Idle execute_model calls must keep polling.
                runner._ft_maybe_resume_backward(coord)
                self.execute_idle(runner)
                coord.gpu_resume_backward.assert_not_called()
                self.assertIs(runner._ft_resume_evt, event)
                self.execute_idle(runner)
                coord.gpu_resume_backward.assert_called_once_with()
                self.assertIsNone(runner._ft_resume_evt)

    def test_idle_step_releases_sticky_throttle(self):
        runner, coord = self.make_idle_runner(None, True)
        runner._throttle_held = True
        coord.fwd_throttle_active = True
        self.execute_idle(runner)
        coord.gpu_resume_backward.assert_called_once_with()
        self.assertFalse(runner._throttle_held)
        self.assertFalse(coord.fwd_throttle_active)

    def test_completed_prefill_keeps_new_throttle_pause(self):
        runner = SimpleNamespace(
            _ft_resume_evt=Mock(query=lambda: True), _throttle_held=True
        )
        coord = Mock()
        GPUModelRunner._ft_maybe_resume_backward(runner, coord)
        coord.gpu_resume_backward.assert_not_called()
        self.assertIsNone(runner._ft_resume_evt)

    def test_completed_prefill_without_throttle_resumes(self):
        runner = SimpleNamespace(
            _ft_resume_evt=Mock(query=lambda: True), _throttle_held=False
        )
        coord = Mock()
        GPUModelRunner._ft_maybe_resume_backward(runner, coord)
        coord.gpu_resume_backward.assert_called_once_with()
        self.assertIsNone(runner._ft_resume_evt)

    def test_unfinished_prefill_does_not_resume(self):
        event = Mock(query=lambda: False)
        runner = SimpleNamespace(_ft_resume_evt=event, _throttle_held=False)
        coord = Mock()
        GPUModelRunner._ft_maybe_resume_backward(runner, coord)
        coord.gpu_resume_backward.assert_not_called()
        self.assertIs(runner._ft_resume_evt, event)


if __name__ == "__main__":
    unittest.main()
