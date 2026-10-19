"""An expiring RPS gate must not put live training to sleep."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from vllm.config.finetune import FinetuneConfig
from vllm.deltaserve.coordinator import FinetuneCoordinator
from vllm.deltaserve.ft_scheduler import FinetuneScheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler


class IdleThrottleTest(unittest.TestCase):
    def make_scheduler(self, relay):
        coord = FinetuneCoordinator(512)
        coord.relay_mode = relay
        coord.rps_tracker.note_arrival(10.0)
        self.check_rate(coord, 10.0)
        scheduler = object.__new__(FinetuneScheduler)
        scheduler._coord = coord
        scheduler._profiling_mode = False
        scheduler.vllm_config = SimpleNamespace(
            finetune_config=FinetuneConfig(enable_finetuning=True)
        )
        scheduler._ft_injector = SimpleNamespace(
            store=SimpleNamespace(
                has_next=lambda: True,
                current_epoch=0,
                total_epochs=1,
                sorted_lengths=[32],
                has_claimed=lambda: False,
            )
        )
        return scheduler, coord

    def check_rate(self, coord, now):
        coord.check_rps_throttle(
            now, close_rps=5, open_rps=2, window_s=0.1, close_time=1
        )

    def has_idle_work(self, scheduler):
        # The inference scheduler reports no more requests. This is exactly
        # when EngineCore otherwise blocks waiting for another client request.
        with patch.object(AsyncScheduler, "has_requests", return_value=False):
            return scheduler.has_requests()

    def test_throttle_keeps_engine_alive_until_quiet_window(self):
        for relay in (False, True):
            with self.subTest(relay=relay):
                scheduler, coord = self.make_scheduler(relay)
                self.assertEqual(coord.next_ft_budget(), 0)
                self.assertTrue(self.has_idle_work(scheduler))
                self.assertFalse(getattr(scheduler, "_ft_exhausted_logged", False))
                # schedule() can now run again after the rate window expires.
                self.check_rate(coord, 10.2)
                self.assertFalse(coord.rps_throttle_active)
                self.assertEqual(coord.next_ft_budget(), 512)
                self.assertTrue(self.has_idle_work(scheduler))

    def test_backward_done_does_not_sleep_while_rate_gate_is_held(self):
        for relay in (False, True):
            with self.subTest(relay=relay):
                scheduler, coord = self.make_scheduler(relay)
                coord.pending_backward = True
                response = {"event": "activations_processed", "n": 512}
                if relay:
                    coord.apply_relayed_done(response)
                else:
                    coord.backward_process = SimpleNamespace(
                        poll_response=lambda: response
                    )
                    coord.poll_backward()
                self.assertFalse(coord.pending_backward)
                self.assertTrue(coord.rps_throttle_active)
                self.assertEqual(coord.next_ft_budget(), 0)
                self.assertTrue(self.has_idle_work(scheduler))

    def test_stopped_and_profiling_training_can_sleep(self):
        scheduler, coord = self.make_scheduler(False)
        coord.ft_started = False
        self.assertFalse(self.has_idle_work(scheduler))
        coord.ft_started = True
        scheduler._profiling_mode = True
        self.assertFalse(self.has_idle_work(scheduler))
        scheduler._profiling_mode = False
        scheduler.vllm_config.finetune_config.enable_finetuning = False
        self.assertFalse(self.has_idle_work(scheduler))

    def test_exhausted_data_can_sleep_with_stale_rate_gate(self):
        scheduler, _ = self.make_scheduler(False)
        store = scheduler._ft_injector.store
        store.has_next = lambda: False
        store.current_epoch = store.total_epochs
        self.assertFalse(self.has_idle_work(scheduler))

    def test_next_epoch_keeps_rate_gate_polling(self):
        scheduler, _ = self.make_scheduler(False)
        scheduler._ft_injector.store.has_next = lambda: False
        self.assertTrue(self.has_idle_work(scheduler))


if __name__ == "__main__":
    unittest.main()
