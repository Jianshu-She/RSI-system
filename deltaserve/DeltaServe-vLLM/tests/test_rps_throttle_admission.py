"""RPS admission must survive backward completion and FT abort rollback."""
import unittest
from types import SimpleNamespace

from vllm.deltaserve.coordinator import FinetuneCoordinator


class RpsAdmissionTest(unittest.TestCase):
    def make_coord(self):
        coord = FinetuneCoordinator(512)
        coord.rps_tracker.note_arrival(10.0)
        self.check(coord, 10.0)
        self.assertTrue(coord.rps_throttle_active)
        self.assertEqual(coord.next_ft_budget(), 0)
        return coord

    def check(self, coord, now):
        coord.check_rps_throttle(now, close_rps=5, open_rps=2,
                                 window_s=.1, close_time=1)

    def test_tp_backward_done_cannot_bypass_throttle(self):
        coord = self.make_coord()
        coord.relay_mode = True
        coord.pending_backward = True
        coord.apply_relayed_done({'event': 'activations_processed', 'n': 512})
        self.check(coord, 10.01)
        self.assertEqual(coord.next_ft_budget(), 0)

    def test_single_rank_backward_done_cannot_bypass_throttle(self):
        coord = self.make_coord()
        coord.pending_backward = True
        coord.backward_process = SimpleNamespace(poll_response=lambda: {'event': 'activations_processed', 'n': 512})
        coord.poll_backward()
        self.check(coord, 10.01)
        self.assertEqual(coord.next_ft_budget(), 0)

    def test_abort_snapshot_cannot_bypass_throttle(self):
        coord = FinetuneCoordinator(512)
        snapshot = coord.snapshot_admission()
        coord.rps_tracker.note_arrival(10.0)
        self.check(coord, 10.0)
        coord.restore_admission(snapshot)
        self.assertEqual(coord.next_ft_budget(), 0)

    def test_quiet_window_releases_and_preserves_other_gates(self):
        coord = self.make_coord()
        self.check(coord, 10.2)
        self.assertFalse(coord.rps_throttle_active)
        self.assertEqual(coord.next_ft_budget(), 512)
        coord.pending_backward = True
        self.assertEqual(coord.next_ft_budget(), 0)
        coord.pending_backward = False
        coord.reserved_fill = 512
        self.assertEqual(coord.next_ft_budget(), 0)
        coord.reserved_fill = 0
        coord.ft_started = False
        self.assertEqual(coord.next_ft_budget(), 0)


if __name__ == '__main__':
    unittest.main()
