from __future__ import annotations

import unittest

from pipeline.diarisation import diarise_from_speaker_vad
from pipeline.steps.interaction import compute_interaction


class PairTimingTests(unittest.TestCase):
    def test_overlap_and_quick_turns_are_preserved(self) -> None:
        speaker_vad = {
            "Speaker 1": [(0.0, 1.0), (1.15, 1.40)],
            "Speaker 2": [(0.80, 1.20), (1.41, 1.60)],
        }

        turns, speaker_map = diarise_from_speaker_vad(
            speaker_vad,
            merge_gap_s=0.10,
            min_turn_duration_s=0.05,
            preserve_overlaps=True,
        )

        self.assertEqual(len(turns), 4)
        self.assertEqual(set(speaker_map.values()), {"Speaker 1", "Speaker 2"})

        interaction, _ratios, _overlaps = compute_interaction(turns, interruption_threshold_s=0.5)
        self.assertGreater(interaction["overlap_duration"], 0.20)
        self.assertGreaterEqual(interaction["interruptions"], 1)
        self.assertAlmostEqual(interaction["avg_response_latency"], 0.01, places=2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
