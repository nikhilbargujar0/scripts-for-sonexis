from __future__ import annotations

import unittest

import numpy as np

from pipeline.steps.alignment import AlignmentError, align_pair


class AlignmentTests(unittest.TestCase):
    def test_align_pair_corrects_delayed_second_waveform(self) -> None:
        sr = 16000
        speaker_a = np.zeros(sr, dtype=np.float32)
        speaker_b = np.zeros(sr, dtype=np.float32)
        speaker_a[2000:2200] = 1.0
        speaker_b[2400:2600] = 1.0

        aligned_a, aligned_b, report = align_pair(
            speaker_a,
            speaker_b,
            sr,
            fail_unreliable=False,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(int(np.argmax(aligned_a)), int(np.argmax(aligned_b)))
        self.assertGreater(report["confidence"], 0.35)

    def test_align_pair_raises_on_low_confidence_when_requested(self) -> None:
        sr = 16000
        speaker_a = np.zeros(sr, dtype=np.float32)
        speaker_b = np.zeros(sr, dtype=np.float32)
        speaker_a[1000:1100] = 1.0

        with self.assertRaises(AlignmentError):
            align_pair(speaker_a, speaker_b, sr, fail_unreliable=True)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
