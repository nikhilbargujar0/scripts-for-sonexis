from __future__ import annotations

import unittest

from pipeline.config import PipelineConfig
from pipeline.premium.quality import build_accuracy_gate
from pipeline.premium.review import build_human_review


class PremiumAccuracyGateTests(unittest.TestCase):
    def test_gate_requires_review_below_99_word_target(self) -> None:
        cfg = PipelineConfig(
            word_accuracy_target=0.99,
            speaker_accuracy_target=0.99,
            timestamp_accuracy_target=0.98,
            code_switch_accuracy_target=0.99,
        )

        gate = build_accuracy_gate(
            cfg=cfg,
            consensus=None,
            alignment=None,
            code_switch={"detected": True, "confidence": 0.7},
            speaker_attribution_confidence=1.0,
        )

        self.assertFalse(gate["passed"])
        self.assertTrue(gate["human_review_required"])
        self.assertFalse(gate["human_review_completed"])
        self.assertTrue(gate["human_review_required_for_delivery"])
        self.assertIsNone(gate["verified_word_accuracy"])
        self.assertIsNone(gate["verified_speaker_accuracy"])
        self.assertIsNone(gate["verified_timestamp_accuracy"])
        self.assertIsNone(gate["verified_code_switch_accuracy"])
        self.assertIn("estimated_word_accuracy_below_target", gate["reasons"])
        self.assertIn("estimated_timestamp_accuracy_below_target", gate["reasons"])
        self.assertIn("estimated_code_switch_accuracy_below_target", gate["reasons"])

    def test_human_review_priority_high_when_estimated_word_accuracy_low(self) -> None:
        review = build_human_review(
            pipeline_mode="premium_accuracy",
            require_human_review=True,
            accuracy_gate={
                "target_word_accuracy": 0.99,
                "estimated_word_accuracy": 0.96,
                "human_review_required": True,
                "reasons": ["estimated_word_accuracy_below_target"],
            },
            targets={"timestamp_accuracy_target": 0.98, "code_switch_accuracy_target": 0.99},
        )

        self.assertTrue(review["required"])
        self.assertEqual(review["priority"]["level"], "high")
        self.assertIn("estimated_word_accuracy_below_target", review["priority"]["reasons"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
