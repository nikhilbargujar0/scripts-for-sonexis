from __future__ import annotations

import unittest

from pipeline.utils.metadata_fields import inferred_field, measured_field, provided_field


class MetadataFieldTests(unittest.TestCase):
    def test_inferred_field_preserves_high_confidence_value(self) -> None:
        field = inferred_field("indoor", 0.71)
        self.assertEqual(field["value"], "indoor")
        self.assertEqual(field["source"], "inferred")
        self.assertEqual(field["confidence"], 0.71)

    def test_inferred_field_downgrades_low_confidence_value(self) -> None:
        field = inferred_field("male", 0.28)
        self.assertEqual(field["value"], "uncertain")
        self.assertEqual(field["source"], "inferred")
        self.assertEqual(field["confidence"], 0.28)

    def test_provided_field_is_authoritative(self) -> None:
        field = provided_field("female")
        self.assertEqual(field["value"], "female")
        self.assertEqual(field["source"], "user_provided")
        self.assertEqual(field["confidence"], 1.0)

    def test_measured_field_carries_method(self) -> None:
        field = measured_field(7300, 0.94, method="spectral_rolloff_95")
        self.assertEqual(field["value"], 7300)
        self.assertEqual(field["source"], "measured")
        self.assertEqual(field["confidence"], 0.94)
        self.assertEqual(field["method"], "spectral_rolloff_95")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
