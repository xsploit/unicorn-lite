from __future__ import annotations

import unittest

from unicorn_lite.response_gate import (
    answered_refractory_features,
    calibrated_reply_probability,
    looks_like_question,
    normalize_policy_text,
    policy_features,
    refractory_features,
)


class PolicyTextNormalizationTests(unittest.TestCase):
    def test_repeated_question_marks_are_canonical(self) -> None:
        self.assertEqual(
            normalize_policy_text("Do you like turtles????"),
            "Do you like turtles",
        )

    def test_decorative_symbol_tokens_are_removed(self) -> None:
        self.assertEqual(normalize_policy_text("Ans >>>"), "Ans")
        self.assertEqual(normalize_policy_text("neuro?!>???>??-??"), "neuro")

    def test_repeated_punctuation_cannot_inflate_length_feature(self) -> None:
        normal = policy_features({"content": "Do you like turtles?"})
        adversarial = policy_features({"content": "Do you like turtles????????????"})
        self.assertEqual(normal[2], adversarial[2])

    def test_question_form_does_not_depend_on_decorative_punctuation(self) -> None:
        self.assertTrue(looks_like_question("Do you like turtles"))
        self.assertTrue(looks_like_question("Do you like turtles????"))
        self.assertFalse(looks_like_question("?"))
        self.assertFalse(looks_like_question("????"))

    def test_memory_novelty_features_are_bounded(self) -> None:
        features = policy_features(
            {
                "content": "repeat me",
                "max_memory_similarity": 1.0,
                "mean_top3_memory_similarity": 0.8,
                "near_duplicate_count": 9,
                "exact_duplicate": True,
                "max_answered_similarity": 0.95,
                "recent_answered_exact_duplicate": True,
            }
        )
        self.assertEqual(len(features), 22)
        self.assertEqual(features[16], 1.0)
        self.assertEqual(features[18], 1.0)
        self.assertEqual(features[19], 1.0)
        self.assertEqual(features[21], 1.0)

    def test_recent_near_answer_is_exposed_as_refractory_signal(self) -> None:
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        refractory = answered_refractory_features(
            now, [(0.954, now - timedelta(seconds=92))]
        )
        features = refractory_features({"content": "repeat", **refractory})
        self.assertAlmostEqual(features[0], 0.954, places=6)
        self.assertEqual(features[1], 1.0)
        self.assertGreater(features[2], 0.5)

    def test_zero_calibrator_preserves_probability(self) -> None:
        row = {
            "recent_answered_similarity": 0.95,
            "recent_answered_near_duplicate": True,
            "seconds_since_answered_match": 90.0,
        }
        self.assertAlmostEqual(
            calibrated_reply_probability(0.6, row, {"weights": [0, 0, 0, 0]}),
            0.6,
            places=6,
        )

    def test_negative_near_repeat_weight_reduces_probability(self) -> None:
        row = {
            "recent_answered_similarity": 0.95,
            "recent_answered_near_duplicate": True,
            "seconds_since_answered_match": 90.0,
        }
        calibrated = calibrated_reply_probability(
            0.6, row, {"weights": [0.0, -2.0, 0.0, 0.0]}
        )
        self.assertLess(calibrated, 0.3)

    def test_legacy_checkpoint_features_remain_available(self) -> None:
        self.assertEqual(len(policy_features({"content": "hello"}, 16)), 16)


if __name__ == "__main__":
    unittest.main()
