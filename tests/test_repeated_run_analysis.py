import unittest

from utils.analyze_repeated_runs import holm_adjust, mean_ci, metrics, paired_bootstrap


class RepeatedRunAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.a = [
            {"video_id": "1", "answer": "cross", "pred_answer": "cross"},
            {"video_id": "2", "answer": "cross", "pred_answer": "yield"},
            {"video_id": "3", "answer": "yield", "pred_answer": "yield"},
            {"video_id": "4", "answer": "yield", "pred_answer": "cross"},
        ]
        self.b = [
            {"video_id": "1", "answer": "cross", "pred_answer": "cross"},
            {"video_id": "2", "answer": "cross", "pred_answer": "cross"},
            {"video_id": "3", "answer": "yield", "pred_answer": "yield"},
            {"video_id": "4", "answer": "yield", "pred_answer": "yield"},
        ]

    def test_metrics_and_label_order(self):
        result = metrics(self.a)
        self.assertEqual(result["labels"], ["cross", "yield"])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(result["accuracy"], 0.5)
        self.assertAlmostEqual(result["macro_f1"], 0.5)

    def test_student_t_interval_zero_variance(self):
        self.assertEqual(mean_ci([0.7, 0.7, 0.7]), (0.7, 0.0, 0.7, 0.7))

    def test_bootstrap_is_reproducible(self):
        first = paired_bootstrap(self.a, self.b, iterations=100, seed=42)
        second = paired_bootstrap(self.a, self.b, iterations=100, seed=42)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["macro_f1_delta_b_minus_a"], 0.5)

    def test_holm_adjustment_is_monotonic_in_rank(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])


if __name__ == "__main__":
    unittest.main()
