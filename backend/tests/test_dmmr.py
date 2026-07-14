import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmmr import (
    DMMRConfig,
    compute_document_lambda,
    filter_candidate_pool,
    normalize_cosine_similarity,
    normalize_relevance,
    select_dynamic_mmr,
)
from summarize import MalayalamSummarizer, choose_dynamic_k, extract_with_mmr


class DMMRTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = np.asarray(
            [[1.0, 0.0], [0.98, 0.02], [0.0, 1.0], [-1.0, 0.0]],
            dtype=np.float64,
        )
        self.probabilities = np.asarray([0.9, 0.8, 0.7, 0.6])

    def test_relevance_normalization_bounds_and_order(self):
        for mode in ("raw", "minmax", "zscore", "rank"):
            values = normalize_relevance(self.probabilities, mode)
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all((values >= 0.0) & (values <= 1.0)))
            self.assertGreater(values[0], values[-1])
        np.testing.assert_allclose(normalize_relevance(self.probabilities, "rank"), [1, 2/3, 1/3, 0])

    def test_all_equal_probabilities_are_finite_and_deterministic(self):
        for mode in ("minmax", "zscore", "rank"):
            values = normalize_relevance([0.5, 0.5, 0.5], mode)
            np.testing.assert_allclose(values, [0.5, 0.5, 0.5])
        first = select_dynamic_mmr(np.eye(3), [0.5] * 3, 2)
        second = select_dynamic_mmr(np.eye(3), [0.5] * 3, 2)
        self.assertEqual(first.selected_indices, second.selected_indices)

    def test_cosine_normalization(self):
        normalized = normalize_cosine_similarity(np.asarray([-1.0, 0.0, 1.0, np.nan]))
        np.testing.assert_allclose(normalized, [0.0, 0.5, 1.0, 0.0])

    def test_dynamic_lambda_direction_and_bounds(self):
        config = DMMRConfig(lambda_min=0.60, lambda_max=0.85)
        low_variance = np.asarray([[1.0, 0.0], [0.999, 0.001], [0.998, 0.002]])
        high_variance = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        low_lambda, _, _ = compute_document_lambda(low_variance, config)
        high_lambda, _, _ = compute_document_lambda(high_variance, config)
        self.assertLess(low_lambda, high_lambda)
        self.assertGreaterEqual(low_lambda, config.lambda_min)
        self.assertLessEqual(high_lambda, config.lambda_max)

    def test_identical_embeddings_use_minimum_lambda(self):
        config = DMMRConfig()
        value, variance, _ = compute_document_lambda(np.ones((4, 3)), config)
        self.assertAlmostEqual(value, config.lambda_min)
        self.assertAlmostEqual(variance, 0.0)

    def test_candidate_filter_keeps_at_least_twice_k(self):
        relevance = np.asarray([1.0, 0.8, 0.2, 0.1, 0.0])
        pool = filter_candidate_pool(relevance, 2, DMMRConfig(relevance_floor=0.75))
        self.assertEqual(len(pool), 4)
        self.assertEqual(pool[:2], [0, 1])

    def test_score_components_and_bonus_cap(self):
        config = DMMRConfig(
            role_weight=1.0,
            cluster_weight=1.0,
            specificity_weight=1.0,
            maximum_total_positive_bonus=0.08,
        )
        result = select_dynamic_mmr(
            self.embeddings, self.probabilities, 2, config,
            role_scores=[1] * 4, cluster_scores=[1] * 4, specificity_scores=[1] * 4,
        )
        for row in result.diagnostics:
            self.assertGreaterEqual(row["normalized_relevance"], 0.0)
            self.assertLessEqual(row["normalized_relevance"], 1.0)
            self.assertLessEqual(row["total_positive_bonus"], 0.08 + 1e-12)
            self.assertTrue(np.isfinite(row["final_dmmr_score"]))

    def test_relevance_guard_and_fallback(self):
        config = DMMRConfig(
            lambda_min=0.05,
            lambda_max=0.10,
            fixed_lambda=0.05,
            use_dynamic_lambda=False,
            minimum_relevance_retention=1.0,
            enable_candidate_floor=False,
        )
        result = select_dynamic_mmr(self.embeddings, self.probabilities, 2, config)
        self.assertTrue(result.rerun_used)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.selected_indices, [0, 1])

    def test_duplicate_threshold_changes_redundancy_record(self):
        config = DMMRConfig(duplicate_threshold=0.90, enable_relevance_guard=False)
        result = select_dynamic_mmr(self.embeddings, self.probabilities, 2, config)
        duplicate_rows = [row for row in result.diagnostics if row["sentence_index"] == 1 and row["selection_iteration"] == 2]
        self.assertTrue(duplicate_rows[0]["duplicate_penalty_applied"])
        self.assertEqual(duplicate_rows[0]["redundancy"], 1.0)

    def test_empty_and_single_sentence_inputs(self):
        empty = select_dynamic_mmr(np.empty((0, 2)), [], 0)
        self.assertEqual(empty.selected_indices, [])
        single = select_dynamic_mmr(np.asarray([[1.0, 0.0]]), [0.7], 1)
        self.assertEqual(single.selected_indices, [0])

    def test_nan_and_infinity_handling(self):
        embeddings = np.asarray([[np.nan, 0.0], [np.inf, 1.0], [1.0, 0.0]])
        result = select_dynamic_mmr(embeddings, [np.nan, np.inf, -np.inf], 2)
        self.assertEqual(len(result.selected_indices), 2)
        self.assertTrue(np.all(np.isfinite(result.normalized_relevance)))

    def test_summary_length_rule(self):
        expected = {0: 0, 1: 1, 3: 1, 5: 1, 6: 2, 8: 2, 9: 3, 12: 3, 13: 4, 20: 4}
        for sentence_count, k in expected.items():
            self.assertEqual(choose_dynamic_k(sentence_count), k)

    def test_requested_count_and_chronological_ordering(self):
        summarizer = MalayalamSummarizer.__new__(MalayalamSummarizer)
        summarizer.model_label = "Synthetic"
        summarizer.score_sentences = lambda sentences: (
            self.embeddings,
            np.asarray([0.7, 0.6, 0.5, 0.9]),
        )
        text = "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence."
        _, selected_sentences = summarizer.summarize(text, k=2)
        self.assertEqual(len(selected_sentences), 2)
        source_positions = [text.index(sentence) for sentence in selected_sentences]
        self.assertEqual(source_positions, sorted(source_positions))

    def test_summarizer_one_sentence_shortcut(self):
        summarizer = MalayalamSummarizer.__new__(MalayalamSummarizer)
        summary, selected = summarizer.summarize("Only one sentence.")
        self.assertEqual(summary, "Only one sentence.")
        self.assertEqual(selected, ["Only one sentence."])


if __name__ == "__main__":
    unittest.main()
