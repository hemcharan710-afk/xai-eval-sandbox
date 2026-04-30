import pytest
import numpy as np
from xai_eval.metrics.rbo import rbo
from xai_eval.metrics.jaccard import jaccard_at_k


class TestRBO:

    def test_identical_lists_returns_1(self):
        """Identical rankings should return perfect agreement."""
        assert rbo(['A', 'B', 'C'], ['A', 'B', 'C']) == 1.0

    def test_empty_list_returns_0(self):
        """Empty lists should return 0."""
        assert rbo([], ['A', 'B', 'C']) == 0.0
        assert rbo(['A', 'B', 'C'], []) == 0.0

    def test_score_between_0_and_1(self):
        """Score should always be between 0 and 1."""
        score = rbo(['A', 'B', 'C'], ['C', 'B', 'A'])
        assert 0.0 <= score <= 1.0

    def test_reversed_list_lower_than_identical(self):
        """Reversed list should score lower than identical list."""
        identical = rbo(['A', 'B', 'C'], ['A', 'B', 'C'])
        reversed_ = rbo(['A', 'B', 'C'], ['C', 'B', 'A'])
        assert reversed_ < identical

    def test_partial_overlap(self):
        """Partial overlap should return score between 0 and 1."""
        score = rbo(['A', 'B', 'C'], ['A', 'C', 'B'])
        assert 0.0 < score < 1.0


class TestJaccard:

    def test_identical_lists_returns_1(self):
        """Identical top-k features should return 1.0."""
        assert jaccard_at_k(['A', 'B', 'C'], ['A', 'B', 'C']) == 1.0

    def test_no_overlap_returns_0(self):
        """Completely different top-k features should return 0.0."""
        assert jaccard_at_k(['A', 'B', 'C'], ['D', 'E', 'F']) == 0.0

    def test_empty_list_returns_0(self):
        """Empty lists should return 0."""
        assert jaccard_at_k([], ['A', 'B', 'C']) == 0.0

    def test_partial_overlap(self):
        """Partial overlap should return score between 0 and 1."""
        score = jaccard_at_k(['A', 'B', 'C'], ['A', 'D', 'E'])
        assert 0.0 < score < 1.0

    def test_different_k(self):
        """Different k values should work correctly."""
        score_k2 = jaccard_at_k(['A', 'B', 'C'], ['A', 'B', 'D'], k=2)
        score_k3 = jaccard_at_k(['A', 'B', 'C'], ['A', 'B', 'D'], k=3)
        assert score_k2 == 1.0   # top 2 are identical
        assert score_k3 < 1.0    # top 3 differ