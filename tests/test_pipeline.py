import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from xai_eval.pipeline.evaluator import evaluate


@pytest.fixture
def model_and_data():
    """A trained model and dataset for pipeline tests."""
    X, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=42
    )
    model = LogisticRegression(max_iter=1000).fit(X, y)
    return model, X


class TestEvaluator:

    def test_returns_dict_with_correct_keys(self, model_and_data):
        """Evaluate should return dict with all expected keys."""
        model, X = model_and_data
        results = evaluate(model, X, "shap", "lime",
                           degradation_levels=[0.01, 0.1])

        assert "baseline_rbo"     in results
        assert "baseline_jaccard" in results
        assert "results"          in results
        assert "breaking_point"   in results
        assert "recommendation"   in results

    def test_baseline_rbo_between_0_and_1(self, model_and_data):
        """Baseline RBO should always be between 0 and 1."""
        model, X = model_and_data
        results = evaluate(model, X, "shap", "lime",
                           degradation_levels=[0.01])

        assert 0.0 <= results["baseline_rbo"] <= 1.0

    def test_baseline_jaccard_between_0_and_1(self, model_and_data):
        """Baseline Jaccard should always be between 0 and 1."""
        model, X = model_and_data
        results = evaluate(model, X, "shap", "lime",
                           degradation_levels=[0.01])

        assert 0.0 <= results["baseline_jaccard"] <= 1.0

    def test_results_length_matches_degradation_levels(self, model_and_data):
        """Results list length should match degradation levels."""
        model, X = model_and_data
        levels = [0.01, 0.05, 0.1]
        results = evaluate(model, X, "shap", "lime",
                           degradation_levels=levels)

        assert len(results["results"]) == len(levels)

    def test_each_result_has_correct_keys(self, model_and_data):
        """Each result entry should have correct keys."""
        model, X = model_and_data
        results = evaluate(model, X, "shap", "lime",
                           degradation_levels=[0.01])

        for r in results["results"]:
            assert "degradation_level" in r
            assert "rbo_score"         in r
            assert "jaccard_score"     in r
            assert "status"            in r

    def test_string_explainer_names_work(self, model_and_data):
        """String explainer names should resolve correctly."""
        model, X = model_and_data
        results = evaluate(model, X, "shap", "lime",
                           degradation_levels=[0.01])
        assert results is not None

    def test_invalid_explainer_raises_error(self, model_and_data):
        """Invalid explainer name should raise ValueError."""
        model, X = model_and_data
        with pytest.raises(ValueError):
            evaluate(model, X, "invalid_explainer", "lime",
                     degradation_levels=[0.01])

    def test_breaking_point_none_when_stable(self, model_and_data):
        """Breaking point should be None when explainers are stable."""
        model, X = model_and_data
        results = evaluate(model, X, "shap", "lime",
                           degradation_levels=[0.01],
                           breaking_point_threshold=0.0)

        assert results["breaking_point"] is None

    def test_recommendation_is_string(self, model_and_data):
        """Recommendation should always be a string."""
        model, X = model_and_data
        results = evaluate(model, X, "shap", "lime",
                           degradation_levels=[0.01])

        assert isinstance(results["recommendation"], str)

    def test_multiple_explainer_pairs(self, model_and_data):
        """Different explainer pairs should all work."""
        model, X = model_and_data
        pairs = [
            ("shap", "lime"),
            ("vanilla_gradient", "smoothgrad"),
            ("gradient_x_input", "integrated_grads"),
            ("shap", "permutation"),
        ]
        for exp1, exp2 in pairs:
            results = evaluate(model, X, exp1, exp2,
                               degradation_levels=[0.01])
            assert "baseline_rbo" in results