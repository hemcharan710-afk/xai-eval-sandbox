import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from xai_eval.degradation.noise import add_weight_noise
from xai_eval.degradation.drift import apply_data_drift
from xai_eval.degradation.quantize import quantize_model


@pytest.fixture
def trained_model():
    """A simple trained logistic regression model."""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    return LogisticRegression().fit(X, y)

@pytest.fixture
def sample_data():
    """Simple sample dataset."""
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    return X, y


class TestNoise:

    def test_weights_change_after_noise(self, trained_model):
        """Weights should be different after adding noise."""
        original = trained_model.coef_.copy()
        add_weight_noise(trained_model, noise_level=0.1)
        assert not np.allclose(original, trained_model.coef_)

    def test_higher_noise_causes_more_change(self, trained_model):
        """Higher noise level should cause bigger weight changes."""
        import copy
        model_low  = copy.deepcopy(trained_model)
        model_high = copy.deepcopy(trained_model)

        original = trained_model.coef_.copy()
        add_weight_noise(model_low,  noise_level=0.01)
        add_weight_noise(model_high, noise_level=0.5)

        low_change  = np.abs(model_low.coef_  - original).mean()
        high_change = np.abs(model_high.coef_ - original).mean()
        assert high_change > low_change

    def test_invalid_model_raises_error(self):
        """Model without coef_ should raise ValueError."""
        from sklearn.neighbors import KNeighborsClassifier
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        model = KNeighborsClassifier().fit(X, y)
        with pytest.raises(ValueError):
            add_weight_noise(model)


class TestDrift:

    def test_data_changes_after_drift(self, sample_data):
        """Data should be different after applying drift."""
        X, _ = sample_data
        X_drifted = apply_data_drift(X, drift_level=0.1)
        assert not np.allclose(X, X_drifted)

    def test_shape_preserved_after_drift(self, sample_data):
        """Shape of data should stay the same after drift."""
        X, _ = sample_data
        X_drifted = apply_data_drift(X, drift_level=0.1)
        assert X.shape == X_drifted.shape

    def test_zero_drift_minimal_change(self, sample_data):
        """Zero drift level should cause minimal change."""
        X, _ = sample_data
        X_drifted = apply_data_drift(X, drift_level=0.0)
        assert np.allclose(X, X_drifted)


class TestQuantize:

    def test_weights_become_less_precise(self, trained_model):
        """Weights should have fewer unique decimal places after quantization."""
        import copy
        model_q = copy.deepcopy(trained_model)
        quantize_model(model_q, bits=4)
        # Quantized weights should be multiples of 1/2^4
        scale = 2 ** 4
        rounded = np.round(model_q.coef_ * scale) / scale
        assert np.allclose(model_q.coef_, rounded)

    def test_lower_bits_causes_more_precision_loss(self, trained_model):
        """Lower bits should cause more precision loss."""
        import copy
        model_16 = copy.deepcopy(trained_model)
        model_4  = copy.deepcopy(trained_model)
        original = trained_model.coef_.copy()

        quantize_model(model_16, bits=16)
        quantize_model(model_4,  bits=4)

        loss_16 = np.abs(model_16.coef_ - original).mean()
        loss_4  = np.abs(model_4.coef_  - original).mean()
        assert loss_4 >= loss_16