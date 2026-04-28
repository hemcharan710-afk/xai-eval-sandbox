import numpy as np

def add_weight_noise(model, noise_level: float = 0.01):
    """
    Adds random gaussian noise to a sklearn model's weights.
    Simulates model decay over time.

    Args:
        model: A trained sklearn model (e.g. LogisticRegression, LinearSVC)
        noise_level: How much noise to add.
                     0.01 = tiny drift
                     0.1  = moderate decay
                     0.5  = severe degradation

    Returns:
        The same model with noisy weights (modified in place)
    """
    if hasattr(model, 'coef_'):
        noise = np.random.normal(0, noise_level, model.coef_.shape)
        model.coef_ = model.coef_ + noise
    else:
        raise ValueError(f"Model {type(model).__name__} has no coef_ attribute. "
                          "Use a linear model like LogisticRegression or LinearSVC.")
    return model