import numpy as np


def shap_explainer(model, X):
    """
    Linear SHAP — works with LogisticRegression, LinearSVC, Ridge.
    Uses Shapley values based on linear model coefficients.

    Returns feature indices ranked by mean absolute SHAP value.
    """
    import shap

    explainer = shap.LinearExplainer(model, X)
    shap_values = explainer.shap_values(X)
    mean_shap = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_shap)[::-1]
    return [f'feature_{i}' for i in indices]


def shap_tree_explainer(model, X):
    """
    TreeSHAP — works with RandomForest, XGBoost, DecisionTree.
    Uses a completely different algorithm than LIME,
    causing genuine disagreement on non-linear models.

    Returns feature indices ranked by mean absolute SHAP value.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_shap = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_shap)[::-1]
    return [f'feature_{i}' for i in indices]


def kernel_shap_explainer(model, X):
    """
    KernelSHAP — works with ANY model (linear, tree, neural network).
    Model agnostic. Directly comparable to LIME.
    This is the SHAP variant used in the Krishna et al. (2024) paper.

    Returns feature indices ranked by mean absolute SHAP value.
    """
    import shap

    # Use a small background dataset for speed
    background = shap.kmeans(X, 10)
    explainer = shap.KernelExplainer(model.predict_proba, background)
    # Use a sample of X for speed
    X_sample = X[:50] if len(X) > 50 else X
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_shap = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_shap)[::-1]
    return [f'feature_{i}' for i in indices]


def lime_explainer(model, X):
    """
    LIME — works with any sklearn model that has predict_proba.
    Builds a local linear approximation around each prediction.
    Used in Krishna et al. (2024) paper.

    Returns feature indices ranked by mean absolute LIME weight.
    """
    import lime
    import lime.lime_tabular

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        X,
        mode='classification',
        feature_names=[f'feature_{i}' for i in range(X.shape[1])]
    )

    feature_importance = np.zeros(X.shape[1])

    for i in range(min(50, len(X))):
        explanation = lime_exp.explain_instance(
            X[i],
            model.predict_proba,
            num_features=X.shape[1]
        )
        for feat, weight in explanation.as_list():
            for j in range(X.shape[1]):
                if f'feature_{j}' in feat:
                    feature_importance[j] += abs(weight)

    feature_importance /= min(50, len(X))
    indices = np.argsort(feature_importance)[::-1]
    return [f'feature_{i}' for i in indices]


def lime_explainer_tree(model, X):
    """
    LIME for tree models — uses predict_proba.
    Builds a local linear approximation around each prediction,
    which will disagree with TreeSHAP on non-linear models.

    Returns feature indices ranked by mean absolute LIME weight.
    """
    import lime
    import lime.lime_tabular

    lime_exp = lime.lime_tabular.LimeTabularExplainer(
        X,
        mode='classification',
        feature_names=[f'feature_{i}' for i in range(X.shape[1])]
    )

    feature_importance = np.zeros(X.shape[1])

    for i in range(min(50, len(X))):
        explanation = lime_exp.explain_instance(
            X[i],
            model.predict_proba,
            num_features=X.shape[1]
        )
        for feat, weight in explanation.as_list():
            for j in range(X.shape[1]):
                if f'feature_{j}' in feat:
                    feature_importance[j] += abs(weight)

    feature_importance /= min(50, len(X))
    indices = np.argsort(feature_importance)[::-1]
    return [f'feature_{i}' for i in indices]


def integrated_gradients_explainer(model, X):
    """
    Integrated Gradients — works with linear sklearn models.
    Used in Krishna et al. (2024) paper.
    Approximates gradient path from baseline to input.

    Returns feature indices ranked by mean absolute IG value.
    """
    baseline = np.zeros_like(X)
    steps = 50
    integrated_grads = np.zeros(X.shape[1])

    for i in range(steps):
        interpolated = baseline + (i / steps) * (X - baseline)
        grads = np.abs(model.coef_[0])
        integrated_grads += grads

    integrated_grads /= steps
    indices = np.argsort(integrated_grads)[::-1]
    return [f'feature_{i}' for i in indices]


def vanilla_gradient_explainer(model, X):
    """
    Vanilla Gradient — works with linear sklearn models.
    Simplest gradient-based method. Used in Krishna et al. (2024) paper.
    Just uses raw model coefficients as feature importance.

    Returns feature indices ranked by absolute gradient value.
    """
    grads = np.abs(model.coef_[0])
    indices = np.argsort(grads)[::-1]
    return [f'feature_{i}' for i in indices]


def gradient_x_input_explainer(model, X):
    """
    Gradient x Input — works with linear sklearn models.
    Multiplies gradient by input value.
    Used in Krishna et al. (2024) paper.

    Returns feature indices ranked by mean absolute Gradient x Input value.
    """
    grads = model.coef_[0]
    grad_x_input = np.abs(grads * X)
    mean_gxi = grad_x_input.mean(axis=0)
    indices = np.argsort(mean_gxi)[::-1]
    return [f'feature_{i}' for i in indices]


def smoothgrad_explainer(model, X, n_samples=50, noise_level=0.1):
    """
    SmoothGrad — works with linear sklearn models.
    Averages gradients over noisy versions of input.
    Used in Krishna et al. (2024) paper.

    Returns feature indices ranked by mean absolute SmoothGrad value.
    """
    grads = np.abs(model.coef_[0])
    smoothed_grads = np.zeros(X.shape[1])

    for _ in range(n_samples):
        noise = np.random.normal(0, noise_level, X.shape)
        noisy_X = X + noise
        sample_grads = np.abs(grads * noisy_X).mean(axis=0)
        smoothed_grads += sample_grads

    smoothed_grads /= n_samples
    indices = np.argsort(smoothed_grads)[::-1]
    return [f'feature_{i}' for i in indices]


def permutation_explainer(model, X):
    """
    Permutation Importance — works with any sklearn model.
    Measures how much model output changes when each feature is shuffled.

    Returns feature indices ranked by permutation importance.
    """
    from sklearn.inspection import permutation_importance

    y_pred = model.predict(X)
    result = permutation_importance(
        model, X, y_pred,
        n_repeats=10,
        random_state=42
    )
    indices = np.argsort(result.importances_mean)[::-1]
    return [f'feature_{i}' for i in indices]