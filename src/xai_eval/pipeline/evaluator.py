import numpy as np
import copy
from xai_eval.metrics.rbo import rbo
from xai_eval.metrics.jaccard import jaccard_at_k
from xai_eval.degradation.noise import add_weight_noise
from xai_eval.degradation.drift import apply_data_drift
from xai_eval.degradation.quantize import quantize_model
from xai_eval.explainers import (
    shap_explainer,
    shap_tree_explainer,
    kernel_shap_explainer,
    lime_explainer,
    lime_explainer_tree,
    integrated_gradients_explainer,
    vanilla_gradient_explainer,
    gradient_x_input_explainer,
    smoothgrad_explainer,
    permutation_explainer
)

EXPLAINER_REGISTRY = {
    "shap":             shap_explainer,
    "shap_tree":        shap_tree_explainer,
    "kernel_shap":      kernel_shap_explainer,
    "lime":             lime_explainer,
    "lime_tree":        lime_explainer_tree,
    "integrated_grads": integrated_gradients_explainer,
    "vanilla_gradient": vanilla_gradient_explainer,
    "gradient_x_input": gradient_x_input_explainer,
    "smoothgrad":       smoothgrad_explainer,
    "permutation":      permutation_explainer,
}

# Reverse map — function to name
EXPLAINER_NAMES = {v: k for k, v in EXPLAINER_REGISTRY.items()}


def evaluate(model, X, explainer_1, explainer_2,
             degradation_levels=None, breaking_point_threshold=0.5):
    """
    Runs a full XAI audit pipeline.

    Args:
        model:                      A trained sklearn model
        X:                          Input data as numpy array
        explainer_1:                Function OR string name from EXPLAINER_REGISTRY
        explainer_2:                Function OR string name from EXPLAINER_REGISTRY
        degradation_levels:         List of noise levels to test
        breaking_point_threshold:   RBO score below which explainers are untrustworthy

    Returns:
        Dictionary with baseline scores, per-level results,
        breaking point, and recommendation.

    Example:
        evaluate(model, X, "shap", "lime")
        evaluate(model, X, "vanilla_gradient", "smoothgrad")
        evaluate(model, X, "shap_tree", "lime_tree")
    """

    # Resolve string names to functions
    if isinstance(explainer_1, str):
        name_1 = explainer_1
        if explainer_1 not in EXPLAINER_REGISTRY:
            raise ValueError(f"Unknown explainer: '{explainer_1}'. "
                           f"Choose from: {list(EXPLAINER_REGISTRY.keys())}")
        explainer_1 = EXPLAINER_REGISTRY[explainer_1]
    else:
        name_1 = EXPLAINER_NAMES.get(explainer_1, explainer_1.__name__)

    if isinstance(explainer_2, str):
        name_2 = explainer_2
        if explainer_2 not in EXPLAINER_REGISTRY:
            raise ValueError(f"Unknown explainer: '{explainer_2}'. "
                           f"Choose from: {list(EXPLAINER_REGISTRY.keys())}")
        explainer_2 = EXPLAINER_REGISTRY[explainer_2]
    else:
        name_2 = EXPLAINER_NAMES.get(explainer_2, explainer_2.__name__)

    if degradation_levels is None:
        degradation_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    # -------------------------------------------------------
    # Step 1 — Baseline disagreement on clean model
    # -------------------------------------------------------
    ranking_1 = explainer_1(model, X)
    ranking_2 = explainer_2(model, X)

    baseline_rbo     = rbo(ranking_1, ranking_2)
    baseline_jaccard = jaccard_at_k(ranking_1, ranking_2)

    # -------------------------------------------------------
    # Step 2 — Apply degradation and track disagreement
    # -------------------------------------------------------
    results = []
    breaking_point = None

    for level in degradation_levels:
        degraded_model = copy.deepcopy(model)
        X_degraded     = apply_data_drift(X, drift_level=level)
        add_weight_noise(degraded_model, noise_level=level)

        new_ranking_1 = explainer_1(degraded_model, X_degraded)
        new_ranking_2 = explainer_2(degraded_model, X_degraded)

        rbo_score     = rbo(new_ranking_1, new_ranking_2)
        jaccard_score = jaccard_at_k(new_ranking_1, new_ranking_2)

        if rbo_score < breaking_point_threshold and breaking_point is None:
            breaking_point = level

        if rbo_score >= 0.7:
            status = "✅ Safe"
        elif rbo_score >= 0.5:
            status = "⚠️  Warning"
        else:
            status = "🚨 Danger"

        results.append({
            "degradation_level": level,
            "rbo_score":         rbo_score,
            "jaccard_score":     jaccard_score,
            "status":            status
        })

    # -------------------------------------------------------
    # Step 3 — Print clean table report
    # -------------------------------------------------------
    col = 18
    print()
    print("=" * 62)
    print("           XAI AUDIT REPORT")
    print("=" * 62)
    print(f"  {name_1} ranking  : {new_ranking_1}")
    print(f"  {name_2} reanking : {new_ranking_2}")
    print(f"  Baseline RBO  : {baseline_rbo:.3f}")
    print(f"  Baseline Jacc : {baseline_jaccard:.3f}")
    print("-" * 62)
    print(f"  {'Degradation':<14} {'RBO Score':<12} {'Jaccard':<12} {'Status'}")
    print("-" * 62)
    for r in results:
        print(f"  {r['degradation_level']:<14.2f} "
              f"{r['rbo_score']:<12.3f} "
              f"{r['jaccard_score']:<12.3f} "
              f"{r['status']}")
    print("-" * 62)

    if breaking_point is None:
        recommendation = "✅ Explainers stable across all levels. Safe to trust."
    else:
        recommendation = (f"🚨 Explainers break at degradation level "
                         f"{breaking_point}. Retrain before this point.")

    print(f"  {recommendation}")
    print("=" * 62)
    print()

    return {
        "baseline_rbo":     baseline_rbo,
        "baseline_jaccard": baseline_jaccard,
        "results":          results,
        "breaking_point":   breaking_point,
        "recommendation":   recommendation
    }