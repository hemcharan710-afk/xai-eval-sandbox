import numpy as np
import copy
from xai_eval.metrics.rbo import rbo
from xai_eval.metrics.jaccard import jaccard_at_k
from xai_eval.degradation.noise import add_weight_noise
from xai_eval.degradation.drift import apply_data_drift
from xai_eval.degradation.quantize import quantize_model


def evaluate(model, X, explainer_1, explainer_2, degradation_levels=None, breaking_point_threshold=0.5):
    """
    Runs a full XAI audit pipeline.

    Measures baseline disagreement between two explainers,
    then applies degradation and tracks how disagreement grows.

    Args:
        model:                      A trained sklearn model
        X:                          Input data as numpy array
        explainer_1:                Function that takes (model, X) and returns ranked feature list
        explainer_2:                Function that takes (model, X) and returns ranked feature list
        degradation_levels:         List of noise levels to test. e.g. [0.01, 0.05, 0.1, 0.3]
        breaking_point_threshold:   RBO score below which we consider explainers untrustworthy
                                    Default is 0.5

    Returns:
        Dictionary containing:
        - baseline_rbo:       Agreement score before any degradation
        - baseline_jaccard:   Top-k overlap score before any degradation
        - results:            List of scores at each degradation level
        - breaking_point:     The degradation level where RBO drops below threshold
        - recommendation:     Human readable verdict on trustworthiness
    """
    if degradation_levels is None:
        degradation_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

    # -------------------------------------------------------
    # Step 1 — Measure baseline disagreement on clean model
    # -------------------------------------------------------
    print("=" * 50)
    print("XAI AUDIT REPORT")
    print("=" * 50)

    ranking_1 = explainer_1(model, X)
    ranking_2 = explainer_2(model, X)

    baseline_rbo = rbo(ranking_1, ranking_2)
    baseline_jaccard = jaccard_at_k(ranking_1, ranking_2)

    print(f"Baseline RBO:     {baseline_rbo:.3f}")
    print(f"Baseline Jaccard: {baseline_jaccard:.3f}")
    print("-" * 50)

    # -------------------------------------------------------
    # Step 2 — Apply degradation and track disagreement
    # -------------------------------------------------------
    results = []
    breaking_point = None

    for level in degradation_levels:
        # Work on a copy so we don't destroy the original model
        degraded_model = copy.deepcopy(model)
        X_degraded = apply_data_drift(X, drift_level=level)
        add_weight_noise(degraded_model, noise_level=level)

        # Get new rankings from degraded model
        new_ranking_1 = explainer_1(degraded_model, X_degraded)
        new_ranking_2 = explainer_2(degraded_model, X_degraded)

        # Measure disagreement
        rbo_score = rbo(new_ranking_1, new_ranking_2)
        jaccard_score = jaccard_at_k(new_ranking_1, new_ranking_2)

        # Check if we crossed the breaking point
        if rbo_score < breaking_point_threshold and breaking_point is None:
            breaking_point = level

        # Status indicator
        if rbo_score >= 0.7:
            status = "✅ Safe"
        elif rbo_score >= 0.5:
            status = "⚠️  Warning"
        else:
            status = "🚨 Danger"

        print(f"Degradation {level:.2f} → RBO: {rbo_score:.3f}  Jaccard: {jaccard_score:.3f}  {status}")

        results.append({
            "degradation_level": level,
            "rbo_score": rbo_score,
            "jaccard_score": jaccard_score,
            "status": status
        })

    # -------------------------------------------------------
    # Step 3 — Generate recommendation
    # -------------------------------------------------------
    print("-" * 50)

    if breaking_point is None:
        recommendation = "✅ Explainers are stable across all degradation levels. Safe to trust."
    else:
        recommendation = f"🚨 Explainers become untrustworthy at degradation level {breaking_point}. Retrain before this point."

    print(recommendation)
    print("=" * 50)

    return {
        "baseline_rbo": baseline_rbo,
        "baseline_jaccard": baseline_jaccard,
        "results": results,
        "breaking_point": breaking_point,
        "recommendation": recommendation
    }