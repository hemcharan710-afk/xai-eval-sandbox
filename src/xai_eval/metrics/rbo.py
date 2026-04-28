def rbo(ranking_1: list, ranking_2: list, p: float = 0.9) -> float:
    """
    Rank-Biased Overlap (RBO) between two ranked lists.

    Works with ANY two explainers (LIME, SHAP, ANCHOR, etc.)
    Just pass in their feature rankings as lists.

    Args:
        ranking_1: Feature ranking from explainer 1. e.g. ['age', 'income', 'debt']
        ranking_2: Feature ranking from explainer 2. e.g. ['income', 'age', 'debt']
        p: Weight parameter. 0.9 means top features matter more than bottom ones.

    Returns:
        Score between 0 and 1.
        1.0 = both explainers ranked features identically
        0.0 = complete disagreement
    """
    if not ranking_1 or not ranking_2:
        return 0.0

    if ranking_1 == ranking_2:
        return 1.0

    score = 0.0
    depth = min(len(ranking_1), len(ranking_2))

    for d in range(1, depth + 1):
        set1 = set(ranking_1[:d])
        set2 = set(ranking_2[:d])
        overlap = len(set1 & set2) / d
        score += overlap * (p ** (d - 1))

    # Normalize to 0-1 range
    max_score = sum(p ** (d - 1) for d in range(1, depth + 1))
    return score / max_score