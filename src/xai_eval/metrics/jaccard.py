def jaccard_at_k(ranking_1: list, ranking_2: list, k: int = 3) -> float:
    """
    Jaccard similarity of the top-k features from two ranked lists.

    Args:
        ranking_1: Feature ranking from explainer 1.
        ranking_2: Feature ranking from explainer 2.
        k: How many top features to compare.

    Returns:
        Score between 0 and 1.
        1.0 = top-k features are identical
        0.0 = top-k features share nothing in common
    """
    if not ranking_1 or not ranking_2:
        return 0.0

    set1 = set(ranking_1[:k])
    set2 = set(ranking_2[:k])

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union != 0 else 0.0