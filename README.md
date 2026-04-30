# xai-eval: XAI Explanation Disagreement Auditing Library

![Tests](https://img.shields.io/badge/tests-18%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A Python library for quantifying and stress-testing disagreement between 
Explainable AI (XAI) methods. Based on the disagreement framework proposed 
in Krishna et al. (2024), published in Transactions on Machine Learning Research.

---

## The Problem

When you ask two explanation methods *why* your model made a decision:

- **LIME** says → *"Low income was the top reason"*
- **SHAP** says → *"High debt was the top reason"*

Same model. Same decision. Two different answers.

This library gives engineers a standardized way to **measure**, **track**, 
and **stress-test** this disagreement — and find the exact point where 
explanations become too unstable to trust.

---

## Installation

```bash
git clone https://github.com/hemcharan710-afk/xai-eval-sandbox.git
cd xai-eval-sandbox
pip install -e ".[dev]"
```

---

## Quick Start

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from xai_eval.pipeline.evaluator import evaluate

# Train a model
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X, y)

# Run a full XAI audit
results = evaluate(
    model=model,
    X=X,
    explainer_1="shap",
    explainer_2="lime",
    degradation_levels=[0.01, 0.05, 0.1, 0.3, 0.5]
)
```

Output:
```
==============================================================
           XAI AUDIT REPORT
==============================================================
  Explainer 1   : shap
  Explainer 2   : lime
  Baseline RBO  : 0.972
  Baseline Jacc : 1.000
--------------------------------------------------------------
  Degradation    RBO Score    Jaccard      Status
--------------------------------------------------------------
  0.01           0.965        1.000        ✅ Safe
  0.05           0.931        0.500        ✅ Safe
  0.10           1.000        1.000        ✅ Safe
  0.30           1.000        1.000        ✅ Safe
  0.50           0.988        1.000        ✅ Safe
--------------------------------------------------------------
  ✅ Explainers stable across all levels. Safe to trust.
==============================================================
```

---

## Interpreting Results

| RBO Score | Meaning |
|---|---|
| 0.8 - 1.0 | ✅ Safe — explainers strongly agree |
| 0.5 - 0.8 | ⚠️ Warning — explainers drifting apart |
| 0.0 - 0.5 | 🚨 Danger — do not trust either explainer |

---

## Supported Explainers

| Explainer | String Name | Works With |
|---|---|---|
| Linear SHAP | `"shap"` | Linear models |
| Tree SHAP | `"shap_tree"` | Tree models |
| Kernel SHAP | `"kernel_shap"` | Any model |
| LIME | `"lime"` | Any model |
| LIME (tree) | `"lime_tree"` | Tree models |
| Integrated Gradients | `"integrated_grads"` | Linear models |
| Vanilla Gradient | `"vanilla_gradient"` | Linear models |
| Gradient x Input | `"gradient_x_input"` | Linear models |
| SmoothGrad | `"smoothgrad"` | Linear models |
| Permutation | `"permutation"` | Any model |

---

## Compare Any Two Explainers

```python
# SHAP vs LIME
evaluate(model, X, "shap", "lime")

# Vanilla Gradient vs SmoothGrad
evaluate(model, X, "vanilla_gradient", "smoothgrad")

# Gradient x Input vs Integrated Gradients
evaluate(model, X, "gradient_x_input", "integrated_grads")

# Tree models
evaluate(model, X, "shap_tree", "lime_tree")
```

---

## Metrics

### Rank-Biased Overlap (RBO)
Measures how similar two ranked feature lists are.
Top features are weighted more heavily than bottom features.

```python
from xai_eval.metrics.rbo import rbo

rbo(['age', 'income', 'debt'], ['age', 'income', 'debt'])  # 1.0
rbo(['age', 'income', 'debt'], ['debt', 'income', 'age'])  # ~0.46
```

### Jaccard@k
Measures what fraction of top-k features both explainers agree on.

```python
from xai_eval.metrics.jaccard import jaccard_at_k

jaccard_at_k(['age', 'income', 'debt'], ['age', 'income', 'debt'], k=3)  # 1.0
jaccard_at_k(['age', 'income', 'debt'], ['age', 'tax',   'loan'],  k=3)  # 0.33
```

---

## Degradation Simulations

### Weight Noise
Simulates model decay by adding gaussian noise to model weights.

```python
from xai_eval.degradation.noise import add_weight_noise

add_weight_noise(model, noise_level=0.1)
```

### Data Drift
Simulates real-world data distribution shift.

```python
from xai_eval.degradation.drift import apply_data_drift

X_drifted = apply_data_drift(X, drift_level=0.1)
```

### Quantization
Simulates model compression by reducing weight precision.

```python
from xai_eval.degradation.quantize import quantize_model

quantize_model(model, bits=8)
```

---

## Real World Use Case

```python
# A bank wants to audit their loan approval model
from sklearn.ensemble import RandomForestClassifier
from xai_eval.pipeline.evaluator import evaluate

# Their existing model
model = RandomForestClassifier().fit(X_train, y_train)

# Run weekly automated audit
results = evaluate(
    model=model,
    X=X_test,
    explainer_1="shap_tree",
    explainer_2="lime_tree",
    degradation_levels=[0.01, 0.05, 0.1, 0.3, 0.5]
)

# Set automated alert
if results["baseline_rbo"] < 0.5:
    print("WARNING: Explainers are unreliable. Do not use for auditing.")

if results["breaking_point"] is not None:
    print(f"WARNING: Model breaks at degradation level {results['breaking_point']}")
```

---

## Supported Datasets

| Type | Examples |
|---|---|
| Tabular | COMPAS, German Credit, loan approval, medical diagnosis |
| Classification | Binary, multiclass |
| Regression | With minor modification |
| sklearn datasets | `load_breast_cancer()`, `load_iris()`, `load_wine()` |

---

## Running Tests

```bash
pytest tests/ -v
```

```
18 passed in 5.43s
```

---

## Project Structure

```
xai-eval-sandbox/
├── src/xai_eval/
│   ├── explainers.py       # 10 XAI explainer wrappers
│   ├── metrics/
│   │   ├── rbo.py          # Rank-Biased Overlap
│   │   └── jaccard.py      # Jaccard@k similarity
│   ├── degradation/
│   │   ├── noise.py        # Weight noise injection
│   │   ├── drift.py        # Data drift simulation
│   │   └── quantize.py     # Model quantization
│   └── pipeline/
│       └── evaluator.py    # End-to-end audit pipeline
└── tests/
    ├── test_metrics.py
    └── test_degradation.py
```

---

## Checklist

- [x] RBO metric
- [x] Jaccard@k metric
- [x] Weight noise degradation
- [x] Data drift degradation
- [x] Model quantization degradation
- [x] 10 explainer wrappers
- [x] End-to-end audit pipeline
- [x] 18 passing tests
- [ ] PyPI publish
- [ ] GitHub Actions CI

---

## Contributing

Contributions welcome! Here is what you can add:

- New explainer wrappers
- New degradation simulations
- New disagreement metrics
- Support for neural networks
- Visualization tools

```bash
git clone https://github.com/hemcharan710-afk/xai-eval-sandbox.git
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Reference

Krishna, S., Han, T., Gu, A., Wu, S., Jabbari, S., & Lakkaraju, H. (2024).
*The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective.*
Transactions on Machine Learning Research.
https://openreview.net/forum?id=jESY2WTZCe

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

**Hemcharan** — [@hemcharan710-afk](https://github.com/hemcharan710-afk)