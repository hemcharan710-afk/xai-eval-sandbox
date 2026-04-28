import numpy as np

def quantize_model(model, bits: int = 8):
    """
    Simulates model quantization by reducing weight precision.
    
    Real world scenario: Companies compress models to run faster
    on cheaper hardware. This reduces weight precision and can
    cause explainers to disagree more.

    Args:
        model: A trained sklearn model (e.g. LogisticRegression)
        bits: How aggressively to quantize.
              16 = mild compression, barely noticeable
              8  = standard compression, some precision loss
              4  = aggressive compression, significant precision loss

    Returns:
        The same model with quantized weights
    """
    if hasattr(model, 'coef_'):
        # Calculate the scale factor based on bits
        scale = 2 ** bits
        # Round weights to simulate reduced precision
        model.coef_ = np.round(model.coef_ * scale) / scale
    else:
        raise ValueError(f"Model {type(model).__name__} has no coef_ attribute. "
                          "Use a linear model like LogisticRegression or LinearSVC.")
    return model