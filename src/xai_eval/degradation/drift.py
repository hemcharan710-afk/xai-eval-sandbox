import numpy as np

def apply_data_drift(X, drift_level: float = 0.1):
    """
    Simulates data drift by shifting input data with gaussian noise.
    
    Represents real world scenario where incoming data slowly
    shifts away from the original training distribution.

    Args:
        X: Input data as numpy array (your features)
        drift_level: How much drift to apply.
                     0.01 = slight distribution shift
                     0.1  = moderate drift
                     0.5  = severe drift (data looks very different)

    Returns:
        Drifted version of X as numpy array
    """
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    
    noise = np.random.normal(0, drift_level, X.shape)
    return X + noise