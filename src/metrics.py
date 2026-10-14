import numpy as np

def rmsle(y_true, y_pred):
    """
    Root Mean Squared Logarithmic Error.
    RMSLE = sqrt( 1/N * sum( (ln(y_true + 1) - ln(y_pred + 1))^2 ) )
    """
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))

def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred, eps=1.0):
    y_true = np.asarray(y_true)
    y_pred = np.clip(np.asarray(y_pred), 0, None)
    return float(np.mean(np.abs(y_true - y_pred) / (y_true + eps)) * 100)

