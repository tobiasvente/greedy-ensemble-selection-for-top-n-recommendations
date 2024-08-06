import pandas as pd
import numpy as np


def apply_model_weights(model_weights: pd.DataFrame, pure_scores=pd.DataFrame):
    """
    This function applies the model weights to the pure score matrix
    Parameters
    ----------
    model_weights   : pd.DataFrame
    pure_scores    : pd.DataFrame

    Returns pd.DataFrame
    -------

    """
    # check if model labels match
    if np.array_equal(model_weights['model'].values, pure_scores.columns.values):
        # multiply each column with the corresponding weight
        pure_scores = pure_scores.mul(model_weights['score'].values, axis=1)
    else:
        raise ValueError('Model Labels do not match')
    
    return pure_scores
