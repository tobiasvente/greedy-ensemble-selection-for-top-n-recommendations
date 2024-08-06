import pandas as pd


def get_n_largest(combined_scores: pd.DataFrame,  n: int = 10):
    """
    This function returns the n largest scores for each user
    Parameters
    ----------
    combined_scores : pd.DataFrame
    pure_scores : pd.DataFrame
    n   : int

    Returns pd.DataFrame
    -------

    """
    combined_scores = combined_scores.groupby('user').apply(lambda x: x.nlargest(n, 'ensemble_probabilities'))

    return combined_scores[['user', 'item', 'ensemble_probabilities']]
