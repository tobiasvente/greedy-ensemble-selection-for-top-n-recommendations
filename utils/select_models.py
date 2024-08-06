import pandas as pd


def select_models(selected_models: pd.DataFrame, pure_scores=pd.DataFrame):
    """
    This function drops all models that were not selected by the n-1 approach
    Parameters
    ----------
    selected_models : pd.DataFrame
    pure_scores : pd.DataFrame

    Returns pd.DataFrame
    -------

    """
    # get selected models
    ensemble_models = selected_models['model'].tolist()
    # get all possible models
    all_models = pure_scores.columns.tolist()
    # get models that are not selected
    unimportant_models = set(ensemble_models) ^ set(all_models)
    # drop all unselcted models
    pure_scores = pure_scores.drop(columns=unimportant_models, axis=1)
    return pure_scores
