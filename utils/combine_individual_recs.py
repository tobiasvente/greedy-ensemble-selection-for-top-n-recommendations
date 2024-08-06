import pandas as pd
from metric import NDCG


def combine_individual_recs(models: pd.DataFrame, dataset_name: str, num_of_recs_for_ensembing: int = 100):
    """
    Combine the individual recommendations of the models in the models dataframe.
    Parameters
    ----------
    models  : pd.DataFrame
    dataset_name    : str
    num_of_recs_for_ensembing   : int

    Returns pd.DataFrame
    -------

    """
    # Initialize all fold ids
    fold_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Initialize empty list for the combined recommendations
    combinde_fold_recs = []
    # Initialize empty dataframe for the single model performance
    single_model_perfomance = pd.DataFrame()
    # Initialize empty list for the test data
    test = []

    # Loop through all folds
    for id in fold_ids:
        combined_recs = pd.DataFrame()
        # Read the test data
        tmp = pd.read_csv('prediction_data/{}/10_fold/splits/{}_test.csv'.format(
            dataset_name,
            str(id)),
            header=0,
            names=['user', 'item'])
        tmp['rating'] = 1
        test.append(tmp)

        # loop through all models and combine the recommendations
        for model in models['model']:
            # Check if the model is a multvae or recvae model
            if str(model).startswith(('predictions_100_multvae', 'predictions_100_recvae')):
                # Read the recommendations
                recs = pd.read_table('prediction_data/{}/10_fold/{}/10_batch/{}_fold/predictions.tsv'.format(
                    dataset_name,
                    str(model),
                    str(id)),
                    names=['id', 'org_index', 'user', 'item', 'score'])
                # Drop the unnecessary columns
                recs.drop(['id', 'org_index'], axis=1, inplace=True)
                # Drop duplicates
                upper_boundry = recs['user'].nunique()
                recs = recs[0:upper_boundry]
            else:
                # Read the recommendations
                recs = pd.read_table('prediction_data/{}/10_fold/{}/10_batch/{}_fold/predictions.tsv'.format(
                    dataset_name,
                    str(model),
                    str(id)),
                    names=['user', 'item', 'score'])
            # Filter the recommendations to the number of recommendations for ensembling
            recs = recs.groupby('user').filter(lambda x: len(x) == num_of_recs_for_ensembing)
            # Sort the recommendations by score
            recs = recs.groupby('user').apply(lambda x: x.nlargest(num_of_recs_for_ensembing, 'score'))
            # Reset the index
            recs = recs.reset_index(drop=True)

            # Calculate the single model performance
            single_model_perfomance = calculate_single_model_performance(test[id],
                                                                         recs,
                                                                         model,
                                                                         single_model_perfomance)
            # Combine the recommendations
            if combined_recs.empty:
                combined_recs = recs
            else:
                combined_recs = combined_recs.merge(recs, on=['item', 'user'], how='outer',
                                                    suffixes=('{}'.format(str(model)), str(model)[0:10]))
            combined_recs.columns = [*combined_recs.columns[:-1], str(model)]
        # Append the combined recommendations to the list
        combinde_fold_recs.append(combined_recs)
    return combinde_fold_recs, test, single_model_perfomance


def calculate_single_model_performance(test: pd.DataFrame,
                                       recs: pd.DataFrame,
                                       model_name: str,
                                       single_model_performance: pd.DataFrame,
                                       k=10):
    """
    Calculate the single model performance for the given recommendations.
    Parameters
    ----------
    test    : pd.DataFrame
    recs    : pd.DataFrame
    model_name  : str
    single_model_performance    : pd.DataFrame
    k   : int

    Returns pd.DataFrame
    -------

    """
    # Initialize NDCG
    ndcg = NDCG()
    # Sort the recommendations by score
    recs = recs.groupby('user').apply(lambda x: x.nlargest(k, 'score'))
    # Calculate NDCG@10 for each model - this is going to be the target score for the ensemble
    if len(recs) == 0:
        ndcg_at_10 = 0
    else:
        ndcg_at_10 = ndcg.score(test, recs, 10, [1] * 10)
    # Append the single model performance to the dataframe
    temp = pd.DataFrame({'model': [model_name], 'fold': [id], 'ndcg_at_10': [ndcg_at_10]})
    single_model_performance = pd.concat([single_model_performance, temp])
    return single_model_performance
