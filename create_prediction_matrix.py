import pandas as pd
import json
from static import *
from pathlib import Path
from file_checker import check_split_exists, check_prediction_exists, check_aggregate_prediction_exists, \
    check_supported_recommenders, \
    check_supported_metrics, check_prediction_matrix_exist
import argparse


def combine_individual_recs(recommenders: list, data_set_name: str, metric: str, topn_score: int, topn_sample: int,
                            num_batches: int, num_folds: int, run_fold: int, num_of_recs_for_ensembing):
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

    base_path = Path(f"./{DATA_FOLDER}/{data_set_name}/"
                     f"{PREDICTION_MATRIX_FOLDER}/"
                     f"{PREDICTION_MATRIX_FOLDER}_{metric}_{num_folds}_{topn_score}_{num_of_recs_for_ensembing}/")
    base_path.mkdir(parents=True, exist_ok=True)

    # Loop through all folds
    combined_recs = pd.DataFrame()

    # loop through all models and combine the recommendations
    for ix, recommender in enumerate(recommenders):
        print(recommender)
        # Read the recommendations
        recs = pd.read_csv(f"./{DATA_FOLDER}/{data_set_name}/"
                           f"{PREDICTION_FOLDER}_{recommender}_{metric}_{topn_score}_{topn_sample}_{num_batches}/"
                           f"{run_fold}_{num_folds}_aggregated_{PREDICTION_BATCH_FILE}", index_col=0)
        # Sort the recommendations by score
        recs = recs.groupby('user', group_keys=False).apply(
            lambda x: x.nlargest(num_of_recs_for_ensembing, 'score'))
        # Filter the recommendations to the number of recommendations for ensembling
        # recs = recs.groupby('user').filter(lambda x: len(x) == 100)
        # Reset the index
        recs = recs.reset_index(drop=True)

        if 'score' in recs.columns:
            recs.rename(columns={'score': recommender}, inplace=True)
        if 'rank' in recs.columns:
            recs.drop(columns=['rank'], inplace=True)
        # Combine the recommendations
        if combined_recs.empty:
            combined_recs = recs
        else:
            combined_recs = combined_recs.merge(recs, on=['item', 'user'], how='outer')
    combined_recs.to_csv(f"{base_path}/{data_set_name}_{run_fold}_{num_folds}_{PREDICTION_MATRIX_FILE}")
    return combined_recs


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Aggregating predictions evaluation!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--run_fold', dest='run_fold', type=int, required=True)
    parser.add_argument('--metric', dest='metric', type=str, required=True)
    parser.add_argument('--topn_score', dest='topn_score', type=int, required=True)
    parser.add_argument('--topn_sample', dest='topn_sample', type=int, required=True)
    parser.add_argument('--num_batches', dest='num_batches', type=int, required=True)
    parser.add_argument('--experiment', dest='experiment', type=str, required=True)
    parser.add_argument('--num_recs_for_ensembling', dest='num_recs_for_ensembling', type=int, required=True)
    args = parser.parse_args()

    print("Evaluating with arguments: ", args.__dict__)

    experiment_settings = json.load(open(f"./experiment_{args.experiment}.json"))
    recommenders = experiment_settings["RECOMMENDERS"]

    for recommender in recommenders:
        for run_fold in range(args.num_folds):
            check_supported_recommenders(recommender=recommender)
            check_supported_metrics(metric=args.metric)

            if not check_split_exists(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=run_fold):
                raise ValueError("Missing the required data splits.")

            if not check_aggregate_prediction_exists(data_set_name=args.data_set_name, num_folds=args.num_folds,
                                                     run_fold=run_fold,
                                                     recommender=recommender, metric=args.metric,
                                                     topn_score=args.topn_score,
                                                     topn_sample=args.topn_sample, num_batches=args.num_batches):
                raise ValueError("Missing the required aggregated predictions.")
    if not check_prediction_matrix_exist(data_set_name=args.data_set_name, num_folds=args.num_folds,
                                         run_fold=args.run_fold,
                                         metric=args.metric,
                                         topn_score=args.topn_score,
                                         num_recs_for_ensembling=args.num_recs_for_ensembling):
        print("Prediction Matrix do not exist. Create prediction matrix...")
    combine_individual_recs(data_set_name=args.data_set_name, num_folds=args.num_folds,
                            recommenders=recommenders, metric=args.metric, topn_score=args.topn_score,
                            topn_sample=args.topn_sample, num_batches=args.num_batches, run_fold=args.run_fold,
                            num_of_recs_for_ensembing=args.num_recs_for_ensembling)
    print("Prediction Matrix created")
else:
    print("Prediction Matrix already exist.")
