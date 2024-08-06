from pathlib import Path
import pandas as pd
import argparse
from static import *
from file_checker import check_split_exists, check_prediction_exists, check_aggregate_prediction_exists, \
    check_supported_recommenders, \
    check_supported_metrics


def evaluate(data_set_name, num_folds, run_fold, recommender, metric, topn_score, topn_sample, num_batches):
    # get required recommendations
    base_path_predictions = Path(f"./{DATA_FOLDER}/{data_set_name}/"
                                 f"{PREDICTION_FOLDER}_{recommender}_{metric}_{topn_score}_{topn_sample}_{num_batches}")
    recommendations = pd.DataFrame()

    for run_batch in range(num_batches):
        recommendations = pd.concat([recommendations, pd.read_csv(
            f"{base_path_predictions}/{run_fold}_{num_folds}_{run_batch}_{PREDICTION_BATCH_FILE}")])
    recommendations = recommendations.loc[:, ~recommendations.columns.str.contains('^Unnamed')]
    recommendations.to_csv(f"{base_path_predictions}/{run_fold}_{num_folds}_aggregated_{PREDICTION_BATCH_FILE}")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Aggregating predictions evaluation!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--run_fold', dest='run_fold', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    parser.add_argument('--metric', dest='metric', type=str, required=True)
    parser.add_argument('--topn_score', dest='topn_score', type=int, required=True)
    parser.add_argument('--topn_sample', dest='topn_sample', type=int, required=True)
    parser.add_argument('--num_batches', dest='num_batches', type=int, required=True)
    args = parser.parse_args()

    print("Evaluating with arguments: ", args.__dict__)

    check_supported_recommenders(recommender=args.recommender)
    check_supported_metrics(metric=args.metric)

    if not check_split_exists(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold):
        raise ValueError("Missing the required data splits.")

    for run_batch in range(args.num_batches):
        if not check_prediction_exists(data_set_name=args.data_set_name, num_folds=args.num_folds,
                                       run_fold=args.run_fold,
                                       recommender=args.recommender, metric=args.metric, topn_score=args.topn_score,
                                       topn_sample=args.topn_sample, num_batches=args.num_batches,
                                       run_batch=run_batch):
            raise ValueError("Missing the required predictions.")

    if not check_aggregate_prediction_exists(data_set_name=args.data_set_name, num_folds=args.num_folds,
                                             run_fold=args.run_fold,
                                             recommender=args.recommender, metric=args.metric,
                                             topn_score=args.topn_score,
                                             topn_sample=args.topn_sample, num_batches=args.num_batches):
        print("Aggregated predictions do not exist. Agregating predictions...")
        evaluate(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold,
                 recommender=args.recommender, metric=args.metric, topn_score=args.topn_score,
                 topn_sample=args.topn_sample, num_batches=args.num_batches)
        print("Aggregating predictions completed.")
    else:
        print("Aggreagted predictions already exist.")
