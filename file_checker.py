from pathlib import Path
from static import *


def get_evaluation_sets():
    return ["validation", "test"]


def check_supported_recommenders(recommender):
    supported_recommenders = ["random", "popularity", "implicit-mf", "user-knn", "item-knn",
                              "alternating-least-squares", "bayesian-personalized-ranking", "logistic-mf",
                              "item-item-cosine", "item-item-tfidf", "item-item-bm25"]
    if recommender not in supported_recommenders:
        raise ValueError(f"Recommender {recommender} is not supported.")


def check_supported_metrics(metric):
    supported_metrics = ["precision", "ndcg", "mrr", "newmetric"]
    if metric not in supported_metrics:
        raise ValueError(f"Metric {metric} is not supported.")


def check_pruned_exists(data_set_name):
    base_path_pruned = f"./{DATA_FOLDER}/{data_set_name}/{PRUNE_FOLDER}"
    return (Path(f"{base_path_pruned}/{PRUNE_FILE}").exists()
            and Path(f"{base_path_pruned}/{SHUFFLE_SEED_FILE}").exists())


def check_split_exists(data_set_name, num_folds, run_fold):
    base_path_splits = f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}"
    return (Path(f"{base_path_splits}/{run_fold}_{num_folds}_{TRAIN_FILE}").exists() and
            Path(f"{base_path_splits}/{run_fold}_{num_folds}_{VALIDATION_FILE}").exists() and
            Path(f"{base_path_splits}/{run_fold}_{num_folds}_{TEST_FILE}").exists())


def check_recommender_exists(data_set_name, num_folds, run_fold, recommender, metric, topn_score):
    # get the files to check
    base_path_recommendations = (f"./{DATA_FOLDER}/{data_set_name}/"
                                 f"{RECOMMENDER_FOLDER}_{recommender}_{metric}_{topn_score}")
    return (Path(f"{base_path_recommendations}/{run_fold}_{num_folds}_{RECOMMENDER_FILE}").exists()
            and Path(f"{base_path_recommendations}/{run_fold}_{num_folds}_{RECOMMENDER_SEED_FILE}").exists()
            and Path(f"{base_path_recommendations}/{run_fold}_{num_folds}_{RECOMMENDER_CONFIGS_FILE}").exists())


def check_prediction_exists(data_set_name, num_folds, run_fold, recommender, metric, topn_score, topn_sample,
                            num_batches, run_batch):
    return Path(f"./{DATA_FOLDER}/{data_set_name}/"
                f"{PREDICTION_FOLDER}_{recommender}_{metric}_{topn_score}_{topn_sample}_{num_batches}/"
                f"{run_fold}_{num_folds}_{run_batch}_{PREDICTION_BATCH_FILE}").exists()


def check_score_exists(data_set_name, num_folds, run_fold, recommender, metric, topn_score, topn_sample, num_batches,
                       run_batch, evaluation_set, jobs_per_task, job_id):
    base_path_results = (f"./{DATA_FOLDER}/{data_set_name}/"
                         f"{EVALUATION_FOLDER}_{recommender}_{metric}_{topn_score}_{topn_sample}_{num_batches}")
    return Path(f"{base_path_results}/{run_fold}_{num_folds}_{run_batch}_{evaluation_set}_"
                f"{job_id}_{jobs_per_task}_{EVALUATION_FILE}").exists()


def check_merged_leaderboards_exist(data_set_name, num_folds, recommender, metric, topn_score, topn_sample,
                                    num_batches):
    base_path_results = (f"./{DATA_FOLDER}/{data_set_name}/"
                         f"{EVALUATION_FOLDER}_{recommender}_{metric}_{topn_score}_{topn_sample}_{num_batches}")
    leaderboard_files = [Path(f"{base_path_results}/{run_fold}_{LEADERBOARD_FILE}") for run_fold in range(num_folds)]
    leaderboard_files.append(Path(f"{base_path_results}/{LEADERBOARD_FILE}"))
    return all([leaderboard_file.exists() for leaderboard_file in leaderboard_files])


def check_aggregate_prediction_exists(data_set_name, num_folds, run_fold, recommender, metric, topn_score, topn_sample,
                                      num_batches):
    return Path(f"./{DATA_FOLDER}/{data_set_name}/"
                f"{PREDICTION_FOLDER}_{recommender}_{metric}_{topn_score}_{topn_sample}_{num_batches}/"
                f"{run_fold}_{num_folds}_aggregated_{PREDICTION_BATCH_FILE}").exists()


def check_prediction_matrix_exist(data_set_name, num_folds, run_fold, metric, topn_score, num_recs_for_ensembling):
    return Path(f"./{DATA_FOLDER}/{data_set_name}/"
                f"{PREDICTION_MATRIX_FOLDER}/"
                f"{PREDICTION_MATRIX_FOLDER}_{metric}_{num_folds}_{topn_score}_{num_recs_for_ensembling}/"
                f"{data_set_name}_{run_fold}_{num_folds}_{PREDICTION_MATRIX_FILE}").exists()


def check_ensemble_exists(data_set_name, num_folds, topn):
    """
    Checks if the particular folder generated from experiments is already present
    :param data_set_name: Name of the dataset
    :param num_folds: The current fold number of the experiment
    :param topn: The current topn of the experiment
    :return: True/False
    """
    data_dir = Path(f"./ensembles_result/{data_set_name}-ensembles/top_{topn}_recommendations")
    for fold in range(num_folds+1):
        if not Path(f"{data_dir}/output_fold_{fold}").exists():
            return False
    return True




