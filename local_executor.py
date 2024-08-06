import json
import subprocess
from select_experiment import experiment_file, stage
from file_checker import check_pruned_exists, check_split_exists, \
    check_recommender_exists, check_prediction_exists, \
    check_aggregate_prediction_exists, check_prediction_matrix_exist, \
    check_ensemble_exists


def execute_prune_original(data_set_names, num_folds):
    for data_set_name in data_set_names:
        if not check_pruned_exists(data_set_name):
            subprocess.run(
                ["python", "prune_original.py", "--data_set_name", f"{data_set_name}", "--num_folds",
                 f"{num_folds}"])


def execute_generate_splits(data_set_names, num_folds):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            if not check_split_exists(data_set_name, num_folds, fold):
                subprocess.run(
                    ["python", "generate_splits.py", "--data_set_name", f"{data_set_name}", "--num_folds",
                     f"{num_folds}", "--run_fold", f"{fold}"])


def execute_fit_recommender(data_set_names, num_folds, recommenders, metrics, topn_score, time_limit):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            for recommender in recommenders:
                for metric in metrics:
                    if not check_recommender_exists(data_set_name, num_folds, fold, recommender, metric, topn_score):
                        subprocess.run(
                            ["python", "fit_recommender.py", "--data_set_name", f"{data_set_name}", "--num_folds",
                             f"{num_folds}", "--run_fold", f"{fold}", "--recommender", f"{recommender}", "--metric",
                             f"{metric}", "--topn_score", f"{topn_score}", "--time_limit", f"{time_limit}"])


def execute_make_predictions(data_set_names, num_folds, recommenders, metrics, topn_score, topn_sample, num_batches):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            for recommender in recommenders:
                for metric in metrics:
                    for batch in range(num_batches):
                        if not check_prediction_exists(data_set_name, num_folds, fold, recommender, metric, topn_score,
                                                       topn_sample, num_batches, batch):
                            subprocess.run(
                                ["python", "make_predictions.py", "--data_set_name", f"{data_set_name}",
                                 "--num_folds", f"{num_folds}", "--run_fold", f"{fold}", "--recommender",
                                 f"{recommender}", "--metric", f"{metric}", "--topn_score", f"{topn_score}",
                                 "--topn_sample", f"{topn_sample}", "--num_batches", f"{num_batches}", "--run_batch",
                                 f"{batch}"])


def execute_aggregate_predictions(data_set_names, num_folds, recommenders, metrics, topn_score, topn_sample,
                                  num_batches):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            for recommender in recommenders:
                for metric in metrics:
                    if not check_aggregate_prediction_exists(data_set_name, num_folds, fold, recommender, metric,
                                                             topn_score, topn_sample, num_batches):
                        subprocess.run(
                            ["python", "aggregate_predictions.py", "--data_set_name", f"{data_set_name}",
                             "--num_folds", f"{num_folds}", "--run_fold", f"{fold}", "--recommender",
                             f"{recommender}", "--metric", f"{metric}", "--topn_score", f"{topn_score}",
                             "--topn_sample", f"{topn_sample}", "--num_batches", f"{num_batches}"])


def create_prediction_matrix(data_set_names, num_folds, metrics, topn_score, topn_sample,
                             num_batches, experiment_name,
                             num_recs_ensembling):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            for metric in metrics:
                for num_recs_for_ensembling in num_recs_ensembling:
                    if not check_prediction_matrix_exist(data_set_name, num_folds, fold, metric, topn_score,
                                                         num_recs_for_ensembling):
                        subprocess.run(
                        ["python", "create_prediction_matrix.py", "--data_set_name", f"{data_set_name}",
                         "--num_folds", f"{num_folds}", "--run_fold", f"{fold}",
                         "--metric", f"{metric}", "--topn_score", f"{topn_score}",
                         "--topn_sample", f"{topn_sample}", "--num_batches", f"{num_batches}", '--experiment',
                         f"{experiment_name}", '--num_recs_for_ensembling', f"{num_recs_for_ensembling}"])


def execute_create_ensemble(data_set_names, num_folds, topn_score):
    """
    Execute the creation of ensembles for given data set names, number of folds, and topn scores.

    :param data_set_names: List of data set names.
    :param num_folds: Number of folds.
    :param topn_score: List of topn scores.
    :param job_time: Maximum Job time for execution.
    :param job_memory: Job memory for execution.
    :param job_cores: Number of CPU cores for the job.
    :param fail_email: Email to notify in case of job failure.
    """
    for data_set_name in data_set_names:
        for topn in topn_score:
            for fold in range(num_folds):
                if not check_ensemble_exists(data_set_name=data_set_name, num_folds=fold, topn=topn):
                    subprocess.run(
                    ["python", "greedy_ensemble_selection.py", "--data_set_name", f"{data_set_name}",
                     "--num_folds", f"{fold}", "--topn_score", f"{topn}"])






experiment_settings = json.load(open(f"./experiment_{experiment_file}.json"))
if stage == 0:
    execute_prune_original(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"])
elif stage == 1:
    execute_generate_splits(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"])
elif stage == 2:
    execute_fit_recommender(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                            experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                            experiment_settings["OPTIMIZATION_TOPN_SCORE"], experiment_settings["HPO_TIME_LIMIT"])
elif stage == 3:
    execute_make_predictions(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                             experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                             experiment_settings["TOPN_SCORE"], experiment_settings["TOPN_SAMPLE"],
                             experiment_settings["NUM_BATCHES"])
elif stage == 4:
    execute_aggregate_predictions(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                                  experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                                  experiment_settings["TOPN_SCORE"], experiment_settings["TOPN_SAMPLE"],
                                  experiment_settings["NUM_BATCHES"])
elif stage == 5:
    create_prediction_matrix(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                             experiment_settings["METRICS"],
                             experiment_settings["TOPN_SCORE"], experiment_settings["TOPN_SAMPLE"],
                             experiment_settings["NUM_BATCHES"],
                             experiment_file,
                             experiment_settings["NUM_RECS_FOR_ENSEMBLING"])
elif stage == 6:
    execute_create_ensemble(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                            experiment_settings["TOPN_SCORE"])
else:
    print("No valid stage selected!")
