import os
import argparse
import json
from static import *

parser = argparse.ArgumentParser("HPC Executor Script for Scoring Optimizer!")
parser.add_argument('--experiment', dest='experiment', type=str, default="template")
args = parser.parse_args()

experiment_settings = json.load(open(f"./experiment_{args.experiment}.json"))

recommenders = experiment_settings["RECOMMENDERS"]
datasets = experiment_settings["DATA_SET_NAMES"]
metrics = experiment_settings["METRICS"]
num_folds = experiment_settings["NUM_FOLDS"]
topn_scores = experiment_settings["TOPN_SCORE"]
num_batches = experiment_settings["NUM_BATCHES"]

for data_set_name in datasets:
    for recommender in recommenders:
        for metric in metrics:
            for topn_score in topn_scores:
                base_path_predictions = (f"./{DATA_FOLDER}/{data_set_name}/"
                                         f"{PREDICTION_FOLDER}_{recommender}_{metric}_{topn_score}_{10}_{10}")
                if os.path.exists(base_path_predictions):
                    for fold in range(num_folds):
                        for batch in range(num_batches):
                            file_path = f"{base_path_predictions}/{fold}_{num_folds}_{batch}_{PREDICTION_BATCH_FILE}"
                            os.system(f"rm {file_path}")
