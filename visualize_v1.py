import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import os

def overall_performance(data_set_name, topn_rec, ndcg_at):
    """
    returns barplots comparing ensemble performance with top individual model performance at given ndcg
    :param data_set_name: name of the datasset
    :param topn_rec: list of topn values
    :param ndcg_at: ndcg at which individual algo are optimized
    :return: Performance comparision of ensemble and individual best algorithm: Line Plot
    """
    # 5-fold cross validation
    num_fold = 5
    # Store best performing ind algo and ensemble across all folds for different ndcg
    ind = dict()
    ensemble = dict()
    ind_min = dict()
    ind_max = dict()
    ens_min = dict()
    ens_max = dict()
    test_algo_perf_sort = dict()
    # Loop over list of datasets
    for data_set_name in data_set_name:
        # Loop over top_n values in the list
        for topn in topn_rec:
            if Path(f'./Milestone_Results/dataset/final_ndcg_at_{ndcg_at}/{data_set_name}/ind_algo_score/top_{topn}_recommendations/').exists():
                test_ind_dir = f'./Milestone_Results/dataset/final_ndcg_at_{ndcg_at}/{data_set_name}/ind_algo_score_test_set/top_{topn}_recommendations/'
                # List all the files in directory
                test_ind_score_folds = os.listdir(test_ind_dir)
                # Get list all the algorithms
                test_df = pd.read_csv(f'{test_ind_dir}/{test_ind_score_folds[0]}')
                test_algorithms = [i for i in test_df.iloc[:, 0]]
                # Store ind algo performance on test set across all folds
                test_algo_perf = dict()
                ind_algo_std = dict()
                sem_ind = dict()
                ci_ind = dict()
                ind_score = []

                for algorithm in test_algorithms:
                    score = 0
                    for i in range(num_fold):
                        test_ind_score = pd.read_csv(f'{test_ind_dir}/{test_ind_score_folds[i]}')
                        ind_score.append((test_ind_score[test_ind_score.iloc[:, 0] == algorithm]['ndcg_score'].values)[0])
                        score = score + (test_ind_score[test_ind_score.iloc[:, 0] == algorithm]['ndcg_score'].values)[0]
                    test_algo_perf[f'{algorithm}'] = round(score / num_fold, 3)
                    # Standard deviation of scores across all folds
                    ind_algo_std[f'{algorithm}'] = np.std(ind_score)
                    # Standard error mean
                    sem_ind[f'{algorithm}'] = ind_algo_std[f'{algorithm}'] / np.sqrt(num_fold)
                    # 95% Confidence Interval
                    ci_ind[f'{algorithm}'] = 1.96 * sem_ind[f'{algorithm}']
                # Sort the individual algo performance
                test_algo_perf_sort = dict(sorted(test_algo_perf.items(), key=lambda item: item[1], reverse=True))
                # Store the overall ind algo performance in csv format
                df_test_algo_perf = pd.DataFrame(list(test_algo_perf.items()), columns=['algorithm','ndcg_score'])
                all_folds_dir = f'./Milestone_Results/dataset/final_ndcg_at_{ndcg_at}/{data_set_name}/ind_score_acorss_folds/top_{topn}_recommendations/'
                os.makedirs(all_folds_dir, exist_ok=True)
                df_test_algo_perf.to_csv(os.path.join(all_folds_dir, f'overall_score.csv'), index=True)
                # Top performing algorithm's min and max deviation with 95% CI
                ind_min[f'{topn}'] = list(test_algo_perf_sort.values())[0] - ci_ind[f"{list(test_algo_perf_sort.keys())[0]}"]
                ind_max[f'{topn}'] = list(test_algo_perf_sort.values())[0] + ci_ind[f"{list(test_algo_perf_sort.keys())[0]}"]

                ens_dir = f'./Milestone_Results/dataset/final_ndcg_at_{ndcg_at}/{data_set_name}/ensemble_score//top_{topn}_recommendations/'
                # List all files in the directory
                ens_score_folds = os.listdir(ens_dir)
                df_ens = pd.read_csv(f'{ens_dir}/{ens_score_folds[1]}')
                # List of all ensembles using Greedy Ensemble Selection(GES)
                algo_ens = [i for i in df_ens.iloc[:, 0]]
                algo_perf_ens = dict()
                ens_algo_std = dict()
                sem = dict()
                ci = dict()
                # List scores across all the folds
                all_score = []
                for algorithm in algo_ens:
                    score = 0
                    for i in range(num_fold):
                        ens_score = pd.read_csv(f'{ens_dir}/{ens_score_folds[i]}')
                        all_score.append((ens_score[ens_score.iloc[:, 0] == algorithm]['ndcg_score']).values[0])
                        score = score + (ens_score[ens_score.iloc[:, 0] == algorithm]['ndcg_score']).values[0]
                    algo_perf_ens[f'{algorithm}'] = round(score / num_fold, 3)
                    # Standard deviation of scores across all folds
                    ens_algo_std[f'{algorithm}'] = np.std(all_score)
                    # Standard error mean
                    sem[f'{algorithm}'] = ens_algo_std[f'{algorithm}'] / np.sqrt(num_fold)
                    # 95% Confidence Interval
                    ci[f'{algorithm}'] = 1.96 * sem[f'{algorithm}']
                
                # Sort the ensemble performance
                algo_perf_ens_sort = dict(sorted(algo_perf_ens.items(), key=lambda item: item[1], reverse=True))
                # Store the overall ensemble performance across all folds
                df_ens_algo_perf = pd.DataFrame([list(algo_perf_ens_sort.items())[0]], columns=['algorithm', 'ndcg_score'])
                ens_all_folds_dir = f'./Milestone_Results/dataset/final_ndcg_at_{ndcg_at}/{data_set_name}/ens_score_acorss_folds/top_{topn}_recommendations/'
                os.makedirs(ens_all_folds_dir, exist_ok=True)
                df_ens_algo_perf.to_csv(os.path.join(ens_all_folds_dir, f'overall_score.csv'), index=True)

                # Top performing ind algo model
                ind[f'{topn}'] = list(test_algo_perf_sort.values())[0]
                # Top performing ensemble model
                ensemble[f'{topn}'] = list(algo_perf_ens_sort.values())[0]
                # Top ensemble's min and max deviation with 95% CI
                ens_min[f'{topn}'] = list(algo_perf_ens_sort.values())[0] - ci[f"{list(algo_perf_ens_sort.keys())[0]}"]
                ens_max[f'{topn}'] = list(algo_perf_ens_sort.values())[0] + ci[f"{list(algo_perf_ens_sort.keys())[0]}"]

        # Plot the line graph
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=list(ensemble.keys()), y=list(ensemble.values()), marker='o', mfc= 'green' ,ax=ax, color='blue'
                                            ,err_style="band", label= 'Top_perf_ens')

        if ndcg_at==5:
            x=0
        elif ndcg_at==10:
            x=1
        else:
            x=3

        # Plot the performance of ind top model at given ndcg
        plt.axhline(y=list(ind.values())[x], linestyle = '--',
                    label=f'{list(test_algo_perf_sort.keys())[0]}', color='orange')
        # Plot the CI ind top model at given ndcg
        plt.fill_between(list(ensemble.keys()), list(ind_min.values())[x], list(ind_max.values())[x], color='orange',
                         alpha=0.2)
        # Plot the CI ensemble top performing ensemble model
        plt.fill_between(list(ensemble.keys()), list(ens_min.values()), list(ens_max.values()), color='blue',
                         alpha=0.2)
        plt.title(f"Individual vs Ensemble Performance @ ndcg_{ndcg_at} - {data_set_name}")
        plt.xlabel('Top N Recommendations')
        plt.ylabel('NDCG_Score')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    overall_performance(data_set_name=["ml-100k", "ml-1m", "ciaodvd", "hetrec-lastfm", "citeulike-a"],
                        topn_rec=[5, 10, 15, 25, 50, 75, 100, 125, 150], ndcg_at=5)

    

