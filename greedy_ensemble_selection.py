import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
import os


def top_n_rec(fold_path, top_n):
    """
    Returns a dictionary of dataframe containing 'user','item','norm_score_per_recommendations' of each algo
    :param fold_path: Takes the file path of the fold
    :param top_n: Select top_n to get top N recommendations
    :return: dictionary of DataFrame with normalized score of each recommendation
    """
    rec_algo = pd.read_csv(fold_path, index_col=0)
    # Sort according to user number
    rec_algo = rec_algo.sort_values('user', ascending=True)
    algorithms = ['implicit-mf', 'user-knn', 'item-knn', 'alternating-least-squares', 'bayesian-personalized-ranking' \
                               , 'logistic-mf', 'item-item-cosine', 'item-item-tfidf', 'item-item-bm25', 'popularity']
    users = rec_algo['user'].unique()
    # Number of recommendations for each user
    N = top_n
    # Dictionaries to store normalized scores for each algorithm
    allUsers_algo = dict()
    norm_scores_algo = dict()
    # Scaler for normalization
    scaler = MinMaxScaler()

    for algorithm in algorithms:
        temp_df = pd.Series(name=f'{algorithm}_norm_score')
        for user in users:
            temp_algo = rec_algo[rec_algo['user'] == user][algorithm]
            temp_algo = temp_algo.dropna(axis=0)
            if len(temp_algo) != 0:
                temp_index = temp_algo.index
                # Normalizing the recommendation score
                temp_norm_algo = scaler.fit_transform(temp_algo.values.reshape(-1, 1))  
                temp_norm_algo = pd.Series(temp_norm_algo.flatten(), index=temp_index, name=f'{algorithm}_norm_score')

                # Sort normalized scores
                temp_norm_algo = temp_norm_algo.sort_values(ascending=False)
                temp_df = pd.concat([temp_df, temp_norm_algo.iloc[:N]], ignore_index=False)

            else:
                continue

        # Store user-item pairs with normalized scores for the current algorithm
        allUsers_algo[f'{algorithm}'] = rec_algo[['user', 'item']].loc[temp_df.index]
        norm_scores_algo[f'{algorithm}'] = allUsers_algo[f'{algorithm}'].join(temp_df)
    return norm_scores_algo


def get_ndcg_score(n_rec, val_path, top_n, ndcg_at=0, ensemble=False):
    """
    Returns a dictionary of ndcg score of each algorithm; Sorted in descending order of score
    :param n_rec: N recommendation per algo; Elements of the top_n_rec dictionary
    :param val_path: file path to the validation set
    :param top_n: top n recommendations
    :return: Dictionary of NDCG Score of all algorithm in descending order of their scores
    """
    val_set = pd.read_csv(val_path)
    algorithms = list(n_rec.keys())
    ndcg_scores = dict()
    # Initialize IDCG value
    idcg = 0
    # Calculate IDCG based on whether it's an ensemble or not
    if not ensemble:
        for i in range(top_n):
            # Assuming all recommendation are in val_set
            idcg += (1 / (np.log2(i + 2)))
    else:
        for i in range(ndcg_at):
            # Assuming all recommendation are in val_set
            idcg += (1 / (np.log2(i + 2)))

    # Calculate NDCG score for each algorithm
    for algorithm in algorithms:
        # Current algorithm
        algo = n_rec[algorithm]
        users = algo['user'].unique()
        user_ndcg_values = np.array([])
        for user in users:
            dcg = 0
            current_user_rec = algo[algo['user'] == user]['item'].values
            val_rec = val_set[val_set['user'] == user]['item'].values
            # Checking recommendation in val list
            for i, item in enumerate(current_user_rec):
                # If item is in val list otherwise 0
                if item in val_rec:
                    dcg += (1 / (np.log2(i + 2)))
            # Calculate user's NDCG value and append to array
            user_ndcg_values = np.append(user_ndcg_values,(dcg / idcg))
        # Calculate mean NDCG score for the algorithm and store it
        ndcg_scores[f'{algorithm}'] = float(format(np.mean(user_ndcg_values), ".4f"))
    return ndcg_scores

def add_weighted_rank_col(df_norm_scores, ind_model_score):
    """
    Boda_weighted_count_Method
    Add a column of weighted rank(rank_of_model*normalized weight of recommendation) to the Dataframe of each model
    :param df_norm_scores:  Dataframe with normalized weight column
    :param ind_model_score: Dictionary of Scores of individual models
    :return: Dictionary of dataframes of each model with added weighted_score column
    """
    for algorithm in ind_model_score.keys():
        # Add a new column to the dataframe for each algorithm, representing the weighted score
        df_norm_scores[f'{algorithm}']['weighted_score'] = ind_model_score[f'{algorithm}'] * \
                                                              df_norm_scores[f'{algorithm}'][f'{algorithm}_norm_score']
    return df_norm_scores


def algo_combination(ind_model_score):
    """
    Creates a list of tuples with ensembles of different models
    :param ind_model_score: Dictionary of individual model performance; sorted from highest to lowest
    :return: list of tuples with ensembles of different models
    """
    algorithms = ind_model_score.keys()
    list_of_combinations = []
    # Generate combinations of algorithms from 2 to the total number of algorithms
    for i in range(2, len(algorithms)+1):
        for combo in combinations(algorithms, i):
            list_of_combinations.append(combo)
    return list_of_combinations


def create_ensembles(poss_com, fold, top_n, ndcg_at, fold_num, data_set_name='ml-100k'):
    """
    Summing up the weighted scores of a particular recommendations among different algo to get a new ranking
    Creates a dictionary of ensembles in a greedy fashion
    Summing up the weighted scores of a particular recommendations among different algo
    :param poss_com: all possible combinations of ensembles
    :param fold: dictionary of all individual models dataframe
    :param top_n: top n recommendations of ensembles
    :param fold_num: current fold
    :param data_set_name: current dataset
    :return: Dictionary of all ensembled dataframe
    """
    ge = dict()  # Greedy Ensembles
    # Directory where ensembles will be saved
    ensembles_dir = f'./ensembles_result/{data_set_name}-ensembles/top_{top_n}_recommendations/output_fold_{fold_num}'
    os.makedirs(ensembles_dir, exist_ok=True) 

    for ensemble in poss_com[:]:
        df_ensemble = pd.DataFrame()
        for user in range(max([fold[i].user.nunique() for i in fold.keys()])):
            df_user = pd.DataFrame()
            for element in range(len(ensemble)):
                # Get the dataframe for the current element in the ensemble
                tmp_df = fold[ensemble[element]][['user', 'item', 'weighted_score']]
                # Concatenate the dataframe for the current element with the user's dataframe
                df_user = pd.concat([df_user, tmp_df[tmp_df['user'] == user]])

            # Get the top_n recommendations for each user in an ensemble
            tmp_user_df = df_user.groupby(['user','item']).sum('weighted_score') \
                                                          .sort_values(by='weighted_score',ascending=False) \
                                                          .reset_index().iloc[:ndcg_at]
            df_ensemble = pd.concat([df_ensemble, tmp_user_df], ignore_index=True)
        ge[f'{ensemble}'] = df_ensemble

        # Save the ensemble as csv file
        ensemble_name = os.path.join(ensembles_dir,f'{ensemble}.csv')
        df_ensemble.to_csv(ensemble_name, index=True)
    return ge


def dict_to_df(dictionary):
    """
    Helper function to store dictionary data to pandas dataframe
    :param dictionary: takes the dictionary to be converted to dataframe
    :return: pandas dataframe
    """
    tmp_df = {}
    for key, value in dictionary.items():
        tmp_df[key] = [value]
    return pd.DataFrame(tmp_df, index=['ndcg_score']).T

