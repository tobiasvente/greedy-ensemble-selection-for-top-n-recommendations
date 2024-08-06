import argparse
import greedy_ensemble_selection as en
from file_checker import check_ensemble_exists
import os


def test(data_set_name, num_folds, topn_score):
    # Define file paths for folds, validation sets, and test sets
    folds_path = [f'data/{data_set_name}/{data_set_name}_0_5_prediction_matrix.csv',
                  f'data/{data_set_name}/{data_set_name}_1_5_prediction_matrix.csv',
                  f'data/{data_set_name}/{data_set_name}_2_5_prediction_matrix.csv',
                  f'data/{data_set_name}/{data_set_name}_3_5_prediction_matrix.csv',
                  f'data/{data_set_name}/{data_set_name}_4_5_prediction_matrix.csv']

    validation_path = [f'data/{data_set_name}/0_5_validation.csv',
                       f'data/{data_set_name}/1_5_validation.csv',
                       f'data/{data_set_name}/2_5_validation.csv',
                       f'data/{data_set_name}/3_5_validation.csv',
                       f'data/{data_set_name}/4_5_validation.csv']

    test_path = [f'data/{data_set_name}/{data_set_name}_0_5_test.csv',
                 f'data/{data_set_name}/{data_set_name}_1_5_test.csv',
                 f'data/{data_set_name}/{data_set_name}_2_5_test.csv',
                 f'data/{data_set_name}/{data_set_name}_3_5_test.csv',
                 f'data/{data_set_name}/{data_set_name}_4_5_test.csv']
    ndcg_at = 5  # NDCG@N value
    fold_num = num_folds  # Select the fold 0,1,2,3,4
    top_n_value = topn_score  # Top k recommendations
    
    # Get normalized score and top_n_recommendations
    fold_norm_scores = en.top_n_rec(folds_path[fold_num], top_n_value)
    
    # Get score of each individual recommendation algorithm on validation set
    scores = en.get_ndcg_score(fold_norm_scores, validation_path[fold_num], top_n_value)
    scores_dict = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    df_ind = en.dict_to_df(scores_dict)
    df_ind_dir = f'./ind_algo_score/{data_set_name}/top_{top_n_value}_recommendations'
    os.makedirs(df_ind_dir, exist_ok=True)
    df_ind.to_csv(os.path.join(df_ind_dir, f'fold_{fold_num}.csv'), index=True)
    
    # Get score of each individual recommendation algorithm on test set
    test_scores = en.get_ndcg_score(fold_norm_scores, test_path[fold_num], top_n_value)
    test_scores_dict = dict(sorted(test_scores.items(), key=lambda item: item[1], reverse=True))
    test_df_ind = en.dict_to_df(test_scores_dict)
    test_df_ind_dir = f'./ind_algo_score_test_set/{data_set_name}/top_{top_n_value}_recommendations'
    os.makedirs(test_df_ind_dir, exist_ok=True)
    test_df_ind.to_csv(os.path.join(test_df_ind_dir, f'fold_{fold_num}.csv'), index=True)
    
    # Add weighted rank column 
    fold_weighted_score = en.add_weighted_rank_col(fold_norm_scores, scores_dict)
    com = en.algo_combination(scores)
    # Create ensembles
    dict_of_ensembles = en.create_ensembles(com, fold_weighted_score, top_n_value, ndcg_at, fold_num=fold_num,data_set_name=data_set_name)
    # Calculate NDCG scores for ensembles
    ndcg_of_ensembles = en.get_ndcg_score(dict_of_ensembles, validation_path[fold_num], top_n_value, ndcg_at=ndcg_at, ensemble= True)
    ndcg_of_ensembles_dict = dict(sorted(ndcg_of_ensembles.items(), key=lambda item: item[1], reverse=True))
    df_ensembles = en.dict_to_df(ndcg_of_ensembles_dict)
    df_ensembles_dir = f'./ensemble_score/{data_set_name}/top_{top_n_value}_recommendations'
    os.makedirs(df_ensembles_dir, exist_ok=True)
    df_ensembles.to_csv(os.path.join(df_ensembles_dir, f'fold_{fold_num}.csv'), index=True)


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser("Creating Ensembles")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--topn_score', dest='topn_score', type=int, required=True, default=10)
    args = parser.parse_args()

    print("Creating ensembles with arguments: ", args.__dict__)

    # Check if ensemble exists for the given dataset and fold
    if not check_ensemble_exists(data_set_name=args.data_set_name, num_folds = args.num_folds, topn= args.topn_score):
        print("Ensemble and shuffle seed do not exist. Ensembling data...")
        # Execute the ensembling process
        test(data_set_name=args.data_set_name, num_folds=args.num_folds, topn_score=args.topn_score)
        print("Ensembling completed.")
    else:
        print("Ensemble and shuffle seed exist.")
