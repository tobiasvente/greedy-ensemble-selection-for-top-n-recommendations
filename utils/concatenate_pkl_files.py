import pandas as pd


def concatenate_pkl_files(dataset_name: str, model: str):
    """
    This function concatenates all pkl files from the 10 batches into one pkl file
    Parameters  dataset_name : str
    ----------
    dataset_name    : str
    model        : str

    Returns None
    -------

    """
    fold_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for fold_id in fold_ids:
        df = pd.DataFrame()
        for batch_id in batch_ids:
            dict = pd.read_pickle('prediction_data/{}/10_fold/{}/10_batch/{}_fold/{}.pkl'.format(dataset_name,
                                                                                                 str(model),
                                                                                                 str(fold_id),
                                                                                                 str(batch_id)))
            for user_id in dict.keys():
                user_df = dict.get(user_id)
                user_df['user'] = user_id
                user_df = user_df[['user', 'item', 'score']]
                df = pd.concat([df, user_df], axis=0)
        df.to_csv('prediction_data/{}/10_fold/{}/10_batch/{}_fold/predictions.tsv'.format(dataset_name,
                                                                                          str(model),
                                                                                          str(fold_id)),
                  sep='\t',
                  header=False)
