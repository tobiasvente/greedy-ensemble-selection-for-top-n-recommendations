import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import json
from static import *
from file_checker import check_pruned_exists


def prune_original(data_set_name, num_folds):
    base_path_original = f"./{DATA_FOLDER}/{data_set_name}/{ORIGINAL_FOLDER}"
    base_path_pruned = f"./{DATA_FOLDER}/{data_set_name}/{PRUNE_FOLDER}"

    if not Path(f"{base_path_pruned}/{PRUNE_FILE}").exists():

        if data_set_name == "ciaodvd":
            data = pd.read_csv(f"{base_path_original}/movie-ratings.txt", header=None, sep=',',
                               names=['userId', 'movieId', 'movie-categoryId', 'reviewId', 'movieRating',
                                      'reviewDate'],
                               usecols=['userId', 'movieId', 'movieRating'],
                               dtype={'userId': np.int64, 'movieId': np.int64, 'movieRating': np.float64})
            data.rename(columns={'userId': 'user', 'movieId': 'item', 'movieRating': 'rating'}, inplace=True)
        elif data_set_name == "ml-1m":
            data = pd.read_csv(f"{base_path_original}/ratings.dat", header=None, sep="::",
                               names=["user", "item", "rating", "timestamp"], usecols=["user", "item", "rating"])
        elif data_set_name == "ml-100k":
            data = pd.read_csv(f"{base_path_original}/u.data", header=None, sep="\t",
                               names=["user", "item", "rating", "timestamp"], usecols=["user", "item", "rating"])
        elif data_set_name == "citeulike-a":
            u_i_pairs = []
            with open(f"{base_path_original}/users.dat", "r") as f:
                for user, line in enumerate(f.readlines()):
                    item_cnt = line.strip("\n").split(" ")[0]  # First element is a count
                    items = line.strip("\n").split(" ")[1:]
                    assert len(items) == int(item_cnt)

                    for item in items:
                        # Make sure the identifiers are correct.
                        assert item.isdecimal()
                        u_i_pairs.append((user, int(item)))

            # Rename columns to default ones ?
            data = pd.DataFrame(
                u_i_pairs,
                columns=["user", "item"],
                dtype=np.int64,
            )
        elif data_set_name == "hetrec-lastfm":
            data = pd.read_csv(f"{base_path_original}/user_artists.dat", names=["user", "item", "weight"],
                               usecols=["user", "item"], header=0, sep="\t")
        else:
            raise ValueError(f"Unknown data set name {data_set_name}.")

        # remove duplicates
        data.drop_duplicates(inplace=True)

        # clip rating
        if "rating" in list(data.columns):
            min_rating = data["rating"].min()
            max_rating = data["rating"].max()
            scaled_max_rating = abs(max_rating) + abs(min_rating)
            rating_cutoff = round(scaled_max_rating * (2 / 3)) - abs(min_rating)
            data = data[data["rating"] >= rating_cutoff][["user", "item"]]

        # prune the data for warm start partitioning with n-core method based on amount of cv folds
        u_cnt, i_cnt = Counter(data["user"]), Counter(data["item"])
        while min(u_cnt.values()) < num_folds or min(i_cnt.values()) < num_folds:
            u_sig = [k for k in u_cnt if (u_cnt[k] >= num_folds)]
            i_sig = [k for k in i_cnt if (i_cnt[k] >= num_folds)]
            data = data[data["user"].isin(u_sig)]
            data = data[data["item"].isin(i_sig)]
            u_cnt, i_cnt = Counter(data["user"]), Counter(data["item"])

        # map user and item to discrete integers
        for col in ["user", "item"]:
            unique_ids = {key: value for value, key in enumerate(data[col].unique())}
            data[col].update(data[col].map(unique_ids))

        # shuffle data randomly
        shuffle_seed = np.random.randint(0, np.iinfo(np.int32).max)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # write data and seed
        Path(base_path_pruned).mkdir(exist_ok=True)
        data.to_csv(f"{base_path_pruned}/{PRUNE_FILE}", index=False)
        with open(f"{base_path_pruned}/{SHUFFLE_SEED_FILE}", "w") as file:
            file.write(str(shuffle_seed))
        print(f"Written pruned data set and shuffle seed to file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Scoring Optimizer prune original!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    args = parser.parse_args()

    print("Pruning original with arguments: ", args.__dict__)

    if not check_pruned_exists(data_set_name=args.data_set_name):
        print("Pruned data set and shuffle seed do not exist. Pruning data...")
        prune_original(data_set_name=args.data_set_name, num_folds=args.num_folds)
        print("Pruning data completed.")
    else:
        print("Pruned data set and shuffle seed exist.")
