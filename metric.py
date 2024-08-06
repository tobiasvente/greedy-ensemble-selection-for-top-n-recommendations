import numpy as np
from utils.normalize_predicted_scores import normalize_predicted_scores


class Metric:

    def __init__(self):
        pass

    def score(self, truth, recommendations, topn_score, index_mask) -> float:
        raise NotImplementedError("This method should be implemented by a subclass.")


class NDCG(Metric):

    def __init__(self):
        super().__init__()
        self.metric_name = "NDCG"

    def score(self, truth, recommendations, topn_score, index_mask) -> float:
        # calculate and store the ndcg for each user
        ndcg_per_user = []
        # pre-compute the dg for each ranked element
        discounted_gain_per_k = np.array([1 / np.log2(i + 1) for i in range(1, topn_score + 1)])
        # pre-compute the idcg
        idcg = discounted_gain_per_k.sum()
        # loop through recommendation lists of each user
        for user in recommendations['user'].unique():
            # if there are no or too few recommendations for this user, skip
            predictions = recommendations[recommendations['user'] == user]
            if predictions.shape[0] < len(index_mask):
                ndcg_per_user.append(0)
                continue
            # get sampling indices
            sample_indices = np.argwhere(index_mask).flatten()
            # look only at the sampled recommendations
            if 'ensemble_probabilities' in predictions.columns:
                top_k_predictions = predictions['item']
            else:
                top_k_predictions = predictions.values[:, 0][sample_indices]
            # filter interactions for current user from test set
            positive_test_interactions = truth["item"][truth["user"] == user].values
            # check how many of the top-k recommendations appear in the test set
            hits = np.in1d(top_k_predictions, positive_test_interactions)
            # calculate the dcg for this user
            user_dcg = discounted_gain_per_k[hits].sum()
            # calculate the ndcg for this user
            user_ndcg = user_dcg / idcg
            # append current ndcg
            ndcg_per_user.append(user_ndcg)
        return sum(ndcg_per_user) / len(ndcg_per_user)


class Precision(Metric):

    def __init__(self):
        super().__init__()
        self.metric_name = "Precision"

    def score(self, truth, recommendations, topn_score, index_mask) -> float:
        # calculate and store the precision for each user
        precision_per_user = []
        # loop through recommendation lists of each user
        for user, predictions in recommendations.items():
            # if there are no or too few recommendations for this user, skip
            if predictions.shape[0] < len(index_mask):
                precision_per_user.append(0)
                continue
            # get sampling indices
            sample_indices = np.argwhere(np.array(index_mask) == 1).flatten()
            # look only at the sampled recommendations
            top_k_predictions = predictions.values[:, 0][sample_indices]
            # filter interactions for current user from test set
            positive_test_interactions = truth["item"][truth["user"] == user].values
            # check how many of the top-k recommendations appear in the test set
            hits = np.in1d(top_k_predictions, positive_test_interactions).sum()
            # calculate the precision for this user
            user_precision = hits / topn_score
            # append current precision
            precision_per_user.append(user_precision)
        # the final result is the average precision over each user
        return sum(precision_per_user) / len(precision_per_user)


class MRR(Metric):
    def __init__(self):
        super().__init__()
        self.metric_name = "MRR"

    def score(self, truth, recommendations, topn_score, index_mask) -> float:
        # calculate and store the mrr for each user
        mrr_per_user = []
        # pre-compute the reciprocal rank for each ranked element
        mrr_per_k = np.array([1 / i for i in range(1, topn_score + 1)])
        # pre-compute the N
        N = (1 / topn_score)
        for user in recommendations['user'].unique():
            # if there are no or too few recommendations for this user, skip
            predictions = recommendations[recommendations['user'] == user]
            if predictions.shape[0] < len(index_mask):
                mrr_per_user.append(0)
                continue
            # get sampling indices
            sample_indices = np.argwhere(index_mask).flatten()
            # look only at the sampled recommendations
            if 'ensemble_probabilities' in predictions.columns:
                top_k_predictions = predictions['item']
            else:
                top_k_predictions = predictions.values[:, 0][sample_indices]

            # filter interactions for current user from test set
            positive_test_interactions = truth["item"][truth["user"] == user].values
            # check how many of the top-k recommendations appear in the test set
            hits = np.in1d(top_k_predictions, positive_test_interactions)
            # calculate the dcg for this user
            user_rr = mrr_per_k[hits].sum()
            # calculate the ndcg for this user
            user_mrr = user_rr * N
            # append current ndcg
            mrr_per_user.append(user_mrr)
        return sum(mrr_per_user) / len(mrr_per_user)


class NewMetric(Metric):
    def __init__(self, random_state: int = 42):
        super().__init__()
        self.random_state = random_state

    def score(self, truth, recommendations, topn_score, index_mask) -> float:
        user_new_metric_list = []
        np.random.seed(42)
        position_probabilities = np.array([1 / i for i in range(1, topn_score + 1)])

        for user in recommendations['user'].unique():
            # if there are no or too few recommendations for this user, skip
            predictions = recommendations[recommendations['user'] == user]
            if predictions.shape[0] < len(index_mask):
                user_new_metric_list.append(0)
                continue
            prediction_probabilities = normalize_predicted_scores(predictions['score'].values)
            predicted_items = np.random.choice(a=predictions['item'].values,
                                               size=len(predictions),
                                               replace=False,
                                               p=prediction_probabilities)
            # filter interactions for current user from test set
            positive_test_interactions = truth["item"][truth["user"] == user].values
            # check how many of the top-k recommendations appear in the test set
            hits = np.in1d(predicted_items, positive_test_interactions)
            ratio_list = position_probabilities[hits].sum()
            user_new_metric_list.append(np.mean(ratio_list))
        return sum(user_new_metric_list) / len(user_new_metric_list)