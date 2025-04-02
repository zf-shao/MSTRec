import math
import numpy as np


def hit_ratio_at_k(actual, predicted, topk):
    sum_hits = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set([actual[i]])  # Actual item for user i (usually 1 item)
        pred_set = set(predicted[i][:topk])  # Top-k predicted items
        # If there's at least one match between actual and predicted, it's a "hit"
        if len(act_set & pred_set) > 0:
            sum_hits += 1
        true_users += 1
    # Return the hit ratio as the fraction of users who had at least one hit
    return sum_hits / true_users if true_users > 0 else 0.0


def MRR(actual, predicted):
    sum_reciprocal_rank = 0.0
    num_users = len(predicted)

    for i in range(num_users):
        # Convert actual to a numpy array if it's not already
        actual_item = actual[i]

        # Use np.where to find the index of the actual item in the predicted list
        pred_array = np.array(predicted[i])  # Make sure predicted items are in numpy array format
        ranks = np.where(pred_array == actual_item)[0]  # Get indices where actual item is found

        if len(ranks) > 0:
            # First relevant item rank (ranks are 0-based, so we add 1 for 1-based rank)
            rank = ranks[0] + 1
            sum_reciprocal_rank += 1.0 / rank
        else:
            # If the actual item is not found in the predictions, contribute 0
            sum_reciprocal_rank += 0

    return sum_reciprocal_rank / num_users if num_users > 0 else 0.0


def precision_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set([actual[i]])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(pred_set))
            true_users += 1
    return sum_recall / true_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set([actual[i]])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len([actual[user_id]]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set([actual[user_id]])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))

# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res



def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)
