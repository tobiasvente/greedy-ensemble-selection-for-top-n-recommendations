def normalize_predicted_scores(score_list: list) -> list:
    x_min = min(score_list)

    if x_min < 0:
        tmp = [x + abs(x_min) + 1 for x in score_list]
        x_sum = sum(tmp)
        normalized_list = [x / x_sum for x in tmp]
    else:
        tmp = [x + 1 for x in score_list]
        x_sum = sum(tmp)
        normalized_list = [x / x_sum for x in tmp]

    return normalized_list
