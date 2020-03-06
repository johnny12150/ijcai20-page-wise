import numpy as np

"""
測資
actual = [['A', 'B', 'X'], ['A', 'B', 'Y']]
predictions = [['X', 'Y', 'Z'], ['X', 'Y', 'B']]
show_average_results(actual, predictions)

結果
K = 1 [precision: 50.00%, recall: 16.67%, F1: 25.00%, mrr: 50.00%, map: 50.00%]
K = 5 [precision: 30.00%, recall: 50.00%, F1: 37.50%, mrr: 75.00%, map: 36.11%]
K = 10 [precision: 15.00%, recall: 50.00%, F1: 23.08%, mrr: 75.00%, map: 36.11%]
K = 25 [precision: 6.00%, recall: 50.00%, F1: 10.71%, mrr: 75.00%, map: 36.11%]
K = 50 [precision: 3.00%, recall: 50.00%, F1: 5.66%, mrr: 75.00%, map: 36.11%]
K = 100 [precision: 1.50%, recall: 50.00%, F1: 2.91%, mrr: 75.00%, map: 36.11%]
K = 150 [precision: 1.00%, recall: 50.00%, F1: 1.96%, mrr: 75.00%, map: 36.11%]
"""


def eval_metrics_at_ks(actual, predictions, scores=None, k_list=None):
    """
    Computes the precision at k, recall at k, F1 at k, mrr at k, map at k
    Parameters:
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k_list : list
        A list of number of predictions to consider
    Returns:
    -------
    results : dict
    """

    def _f1(p, r):
        if p + r == 0.0:
            return 0.0
        else:
            return 2 * p * r / (p + r)

    def _mrr(ranked_list):
        try:
            idx = ranked_list.index(True)
            return 1. / (idx + 1)
        except ValueError:
            return 0.0

    def _apk(ranked_list):
        score = 0.0
        num_hits = 0.0
        for i in range(len(ranked_list)):
            if ranked_list[i]:
                score += (np.sum(ranked_list[:i + 1]) / (i + 1.0))
        return score

    if k_list is None:
        k_list = [1, 5, 10, 25, 50, 100, 150]

    if scores is not None:
        sorted_predictions = [p for p, _ in
                              sorted(zip(predictions, scores), key=lambda x: x[1], reverse=True)]
    else:
        sorted_predictions = predictions

    actual_set = set(actual)
    sorted_correct = [y_pred in actual_set for y_pred in sorted_predictions]

    results = {
        'precision': [],
        'recall': [],
        'f1': [],
        'mrr': [],
        'map': [],
        'k': k_list
    }
    num_actual = len(actual)

    for k in k_list:
        num_correct = np.sum(sorted_correct[:k])
        p = num_correct / k
        r = num_correct / num_actual
        f = _f1(p, r)
        mrr = _mrr(sorted_correct[:k])
        ap = _apk(sorted_correct[:k]) / min(k, num_actual)

        results['precision'].append(p)
        results['recall'].append(r)
        results['f1'].append(f)
        results['mrr'].append(mrr)
        results['map'].append(ap)

    return results, k_list


def show_average_results(actual, predictions, k_list=None):
    """
    Computes the precision at k, recall at k, F1 at k, mrr at k, map at k
    Parameters:
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    k_list : list
        A list of number of predictions to consider
    Returns:
    -------
    None
    """

    results = {}
    for i in range(len(actual)):
        results[i], k_list = eval_metrics_at_ks(actual[i], predictions[i])

    p_matrix = []
    r_matrix = []
    f_matrix = []
    mrr_list = []
    map_list = []

    for r in results:
        p_matrix.append(results[r]['precision'])
        r_matrix.append(results[r]['recall'])
        f_matrix.append(results[r]['f1'])
        mrr_list.append(results[r]['mrr'])
        map_list.append(results[r]['map'])

    mean_p = list(np.mean(p_matrix, axis=0))
    mean_r = list(np.mean(r_matrix, axis=0))
    mean_f = list(np.mean(f_matrix, axis=0))
    mean_m = list(np.mean(mrr_list, axis=0))
    mean_ap = list(np.mean(map_list, axis=0))

    for i in range(len(k_list)):
        print('K = %d [precision: %.2f%%, recall: %.2f%%, F1: %.2f%%, mrr: %.2f%%, map: %.2f%%]' %
              (k_list[i], mean_p[i] * 100, mean_r[i] * 100, mean_f[i] * 100, mean_m[i] * 100, mean_ap[i] * 100))
