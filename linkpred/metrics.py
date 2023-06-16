def hits_at_k(score_list, true_edges, k=1):
    """
    Given a list of scores and a list of true edges, return the percentage and number of true
    edges among the top-k scoring edges.
    """
    top_k = sorted(score_list, key=lambda x: x[2], reverse=True)[:k]
    top_k_edges = [(x[0], x[1]) for x in top_k]
    num_true_edges = len(set(top_k_edges).intersection(set(true_edges)))
    return num_true_edges / k, num_true_edges

def mean_rank(score_list, true_edges):
    """
    Given a list of scores and a list of true edges, return the mean rank of the
    true edges.
    """
    true_edges = set(true_edges)
    ranks = []
    for i, (n1, n2, _) in enumerate(sorted(score_list, key=lambda x: x[2], reverse=True)):
        if (n1, n2) in true_edges:
            ranks.append(i+1)
    return sum(ranks) / len(true_edges)

def mean_reciprocal_rank(score_list, true_edges):
    """
    Given a list of scores and a list of true edges, return the mean reciprocal
    rank of the true edges.
    """
    true_edges = set(true_edges)
    ranks = []
    for i, (n1, n2, _) in enumerate(sorted(score_list, key=lambda x: x[2], reverse=True)):
        if (n1, n2) in true_edges:
            ranks.append(1 / (i + 1))
    return sum(ranks) / len(true_edges)