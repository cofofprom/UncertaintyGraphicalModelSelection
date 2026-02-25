import networkx as nx
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import hmean

def generateDominantDiagonal(dim, density):
    graph = nx.gnp_random_graph(dim, density)
    adj = nx.adjacency_matrix(graph).toarray()

    A = np.random.uniform(0.5, 1, size=(dim, dim))
    B = np.random.choice([-1, 1], size=(dim, dim))

    prec = adj * A * B
    rowsums = np.sum(np.abs(prec), axis=1)
    rowsums[rowsums == 0] = 1e-4

    prec = prec / (1.5 * rowsums[:, None])
    prec = (prec + prec.T) / 2 + np.eye(dim)

    return prec

def generateDiagonalShift(dim, density):
    graph = nx.gnp_random_graph(dim, density)
    adj = nx.adjacency_matrix(graph).toarray()

    A = np.random.uniform(0.5, 1, size=(dim, dim))
    B = np.random.choice([-1, 1], size=(dim, dim))

    prec = adj * A * B
    prec = (prec + prec.T) / 2

    return prec

def matrix2Edges(mat):
    triu = mat[np.triu_indices_from(mat, k=1)]
    return (triu != 0.).astype(np.uint64)

def evaluate(true_edges, pred_edges):
    conf = confusion_matrix(true_edges, pred_edges, labels=[0, 1])
    tn = conf[0, 0]
    fp = conf[0, 1]
    fn = conf[1, 0]
    tp = conf[1, 1]
    
    fdr = np.nan_to_num(fp / (tp + fp), nan=0)
    fomr = np.nan_to_num(fn / (tn + fn), nan=0)
    tpr = np.nan_to_num(tp / (tp + fn), nan=1)
    tnr = np.nan_to_num(tn / (tn + fp), nan=1)
    
    ba = (tpr + tnr) / 2
    f1 = hmean([1 - fdr, tpr])

    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        mcc = np.float64(0.)
    else:
        mcc = (tp * tn - fp * fn) / denominator
    
    return [tn, fp, fn, tp, fdr, fomr, tpr, tnr, ba, f1, mcc]
