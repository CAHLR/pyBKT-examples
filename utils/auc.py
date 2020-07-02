import sys
sys.path.append('../')
import numpy as np
import sklearn.metrics as sk

def compute_auc(true_values, pred_values, verbose):
    # true_values and pred_values may be 2d arrays if num_subparts>1
    flat_true_values = np.empty((len(true_values[0]),))
    for i in range(len(true_values)):
        for j in range(len(true_values[0])):
            if true_values[i][j] != 0:
                flat_true_values[j] = true_values[i][j]
    # represent correct as 1, incorrect as 0 for RMSE calculation
    flat_true_values = [x-1 for x in flat_true_values] 
    pred_values = pred_values.flatten()
    auc = sk.roc_auc_score(flat_true_values, pred_values)
    if verbose:
        print("AUC:", auc)
    return auc
