import sys
sys.path.append('../')
import numpy as np

def compute_rmse(flat_true_values, pred_values, verbose):
    # true_values and pred_values may be 2d arrays if num_subparts>1
    #flat_true_values = np.empty((len(true_values[0]),))
    #for i in range(len(true_values)):
    #    for j in range(len(true_values[0])):
    #        if true_values[i][j] != 0:
    #            flat_true_values[j] = true_values[i][j]
    # represent correct as 1, incorrect as 0 for RMSE calculation
    flat_true_values = [x-1 for x in flat_true_values]
    #pred_values = pred_values.flatten()
    rmse = 0
    #print(flat_true_values.shape)
    for i in range(len(flat_true_values)):
        if flat_true_values[i] != -1:
            rmse += ((flat_true_values[i] - pred_values[i]) ** 2)
    rmse /= len([x for x in flat_true_values if (x == 0 or x == 1)])
    rmse = rmse ** 0.5
    if verbose:
        print("RMSE:", rmse)
    return rmse
