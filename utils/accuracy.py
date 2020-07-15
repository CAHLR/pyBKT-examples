import sys
sys.path.append('../')
import numpy as np

def compute_acc(flat_true_values, pred_values, verbose):
    # true_values and pred_values may be 2d arrays if num_subparts>1
   # flat_true_values = np.empty((len(true_values[0]),))
   # for i in range(len(true_values)):
   #     for j in range(len(true_values[0])):
   #         if true_values[i][j] != 0:
   #             flat_true_values[j] = true_values[i][j]
    # represent correct as 1, incorrect as 0 for RMSE calculation
    flat_true_values = [x-1 for x in flat_true_values]
    #pred_values = pred_values.flatten()
    correct = 0
    for i in range(len(pred_values)):
        if pred_values[i] >= 0.5 and flat_true_values[i] == 1:
            correct += 1
        if pred_values[i] < 0.5 and flat_true_values[i] == 0:
            correct += 1  
    if verbose:
        print("Accuracy:", correct/len([x for x in flat_true_values if (x == 0 or x == 1)]))
    return correct/len([x for x in flat_true_values if (x == 0 or x == 1)])
    
