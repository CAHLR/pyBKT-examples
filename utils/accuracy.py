import sys
sys.path.append('../')
import numpy as np

def compute_acc(true_values, pred_values):
    #true_values and pred_values may be 2d arrays if num_subparts>1
    true_values = true_values.flatten()
    true_values = [x-1 for x in true_values] #represent correct as 1, incorrect as 0 for RMSE calculation
    pred_values = pred_values.flatten()
    
    correct = 0
    for i in range(len(pred_values)):
        if pred_values[i] >= 0.5 and true_values[i] == 1:
            correct += 1
        if pred_values[i] < 0.5 and true_values[i] == 0:
            correct += 1  
    
    print("Accuracy:", correct/len(pred_values))
    return correct/len(pred_values)
    