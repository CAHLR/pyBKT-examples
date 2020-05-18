import sys
sys.path.append('../')
from sklearn.metrics import mean_squared_error
import numpy as np

def compute_rmse(true_values, pred_values, verbose):
    # true_values and pred_values may be 2d arrays if num_subparts>1
    true_values = true_values.flatten()
    # represent correct as 1, incorrect as 0 for RMSE calculation
    true_values = [x-1 for x in true_values] 
    pred_values = pred_values.flatten()
    
    rmse = mean_squared_error(true_values, pred_values, squared=False)
    if verbose:
        print("RMSE:", rmse)
    return rmse
