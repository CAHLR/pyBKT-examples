import sys
sys.path.append('../')
from sklearn.metrics import mean_squared_error
import numpy as np

def compute_rmse(true_values, pred_values):
    #true_values and pred_values may be 2d arrays if num_subparts>1
    true_values = true_values.flatten()
    true_values = [x-1 for x in true_values] #represent correct as 1, incorrect as 0 for RMSE calculation
    pred_values = pred_values.flatten()
    
    rmse = mean_squared_error(true_values, pred_values, squared=False)
    print("RMSE:", rmse)
    return rmse