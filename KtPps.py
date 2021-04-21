import sys
sys.path.append('../')
import numpy as np
import os
from pyBKT.models import Model
import copy
np.seterr(divide='ignore', invalid='ignore')
num_fit_initializations = 20
seed, folds = 2020, 5 #can customize to anything, keep same seed and # folds over all trials
results = {} #create dictionary to store accuracy and rmse results


# IMPORTANT!!! Choose whether to use all 42 problem sets or just the problem sets of length 4
all_files = os.listdir("./data/glops-exact-processed") # all 42 sets


total_responses = 0
kt_better = 0
pps_better = 0

for i in all_files:
    if i == "README.txt" or i == ".DS_Store":
        continue
        
    print("Creating model for ", i)
    
    model = Model(num_fits = 20, seed=2020)
    bkt_rmse = model.crossvalidate(data_path = "./data/glops-exact-processed/"+i, metric = "rmse")["rmse"].values[0]
    model2 = Model(num_fits = 20, seed=2020)
    mp_rmse = model2.crossvalidate(data_path = "./data/glops-exact-processed/"+i, multiprior = True, metric = "rmse")["rmse"].values[0]
    print("Standard BKT RMSE:", bkt_rmse)
    print("PPS RMSE:", mp_rmse)
    
    if bkt_rmse < mp_rmse:
        kt_better += 1
    else:
        pps_better += 1
        
    


print("PPS performs better in", pps_better, "datasets")
print("KT performs better in", kt_better, "datasets")

