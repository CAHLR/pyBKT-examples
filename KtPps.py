import sys
sys.path.append('../')
import numpy as np
import os
from pyBKT.models import Model
import copy
import pandas as pd
import sklearn.metrics as sk
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
    
    df0 = pd.read_csv("./data/glops-exact-processed/"+i)
    seq_length = round(len(df0) / len(df0["user_id"].unique()))
    df_train = (df0.groupby('user_id').apply(lambda x: x.iloc[:-1] if len(x)>1 else x).reset_index(drop=True)) #remove all but last element
    
    model = Model(num_fits = 10, seed=2020)

    model.fit(data = df_train)
    predictions = model.predict(data = df0)
    
    y_pred = predictions["correct_predictions"][::seq_length]
    y_true = df0["correct"][::seq_length]
    bkt_rmse = sk.mean_squared_error(y_true, y_pred)
    
    
    
    df0 = pd.read_csv("./data/glops-exact-processed/"+i)
    seq_length = round(len(df0) / len(df0["user_id"].unique()))
    df_train = (df0.groupby('user_id').apply(lambda x: x.iloc[:-1] if len(x)>1 else x).reset_index(drop=True)) #remove all but last element
    
    model2 = Model(num_fits = 10, seed=2020)
    model2.fit(data = df_train, multiprior=True)
    predictions2 = model2.predict(data = df0)
    
    y_pred = predictions2["correct_predictions"][::seq_length]
    y_true = df0["correct"][::seq_length]
    mp_rmse = sk.mean_squared_error(y_true, y_pred)
    
    print("Standard BKT RMSE:", bkt_rmse)
    print("PPS RMSE:", mp_rmse)
    
    if bkt_rmse < mp_rmse:
        kt_better += 1
    else:
        pps_better += 1
        
    


print("PPS performs better in", pps_better, "datasets")
print("KT performs better in", kt_better, "datasets")

