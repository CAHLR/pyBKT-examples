#compares accuracy and rmse of different models, may take a LONG time to run since many models (25 total) are generated
#item_order_effect can be especially slow since many different pairs are possible
#assistments data set does not include templates, so multiguess will perform worse than expected
import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit
from utils import crossvalidate, accuracy, rmse, auc, check_data, data_helper
import copy
np.seterr(divide='ignore', invalid='ignore')
num_fit_initializations = 20
skill_name = "Box and Whisker"
seed, folds = 2020, 5 #can customize to anything, keep same seed and # folds over all trials
results = {} #create dictionary to store accuracy and rmse results

#data!
print("starting simple model data collection")
data, df = data_helper.convert_data("as.csv", skill_name, return_df=True)#save dataframe for further trials
check_data.check_data(data)
print("creating simple model")
results["Simple Model"] = crossvalidate.crossvalidate(data, folds=folds, seed=seed)

print("starting majority class calculation")
majority = 0
if np.sum(data["data"][0]) - len(data["data"][0]) > len(data["data"][0]) - (np.sum(data["data"][0]) - len(data["data"][0])):
    majority = 1
pred_values = np.zeros((len(data["data"][0]),))
pred_values.fill(majority)
true_values = data["data"][0].tolist()
pred_values = pred_values.tolist()
results["Majority Class"] = (accuracy.compute_acc(true_values,pred_values), rmse.compute_rmse(true_values,pred_values), auc.compute_auc(true_values, pred_values))


print("starting item_learning_effect data collection")
data_multilearn = data_helper.convert_data(df, skill_name, multilearn=True)
check_data.check_data(data_multilearn)
print("creating item_learning_effect model")
results["Multilearn"] = crossvalidate.crossvalidate(data_multilearn, folds=folds, seed=seed)

print("starting kt_idem data collection")
data_multiguess = data_helper.convert_data(df, skill_name, multiguess=True)
check_data.check_data(data_multiguess)
print("creating kt_idem model")
results["Multiguess"] = crossvalidate.crossvalidate(data_multiguess, folds=folds, seed=seed)

print("starting item_order_effect data collection")
data_multipair = data_helper.convert_data("as.csv", skill_name, df=df, multipair=True)
check_data.check_data(data_multipair)
print("creating item_order_effect model")
results["Multipair"] = crossvalidate.crossvalidate(data_multipair, folds=folds, seed=seed)

print("starting kt_pps model data collection")
data_multiprior = data_helper.convert_data(df, skill_name, multiprior=True)
check_data.check_data(data_multiprior)
print("creating kt_pps model")
results["Multiprior"] = crossvalidate.crossvalidate(data_multiprior, folds=folds, seed=seed)

results = {k: v for k, v in sorted(results.items(), key=lambda item: -item[1][0])}
print("Model\t\tAccuracy\tRMSE\t\tAUC")
for k, v in results.items():
    print("%s\t%.5f\t\t%.5f\t\t%.5f" % (k, v[0], v[1], v[2]))
