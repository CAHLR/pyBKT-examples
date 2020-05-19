import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit
from utils import crossvalidate, check_data, data_helper
import copy
np.seterr(divide='ignore', invalid='ignore')
num_fit_initializations = 20
skill_name = "Identifying units"
r_name = "Problem Hierarchy"
seed = 2020 #can customize to anything
results = {}

#data!
print("starting simple model")
data = data_helper.convert_data("ct.csv", skill_name)
check_data.check_data(data)
results["Simple Model"] = crossvalidate.crossvalidate(data, seed=seed)
print("simple model finished")

print("starting item_learning_effect model")
data_multilearn = data_helper.convert_data("ct.csv", skill_name, multilearn=True)
check_data.check_data(data_multilearn)
results["Multilearn"] = crossvalidate.crossvalidate(data_multilearn, seed=seed)
print("simple item_learning_effect finished")
#predict one step doesn't support multiple guess rates...
#data_multiguess = data_helper.convert_data("ct.csv", skill_name, multiguess=True)
#check_data.check_data(data_multiguess)
#num_gs = len(data_multiguess["gs_names"])
#num_learns = len(data_multiguess["resource_names"])
#results["Multiguess"] = crossvalidate.crossvalidate(data_multiguess, seed=seed)


print("starting item_order_effect model")
data_multipair = data_helper.convert_data("ct.csv", skill_name, multipair=True)
check_data.check_data(data_multipair)
results["Multipair"] = crossvalidate.crossvalidate(data_multipair, seed=seed)
print("simple item_order_effect finished")

print("starting kt_pps model")
data_multiprior = data_helper.convert_data("ct.csv", skill_name, multiprior=True)
check_data.check_data(data_multiprior)
results["Multiprior"] = crossvalidate.crossvalidate(data_multiprior, seed=seed)
print("simple kt_pps finished")

results = {k: v for k, v in sorted(results.items(), key=lambda item: -item[1][0])}
print("Model\t\tAccuracy\tRMSE")
for k, v in results.items():
    print("%s\t%.5f\t\t%.5f" % (k, v[0], v[1]))
