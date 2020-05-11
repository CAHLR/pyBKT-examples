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

#data!
data = data_helper.convert_data("ct.csv", skill_name)
num_learns = len(data["resource_names"])
num_gs = len(data["gs_names"])
check_data.check_data(data)
print("Simple Model:")
crossvalidate.crossvalidate(data, num_gs, 1)
print(" ")

data_multilearn = data_helper.convert_data("ct.csv", skill_name, resource_name = r_name)
check_data.check_data(data_multilearn)
num_gs = 1
num_learns = len(data_multilearn["resource_names"])
print("Model With Multiple Learn Rates Based On %s" % r_name)
crossvalidate.crossvalidate(data_multilearn, num_gs, num_learns)

#predict one step doesn't support multiple guess rates...
#print(" ")
#data_multiguess = ct_data_helper.ct_data("ct.csv", skill_name, gs_name = r_name)
#check_data.check_data(data_multiguess)
#num_learns = 1
#num_gs = data_multiguess["data"].shape[0]
#print(data_multiguess["data"].shape)
#print("Model With Multiple Guess/Slip Rates Based On %s" % r_name)
#crossvalidate.crossvalidate(data_multiguess, num_gs, num_learns)
