import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit, predict_onestep
from utils import crossvalidate, nips_data_helper, check_data, auc
from copy import deepcopy
np.seterr(divide='ignore', invalid='ignore')

num_fit_initializations = 20
skill_count = 124 #hardcoded for nips data set

#data!
Data = nips_data_helper.convert_data("builder_train.csv", url2="builder_test.csv")

print("Data preprocessing finished")

for i in range(skill_count):
    check_data.check_data(Data[i])

print("All data okay")

all_true=[]
all_pred=[]
for skill in range(skill_count):

    if len(Data[skill]["resources"]) < 5:#auc only calculated when there are 2+ classifiers
        print("Not enough data for skill %s" % skill)
        continue

    temp = crossvalidate.crossvalidate(Data[skill], verbose=False, return_arrays=True)
    print("Skill %s of %s calculation completed" % (skill, skill_count-1))
    all_true.extend(temp[0])
    all_pred.extend(temp[1])
total_auc = auc.compute_auc(all_true, all_pred)
print("Overall AUC:", total_auc)
