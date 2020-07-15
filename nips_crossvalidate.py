import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit, predict_onestep
from utils import crossvalidate, nips_data_helper, check_data, auc
from copy import deepcopy
np.seterr(divide='ignore', invalid='ignore')

num_fit_initializations = 20
skill_count = 124

#data!
Data = nips_data_helper.convert_data("builder_train.csv")
test_data = nips_data_helper.convert_data("builder_test.csv")

print("Data preprocessing finished")

for i in range(skill_count):
    check_data.check_data(Data[i])
    check_data.check_data(test_data[i])
    
print("All data okay")

total_auc = 0
total_trials = 0
for skill in range(skill_count):
    num_fit_initializations = 5
    best_likelihood = float("-inf")
    if len(Data[skill]["resources"]) < 1:
        print("No data for skill %s" % skill)
        continue
    else:
        for i in range(num_fit_initializations):
            fitmodel = random_model_uni.random_model_uni(1, 1)
            (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, Data[skill])
            if(log_likelihoods[-1] > best_likelihood):
                best_likelihood = log_likelihoods[-1]
                best_model = fitmodel
        
        if len(test_data[skill]["resources"]) > 1 and len(np.unique(test_data[skill]["data"])) > 1:#auc only calculated when there are 2+ classifiers
            (correct_predictions, state_predictions) = predict_onestep.run(best_model, test_data[skill])
            flat_true_values = np.empty((len(true_values[0]),))
            for i in range(len(true_values)):
                for j in range(len(true_values[0])):
                    if test_data[skill]["data"][i][j] != 0:
                        flat_true_values[j] = test_data[skill]["data"][i][j]

            print("Skill %s of %s calculation completed with AUC of %.4f" % (skill, skill_count, curr_auc))

        else:
            print("No test data for skill %s" % skill)
            
total_auc = auc.compute_auc(test_data[skill]["data"], correct_predictions, False)
print("Overall AUC:", total_auc)
#specifying verbose allows data from all iterations of crossvalidation to be printed out
#crossvalidate.crossvalidate(data, verbose=True)