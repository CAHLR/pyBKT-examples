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
all_true = []
all_pred = []
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
                
        #print(" ")
        #print('\tlearned')
        #print('prior\t%.4f' % (best_model["pi_0"][1][0]))
        #for r in range(1):
        #    print('learn%d\t%.4f' % (r+1, best_model['As'][r, 1, 0].squeeze()))
        #for r in range(1):
        #    print('forget%d\t%.4f' % (r+1, best_model['As'][r, 0, 1].squeeze()))
          
        #for s in range(1):
        #    print('guess%d\t%.4f' % (s+1, best_model['guesses'][s]))
        #for s in range(1):
        #    print('slip%d\t%.4f' % (s+1, best_model['slips'][s]))

        if len(test_data[skill]["resources"]) > 0:
            (correct_predictions, state_predictions) = predict_onestep.run(best_model, test_data[skill])
            if len(np.unique(test_data[skill]["data"])) > 1:#auc for single skill only calculated when there are 2+ classifiers
                curr_auc = auc.compute_auc(test_data[skill]["data"][0], correct_predictions)
            else:
                curr_auc = 0
        
            all_true.extend(test_data[skill]["data"][0])
            all_pred.extend(correct_predictions)
            print("Skill %s of %s calculation completed with AUC of %.4f" % (skill, skill_count, curr_auc))
        else:
            print("No test data for skill %s" % skill)
total_auc = auc.compute_auc(all_true, all_pred)
print("Overall AUC:", total_auc)
