import sys
sys.path.append('../')
import numpy as np
import os
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit, predict_onestep
from utils import crossvalidate, accuracy, rmse, auc, check_data, data_helper, ktpps_data_helper
import copy
np.seterr(divide='ignore', invalid='ignore')
num_fit_initializations = 20
seed, folds = 2020, 5 #can customize to anything, keep same seed and # folds over all trials
results = {} #create dictionary to store accuracy and rmse results


# IMPORTANT!!! Choose whether to use all 42 problem sets or just the problem sets of length 4
all_files = os.listdir("./data/glops-exact") # all 42 sets
#all_files = os.listdir("./data/glops-exact-4") # only sets of length 4
adhoc_guess = 0.15
adhoc_slip = 0.10


pps_values={}
kt_values={}
kt_correct={}
kt_incorrect={}
pps_correct={}
pps_incorrect={}

for a in range(1, 3):
    for b in range(1, 3):
        for c in range(1, 3):
            for d in range(1, 3):
                pps_values[str(a)+str(b)+str(c)+str(d)]=0
                kt_values[str(a)+str(b)+str(c)+str(d)]=0
                kt_correct[str(a)+str(b)+str(c)+str(d)]=0
                kt_incorrect[str(a)+str(b)+str(c)+str(d)]=0
                pps_correct[str(a)+str(b)+str(c)+str(d)]=0
                pps_incorrect[str(a)+str(b)+str(c)+str(d)]=0

total_responses = 0

kt_better = 0
pps_better = 0

for i in all_files:
    if i == "README.txt" or i == ".DS_Store":
        continue
        
    print("Creating model for ", i)
        
    data, pps_data = ktpps_data_helper.convert_data(i)
    
    
    total_responses += len(data["starts"])
    
    check_data.check_data(data)
    check_data.check_data(pps_data)
    
    # first, generate the basic model and run accuracy tests using MAE as evaluator
    num_fit_initializations = 20
    best_likelihood = float("-inf")
    for i in range(num_fit_initializations):
        fitmodel = random_model_uni.random_model_uni(1, 1)
        (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
        if(log_likelihoods[-1] > best_likelihood):
            best_likelihood = log_likelihoods[-1]
            best_model = fitmodel
        
    data["lengths"] = data["lengths_full"]
    
    (correct_predictions, state_predictions) = predict_onestep.run(best_model, data)
    
    kt_mae = 0
    for i in data["starts"]:
        true = data["data"][0][i + 2] - 1
        predicted = correct_predictions[i + 2]
        kt_values[str(data["data"][0][i - 1]) + str(data["data"][0][i]) + str(data["data"][0][i + 1]) + str(round(predicted) + 1)] += 1
        kt_mae += abs(true - predicted)
        if (true == 1 and predicted > 0.5) or (true == 0 and predicted < 0.5):
            kt_correct[str(data["data"][0][i - 1]) + str(data["data"][0][i]) + str(data["data"][0][i + 1]) + str(data["data"][0][i + 2])] += 1
        elif (true == 0 and predicted > 0.5) or (true == 1 and predicted < 0.5):
            kt_incorrect[str(data["data"][0][i - 1]) + str(data["data"][0][i]) + str(data["data"][0][i + 1]) + str(data["data"][0][i + 2])] += 1
    kt_mae /= len(data["starts"])
            
    print("KT MAE:", kt_mae)

    
    
    # next, generate the pps model and run accuracy tests
    num_fit_initializations = 20
    best_likelihood = float("-inf")
    for i in range(num_fit_initializations):
       fitmodel = random_model_uni.random_model_uni(3, 1)
       fitmodel["pi_0"] = np.array([[1], [0]])
       fitmodel["prior"] = 0
       (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, pps_data)
       if(log_likelihoods[-1] > best_likelihood):
           best_likelihood = log_likelihoods[-1]
           best_model_pps = fitmodel


    best_model_pps['As'][1, 1, 0] = adhoc_slip
    best_model_pps['As'][2, 1, 0] = 1 - adhoc_guess
    best_model_pps['learns'] = best_model_pps['As'][:, 1, 0]
    
    pps_data["lengths"] = pps_data["lengths_full"]
    
    (correct_predictions, state_predictions) = predict_onestep.run(best_model_pps, pps_data)
    
    print('prior\t\t%.4f' % (best_model_pps["pi_0"][1][0]))
    print('Prior: \t\t%.4f' % (best_model_pps['As'][1, 1, 0].squeeze()))
    print('Prior: \t\t%.4f' % (best_model_pps['As'][2, 1, 0].squeeze()))
    print('Learn Rate: \t\t%.4f' % (best_model_pps['As'][0, 1, 0].squeeze()))
    print('Forget Rate: \t\t%.4f' % (best_model_pps['As'][0, 0, 1].squeeze()))

    print('Guess: \t\t%.4f' % (best_model_pps['guesses'][0]))
    print('Slip: \t\t%.4f' % (best_model_pps['slips'][0]))

        
    pps_mae = 0
    for i in pps_data["starts"]:
        true = pps_data["data"][0][i + 3] - 1
        predicted = correct_predictions[i + 3]
        pps_values[str(pps_data["data"][0][i]) + str(pps_data["data"][0][i + 1]) + str(pps_data["data"][0][i + 2]) + str(round(predicted) + 1)] += 1
        pps_mae += abs(true - predicted)
        if (true == 1 and predicted > 0.5) or (true == 0 and predicted < 0.5):
            pps_correct[str(pps_data["data"][0][i]) + str(pps_data["data"][0][i + 1]) + str(pps_data["data"][0][i + 2]) + str(pps_data["data"][0][i + 3])] += 1
        elif (true == 0 and predicted > 0.5) or (true == 1 and predicted < 0.5):
            pps_incorrect[str(pps_data["data"][0][i]) + str(pps_data["data"][0][i + 1]) + str(pps_data["data"][0][i + 2]) + str(pps_data["data"][0][i + 3])] += 1
    pps_mae /= len(pps_data["starts"])
    
    print("PPS MAE:", pps_mae)
    
    if kt_mae > pps_mae:
        pps_better += 1
    else:
        kt_better += 1
        
    
    # correct heuristic
    # pps_mae = 0
    
    # best_model["pi_0"] = [[adhoc_guess], [1 - adhoc_guess]]
    # best_model["prior"] = 1 - adhoc_guess
    # (correct_predictions, state_predictions) = predict_onestep.run(best_model, data)
    
    # for i in data["starts"]:
    #    if data["data"][0][i - 1] == 2:
    #        true = data["data"][0][i + 2] - 1
    #        predicted = correct_predictions[i + 2]
    #        pps_mae += abs(true - predicted)
    #        if (true == 1 and predicted > 0.5) or (true == 0 and predicted < 0.5):
    #            pps_correct[str(data["data"][0][i - 1]) + str(data["data"][0][i]) + str(data["data"][0][i + 1]) + str(data["data"][0][i + 2])] += 1
    #        elif (true == 0 and predicted > 0.5) or (true == 1 and predicted < 0.5):
    #            pps_incorrect[str(data["data"][0][i - 1]) + str(data["data"][0][i]) + str(data["data"][0][i + 1]) + str(data["data"][0][i + 2])] += 1
    
    
    # incorrect heuristic
    #best_model["pi_0"] = [[1 - adhoc_slip], [adhoc_slip]]
    #best_model["prior"] = adhoc_slip
    #(correct_predictions, state_predictions) = predict_onestep.run(best_model, data)
    
    #for i in data["starts"]:
    #    if data["data"][0][i - 1] == 1:
    #        true = data["data"][0][i + 2] - 1
    #        predicted = correct_predictions[i + 2]
    #        pps_mae += abs(true - predicted)
    #        if (true == 1 and predicted > 0.5) or (true == 0 and predicted < 0.5):
    #            pps_correct[str(data["data"][0][i - 1]) + str(data["data"][0][i]) + str(data["data"][0][i + 1]) + str(data["data"][0][i + 2])] += 1
    #        elif (true == 0 and predicted > 0.5) or (true == 1 and predicted < 0.5):
    #            pps_incorrect[str(data["data"][0][i - 1]) + str(data["data"][0][i]) + str(data["data"][0][i + 1]) + str(data["data"][0][i + 2])] += 1



print(kt_correct)
print(kt_incorrect)
print(pps_correct)
print(pps_incorrect)
print(kt_values)
print(pps_values)
print("PPS performs better in", pps_better, "datasets")
print("KT performs better in", kt_better, "datasets")

