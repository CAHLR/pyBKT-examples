import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit, predict_onestep
from utils import accuracy, rmse, check_data, auc
from copy import deepcopy

# returns data only for the indices given based on starts array
def fix_data(data, indices):
    training_data = {}
    resources = []
    d = [[] for _ in range(len(data["data"]))]
    start_temp = [data["starts"][i] for i in indices]
    length_temp = [data["lengths"][i] for i in indices]
    if "resource_names" in data:
        training_data["resource_names"] = data["resource_names"]
    if "gs_names" in data:
        training_data["gs_names"] = data["gs_names"]
    starts = []
    for i in range(len(start_temp)):
        starts.append(len(resources)+1)
        #print("A", start_temp[i], start_temp[i]+length_temp[i])
        resources.extend(data["resources"][start_temp[i]-1:start_temp[i]+length_temp[i]-1])
        for j in range(len(data["data"])):
            d[j].extend(data["data"][j][start_temp[i]-1:start_temp[i]+length_temp[i]-1])
    training_data["starts"] = np.asarray(starts)
    training_data["lengths"] = np.asarray(length_temp)
    training_data["data"] = np.asarray(d,dtype='int32')
    resource=np.asarray(resources)
    stateseqs=np.copy(resource)
    training_data["stateseqs"]=np.asarray([stateseqs],dtype='int32')
    training_data["resources"]=resource
    training_data=(training_data)
    return training_data

def crossvalidate(data, folds=5, verbose=False, seed=0, return_arrays=False):

    if "resource_names" in data:
        num_learns = len(data["resource_names"])
    else:
        num_learns = 1
        
    if "gs_names" in data:
        num_gs = len(data["gs_names"])
    else:
        num_gs = 1
        
    total = 0
    acc = 0
    area_under_curve = 0
    num_fit_initializations = 20
    split_size = (len(data["starts"])//folds)
    #create random permutation to act as indices for folds for crossvalidation
    shuffle = np.random.RandomState(seed=seed).permutation(len(data["starts"]))
    all_true, all_pred = [], []

    # crossvalidation on students which are identified by the starts array
    for iteration in range(folds):
        #create training/test data based on random permutation from earlier
        train = np.concatenate((shuffle[0:iteration*split_size],shuffle[(iteration+1)*split_size:len(data["starts"])]))
        test = shuffle[iteration*split_size:(iteration+1)*split_size]
        training_data = fix_data(data, train)
        num_fit_initializations = 5
        best_likelihood = float("-inf")

        for i in range(num_fit_initializations):
        	fitmodel = random_model_uni.random_model_uni(num_learns, num_gs) # include this line to randomly set initial param values
        	(fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, training_data)
        	if(log_likelihoods[-1] > best_likelihood):
        		best_likelihood = log_likelihoods[-1]
        		best_model = fitmodel
	
        if verbose:
            print(" ")
            print('Iteration %d' % (iteration))
            print('\tlearned')
            print('prior\t%.4f' % (best_model["pi_0"][1][0]))
            for r in range(num_learns):
                print('learn%d\t%.4f' % (r+1, best_model['As'][r, 1, 0].squeeze()))
            for r in range(num_learns):
                print('forget%d\t%.4f' % (r+1, best_model['As'][r, 0, 1].squeeze()))
            
            for s in range(num_gs):
                print('guess%d\t%.4f' % (s+1, best_model['guesses'][s]))
            for s in range(num_gs):
                print('slip%d\t%.4f' % (s+1, best_model['slips'][s]))
                
        
        test_data = fix_data(data, test)
        
        # run model predictions from training data on test data
        (correct_predictions, state_predictions) = predict_onestep.run(best_model, test_data)
        
        flat_true_values = np.zeros((len(test_data["data"][0]),), dtype=np.intc)
        for i in range(len(test_data["data"])):
            for j in range(len(test_data["data"][0])):
                if test_data["data"][i][j] != 0:
                    flat_true_values[j] = test_data["data"][i][j]
        flat_true_values = flat_true_values.tolist()
        
       # print(len(flat_true_values))
       # print(len(correct_predictions))
       # print(auc.compute_auc(flat_true_values, correct_predictions))
        all_true.extend(flat_true_values)
        all_pred.extend(correct_predictions)

        
    if return_arrays:
        return (all_true, all_pred)
        
   # print(len(all_true))
    print(len(all_pred))
    total += rmse.compute_rmse(all_true, all_pred)
    acc += accuracy.compute_acc(all_true, all_pred)
    area_under_curve += auc.compute_auc(all_true, all_pred)
    if verbose:
        print("Average RMSE: ", total)
        print("Average Accuracy: ", acc)
        print("Average AUC: ", area_under_curve)
    return (acc, total, area_under_curve)

