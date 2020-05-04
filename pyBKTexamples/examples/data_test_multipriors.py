import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit
from test_utils import crossvalidate
from data_utils import assistments_data_helper, check_data
np.seterr(divide='ignore', invalid='ignore')

skill_name = "Box and Whisker"

#data!
temp_data = assistments_data_helper.assistments_data("test.csv", skill_name, multipriors = True)
check_data.check_data(temp_data)
num_learns = len(temp_data["resource_names"])
num_gs = len(temp_data["gs_names"])

for correct in range(1,3):
    num_fit_initializations = 5
    best_likelihood = float("-inf")
    indices = []
    for j in range(len(temp_data["priors"])):
      if temp_data["priors"][j] == correct:
        indices.append(j)
        
    data = crossvalidate.fix_data(temp_data, indices)
    check_data.check_data(data)
    #print(data["data"].shape, data["starts"].shape, data["lengths"].shape, data["resources"].shape)
    for i in range(num_fit_initializations):
    	fitmodel = random_model_uni.random_model_uni(num_learns, num_gs) # include this line to randomly set initial param values
    	(fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
    	if(log_likelihoods[-1] > best_likelihood):
    		best_likelihood = log_likelihoods[-1]
    		best_model = fitmodel
    
    print('')
    print('Trained model for %s skill given %d learning rates, %d guess/slip rate, and prior rate %d' % (skill_name, num_learns, num_gs, correct))
    print('\t\tlearned')
    print('prior\t\t%.4f' % (best_model["pi_0"][1][0]))
    for key, value in data["resource_names"].items():
        print('Learn: %s\t\t%.4f' % (key, best_model['As'][value-1, 1, 0].squeeze()))
    for key, value in data["resource_names"].items():
        print('Forget: %s\t\t%.4f' % (key, best_model['As'][value-1, 0, 1].squeeze()))
    
    for key, value in data["gs_names"].items():
        print('Guess: %s\t\t%.4f' % (key, best_model['guesses'][value-1]))
    for key, value in data["gs_names"].items():
        print('Slip: %s\t\t%.4f' % (key, best_model['slips'][value-1]))
