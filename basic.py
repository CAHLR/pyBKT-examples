import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit
from utils import data_helper, check_data
from copy import deepcopy
np.seterr(divide='ignore', invalid='ignore')

num_fit_initializations = 20

#data!
data = data_helper.convert_data("as.csv", "Box and Whisker")
check_data.check_data(data)
num_learns = len(data["resource_names"])
num_gs = len(data["gs_names"])
num_fit_initializations = 5
best_likelihood = float("-inf") 

for i in range(num_fit_initializations):
	fitmodel = random_model_uni.random_model_uni(num_learns, num_gs) # include this line to randomly set initial param values
	(fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
	if(log_likelihoods[-1] > best_likelihood):
		best_likelihood = log_likelihoods[-1]
		best_model = fitmodel

print('')
print('Trained model given %d learning rates, %d guess/slip rate' % (num_learns, num_gs))
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
