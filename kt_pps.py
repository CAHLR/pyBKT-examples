import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit
from utils import data_helper, check_data
np.seterr(divide='ignore', invalid='ignore')

skill_name = "Box and Whisker"

#data!
data = data_helper.convert_data("as.csv", skill_name, multiprior=True)
check_data.check_data(data)
num_learns = len(data["resource_names"])
num_gs = len(data["gs_names"])

num_fit_initializations = 5
best_likelihood = float("-inf")

for i in range(num_fit_initializations):
    fitmodel = random_model_uni.random_model_uni(num_learns, num_gs) # include this line to randomly set initial param values
    #set prior to 0
    fitmodel["pi_0"] = np.array([[1], [0]])
    fitmodel["prior"] = 0
    (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
    if(log_likelihoods[-1] > best_likelihood):
        best_likelihood = log_likelihoods[-1]
        best_model = fitmodel

#treat learn rates of false timeslices as prior rate
print('')
print('Trained model for %s skill given %d learning rates, %d guess/slip rate' % (skill_name, num_learns, num_gs))
print('\t\tlearned')
print('prior\t\t%.4f' % (best_model["pi_0"][1][0]))
for key, value in data["resource_names"].items():
  if key != "N/A":
    print('Prior: %s\t\t%.4f' % (key, best_model['As'][value-1, 1, 0].squeeze()))
print('Learn Rate: \t\t%.4f' % (best_model['As'][0, 1, 0].squeeze()))
print('Forget Rate: \t\t%.4f' % (best_model['As'][0, 0, 1].squeeze()))

for key, value in data["gs_names"].items():
    print('Guess: %s\t\t%.4f' % (key, best_model['guesses'][value-1]))
for key, value in data["gs_names"].items():
    print('Slip: %s\t\t%.4f' % (key, best_model['slips'][value-1]))
