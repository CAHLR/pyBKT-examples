import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit
from utils import crossvalidate, data_helper, check_data
from copy import deepcopy
np.seterr(divide='ignore', invalid='ignore')

num_fit_initializations = 20
skill_name = "Range"

#data!
data = data_helper.convert_data("as.csv", skill_name)

check_data.check_data(data)

#specifying verbose allows data from all iterations of crossvalidation to be printed out
crossvalidate.crossvalidate(data, verbose=True)
