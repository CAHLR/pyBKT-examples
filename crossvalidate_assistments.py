#compares accuracy and rmse of different models, may take a LONG time to run since many models (25 total) are generated
#item_order_effect can be especially slow since many different pairs are possible
#assistments data set does not include templates, so multiguess will perform worse than expected
import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit
from utils import crossvalidate, accuracy, rmse, check_data, data_helper
import copy
np.seterr(divide='ignore', invalid='ignore')
num_fit_initializations = 20
skill_name = "Box and Whisker"
seed, folds = 2020, 5 #can customize to anything, keep same seed and # folds over all trials
results = {} #create dictionary to store accuracy and rmse results


print("starting kt_idem data collection")
data_multiguess = data_helper.convert_data("as.csv", skill_name, multiguess=True)
check_data.check_data(data_multiguess)
print("creating kt_idem model")
results["Multiguess"] = crossvalidate.crossvalidate(data_multiguess, folds=folds, seed=seed)


results = {k: v for k, v in sorted(results.items(), key=lambda item: -item[1][0])}
print("Model\t\tAccuracy\tRMSE")
for k, v in results.items():
    print("%s\t%.5f\t\t%.5f" % (k, v[0], v[1]))
