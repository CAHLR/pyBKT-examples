#compares accuracy and rmse of different models, may take a LONG time to run since many models (25 total) are generated
#item_order_effect can be especially slow since many different pairs are possible
#assistments data set does not include templates, so multiguess will perform worse than expected
import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit
from utils import crossvalidate, accuracy, rmse, auc, check_data, data_helper, ktidem_skills
import copy
np.seterr(divide='ignore', invalid='ignore')
num_fit_initializations = 20
skill_name = "Box and Whisker"
seed, folds = 2020, 5 #can customize to anything, keep same seed and # folds over all trials
results = {} #create dictionary to store accuracy and rmse results

df, skill_list, student_count = ktidem_skills.find_skills()
for i in range(10):
    skill_name = skill_list[i]
    num_students = student_count[i]
    results[skill_name]=[num_students]
    
    data = data_helper.convert_data(df, skill_name)
    check_data.check_data(data)
    print("creating simple model")
    results[skill_name].append(crossvalidate.crossvalidate(data, folds=folds, seed=seed)[2])

    data_multiguess = data_helper.convert_data(df, skill_name, multiguess=True)
    check_data.check_data(data_multiguess)
    print("creating kt_idem model")
    results[skill_name].append(crossvalidate.crossvalidate(data_multiguess, folds=folds, seed=seed)[2])
    #print(results)

print("Model\tNum Students\tSimple AUC\tKT_IDEM AUC")
for k, v in results.items():
    print("%s\t%d\t%.5f\t%.5f" % (k, v[0], v[1], v[2]))
