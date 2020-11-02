import sys
sys.path.append('../')
import numpy as np
from pyBKT.generate import synthetic_data, random_model_uni
from pyBKT.fit import EM_fit
from utils import crossvalidate, accuracy, rmse, auc, check_data, data_helper, ktidem_skills
import copy
np.seterr(divide='ignore', invalid='ignore')
num_fit_initializations = 20
seed, folds = 2020, 5 #can customize to anything, keep same seed and # folds over all trials
results = {} #create dictionary to store accuracy and rmse results

df, skill_list, student_count, data_count, template_count = ktidem_skills.find_skills()
for i in range(10):
    skill_name = skill_list[i]
    results[skill_name]=[student_count[i], data_count[i], template_count[i]]
    
    data = data_helper.convert_data(df, skill_name)
    check_data.check_data(data)
    results[skill_name].append((np.sum(data["data"][0]) - len(data["data"][0]))/len(data["data"][0]))
    print("creating simple model")
    results[skill_name].append(crossvalidate.crossvalidate(data, folds=folds, seed=seed)[2])

    data_multiguess = data_helper.convert_data(df, skill_name, multiguess=True)
    check_data.check_data(data_multiguess)
    print("creating kt_idem model")
    results[skill_name].append(crossvalidate.crossvalidate(data_multiguess, folds=folds, seed=seed)[2])
    #print(results)

print("Model\tNum Students\tNum Data\tNum Templates\tCorrect Percent\tSimple AUC\tKT_IDEM AUC")
for k, v in results.items():
    print("%s\t%d\t%d\t%d\t%.5f\t%.5f\t%.5f" % (k, v[0], v[1], v[2], v[3], v[4], v[5]))
