import sys
sys.path.append('../')
import numpy as np
from pyBKT.models import Model
np.seterr(divide='ignore', invalid='ignore')

#only one skill is shown due to the high time complexity needed to fit the item order effect model
skills = ["Venn Diagram"]

model = Model(seed = 0, num_fits = 10)
print("BKT")
print(model.crossvalidate(data_path = "data/as.csv", skills = skills, metric = "rmse"))
print()
print("Item Order Effect")
print(model.crossvalidate(data_path = "data/as.csv", skills = skills, multipair = True, metric = "rmse")["rmse"].values)
