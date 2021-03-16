import sys
sys.path.append('../')
import numpy as np
from pyBKT.models import Model
np.seterr(divide='ignore', invalid='ignore')

skills = ["Percent Of", "Addition and Subtraction Integers", "Conversion of Fraction Decimals Percents", "Volume Rectangular Prism", "Venn Diagram", "Equation Solving Two or Fewer Steps", "Volume Cylinder", "Multiplication and Division Integers", "Area Rectangle", "Addition and Subtraction Fractions", ]

model = Model(seed = 0, num_fits = 20)
print("BKT")
print(model.crossvalidate(data_path = "data/as.csv", skills = skills, metric = "rmse"))
print(model.params())
print()
print("Item Learning Effect")
print(model.crossvalidate(data_path = "data/as.csv", skills = skills, multilearn = True, metric = "rmse"))
print(model.params().to_string())
