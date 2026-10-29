import sys
sys.path.append('../')
import numpy as np
from pyBKT.models import Model
import results
np.seterr(divide='ignore', invalid='ignore')

if __name__ == '__main__':
    skills = ["Percent Of", "Addition and Subtraction Integers", "Conversion of Fraction Decimals Percents", "Volume Rectangular Prism", "Venn Diagram", "Equation Solving Two or Fewer Steps", "Volume Cylinder", "Multiplication and Division Integers", "Area Rectangle", "Addition and Subtraction Fractions", ]

    model = Model(seed = 0, num_fits = 1)
    print("BKT")
    bkt = model.crossvalidate(data_path = "data/as.csv", skills = skills, metric = "rmse")
    print(bkt["rmse"].values)
    print()
    print("Item Learning Effect")
    # A new Model for each crossvalidate call, so the second keeps seed and num_fits (CAHLR/pyBKT#120).
    model = Model(seed = 0, num_fits = 1)
    item_learning = model.crossvalidate(data_path = "data/as.csv", skills = skills, multilearn = "problem_id", metric = "rmse")
    print(item_learning["rmse"].values)
    results.save("ItemLearningEffect", {"rmse_bkt": bkt["rmse"].to_dict(), "rmse_item_learning": item_learning["rmse"].to_dict()})
