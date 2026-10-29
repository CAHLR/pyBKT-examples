import sys
sys.path.append('../')
import numpy as np
from pyBKT.models import Model
import results
np.seterr(divide='ignore', invalid='ignore')

if __name__ == '__main__':
    skills = ["Percent Of", "Addition and Subtraction Integers", "Conversion of Fraction Decimals Percents", "Volume Rectangular Prism", "Venn Diagram", "Equation Solving Two or Fewer Steps", "Volume Cylinder", "Multiplication and Division Integers", "Area Rectangle", "Addition and Subtraction Fractions", ]

    model = Model(seed = 0, num_fits = 20)
    print("BKT")
    #print(model.crossvalidate(data_path = "data/as.csv", skills = skills, metric = "auc"))
    bkt = model.crossvalidate(data_path = "data/as.csv", skills = skills, metric = "rmse")
    print(bkt)
    print()
    print("KT-IDEM")
    #print(model.crossvalidate(data_path = "data/as.csv", skills = skills, multigs = True, metric = "auc"))
    # A new Model for each crossvalidate call, so the second keeps seed and num_fits (CAHLR/pyBKT#120).
    model = Model(seed = 0, num_fits = 20)
    idem = model.crossvalidate(data_path = "data/as.csv", skills = skills, multigs = True, metric = "rmse")
    print(idem)
    results.save("KtIdem", {"rmse_bkt": bkt["rmse"].to_dict(), "rmse_idem": idem["rmse"].to_dict()})
