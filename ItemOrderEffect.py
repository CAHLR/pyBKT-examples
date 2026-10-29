import sys
sys.path.append('../')
import numpy as np
from pyBKT.models import Model
import results
np.seterr(divide='ignore', invalid='ignore')

if __name__ == '__main__':
    #only one skill is shown due to the high time complexity needed to fit the item order effect model
    skills = ["Venn Diagram"]

    model = Model(seed = 0, num_fits = 10)
    print("BKT")
    bkt = model.crossvalidate(data_path = "data/as.csv", skills = skills, metric = "rmse")
    print(bkt)
    print()
    print("Item Order Effect")
    # A new Model for each crossvalidate call, so the second keeps seed and num_fits (CAHLR/pyBKT#120).
    model = Model(seed = 0, num_fits = 10)
    item_order = model.crossvalidate(data_path = "data/as.csv", skills = skills, multipair = True, metric = "rmse")
    print(item_order["rmse"].values)
    results.save("ItemOrderEffect", {"rmse_bkt": bkt["rmse"].to_dict(), "rmse_item_order": item_order["rmse"].to_dict()})
