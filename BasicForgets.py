import sys
sys.path.append('../')
import numpy as np
from pyBKT.models import Model
import results
np.seterr(divide='ignore', invalid='ignore')

if __name__ == '__main__':
    model = Model(seed = 0, num_fits = 20)
    model.fit(data_path = "data/builder_train_preprocessed.csv")
    auc_bkt = model.evaluate(data_path = "data/builder_test_preprocessed.csv", metric="auc")
    print("Standard BKT:", auc_bkt)
    model2 = Model(seed = 0, num_fits = 20)
    model2.fit(data_path = "data/builder_train_preprocessed.csv", forgets=True)
    auc_forgets = model2.evaluate(data_path = "data/builder_test_preprocessed.csv", metric="auc")
    print("BKT+Forgets:", auc_forgets)
    results.save("BasicForgets", {"auc_bkt": auc_bkt, "auc_forgets": auc_forgets})
