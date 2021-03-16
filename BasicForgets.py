import sys
sys.path.append('../')
import numpy as np
from pyBKT.models import Model
np.seterr(divide='ignore', invalid='ignore')

model = Model(seed = 0, num_fits = 20)
model.fit(data_path = "data/builder_train_preprocessed.csv")
model.evaluate(data_path = "data/builder_test_preprocessed.csv")

model2 = Model(seed = 0, num_fits = 20)
model2.fit(data_path = "data/builder_train_preprocessed.csv", forgets=True)
model2.evaluate(data_path = "data/builder_test_preprocessed.csv")
