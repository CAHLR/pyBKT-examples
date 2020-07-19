# pyBKT-examples
A collection of utility and example files demonstrating the usage of pyBKT, a python implementation of different variants of Bayesian Knowledge Tracing algorithms to model student learning rates of particular skills. Examples and utility files created by Frederic Wang (fredwang@berkeley.edu) using Professor Zachary Pardos' (zp@berkeley.edu) work from the pyBKT library (https://github.com/CAHLR/pyBKT). Example data sets are from Piech (https://github.com/chrispiech/DeepKnowledgeTracing), Assistments (https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect) and Cognitive Tutors (https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp).

# Installation
Dependencies: numpy, pandas, requests, and sklearn (All of which are likely already installed on your device)

Install the pyBKT library
```
pip install pyBKT
```
Clone pyBKT-examples to your computer:
```
git clone https://github.com/CAHLR/pyBKT-examples.git
```
Navigate to the cloned repository and run
```
python3 <example.py>
```

# Examples Included
## Supported Model Variants
### basic (basic.py)
The simple BKT model, calculates a single value of prior knowledge, learn, forget, guess, and slip rates for all data given by the skill provided.
### kt_idem (kt_idem.py)
Uses the kt\_idem model, which takes into account different guess and slip rates per question type.
### item_learning_effect (item_learning_effect.py)
Uses the item\_learning\_effect model, which takes into account different learning and forget rates for each question type.
### item_order_effect (item_order_effect.py)
Uses the item\_order\_effect model, which takes into account different learning and forget rates for each unique pair of consecutive items.
### kt_pps (kt_pps.py)
Uses the kt\_pps model, calculates different prior knowledge rates based on whether the student answers correctly on their first attempt.

## Model Evaluation Files
### crossvalidate_basic.py
Simple demonstration of crossvalidation on the basic BKT model.

### crossvalidate_cognitive.py, crossvalidate_assistments.py
Showcases the accuracy (assuming 50% threshold for incorrect/correct), RMSE, and AUC for different BKT model variants on real world Assistments and Cognitive Tutor data.

## Synthetic Data Testing
### test_predict_onestep.py
Predicts the probability of at each timestep of the student answering the question correctly given a hand built model.

### hand_specified_model3.py
Sees how well the model performs when data is generated using specified parameters.

# Utility Functions
### data\_helper.convert\_data(url, skill\_name, df=None, return_df=False,  defaults=None, multiguess=False, multilearn=False, multiprior=False, multipair=False)
Converts a given csv or txt file into a data structure that can be passed into pyBKT's modeling and utility functions such as EM\_Fit, crossvalidate, and predict\_one\_step.

**Parameters:**
* URL: either a local filepath, web URL, or pandas DataFrame to read data from, in txt or csv format. Note: if a web URL is passed in, data_helper will save a local copy of the retrieved csv in the data/ folder for faster retrieval for future runs.
* skill\_name: name of specific skill to gather data for.
* return_df: optional argument, returns dataframe (without modifications based on multiple resources types, guess types, etc.) if True.
* defaults: optional argument, defaults to None, can be passed in as adictionary specifying custom columns for skill name, user id, etc. If provided URL/data set is not in the format of assistments or cognitive tutor, all columns must be set here. All column name keys: 
    - 'order_id', specifies the ordering of data (usually defaults to time).
    - 'skill_name', specifies column to search for inputted skill\_name.
    - 'correct', specifies column that determines if a student answers correctly.
    - 'user_id', specifies column to differentiate between different students.
    - 'multilearn', specifies column for item\_learning\_effect.
    - 'multiprior', specifies column for kt_pps.
    - 'multipair', specifies column for item\_order\_effect.
    - 'multiguess', specifies column for kt_idem.
* multiguess: defaults to False, uses kt_idem model if True. 
* multilearn: defaults to False, uses item\_learning\_effect model if True.
* multiprior: defaults to False, uses kt_pps model if True.
* multipair: defaults to False, uses item\_order\_effect model if True.

### nips_data_helper.convert_data(url, url2=None)
Converts data in the format of that provided by https://github.com/chrispiech/DeepKnowledgeTracing into a data structure that can be passed into pyBKT's functions.

**Parameters:**
* URL: a local filepath to read data from in csv format (specific data located in data/builder_train.csv and data/builder_test.csv)
* URL2: another local filepath in the format of URL, used to join the data provided by URL and URL2 into a single data structure in order to perform crossvalidation on the complete data set.

### crossvalidate.crossvalidate(data, folds=5, verbose=False, seed=0, return_arrays=False)
Performs crossvalidation on data, returning a tuple containing accuracy and RMSE.

**Parameters:**
* data: input data, in the format of that returned by data\_helper.convert\_data.
* folds: optional argument specifying how many folds the crossvalidation runs on.
* verbose: optional argument that prints out learned model, RMSE, and accuracy for each iteration if True.
* seed: optional argument specifying the seed in which the data is partitioned into by k-fold crossvalidation.
* return_arrays: optional argument determining if crossvalidate should return the tuple (accuracy, rmse, auc) if false or (true values, predicted values) if false

### check\_data.check\_data(data)
Checks input data structure to see if it's dimensions are valid for pyBKT functionality. Throws an error if it is not.

**Parameters:**
* data: input data, hopefully in the format of that returned by data\_helper.convert\_data.

### accuracy.compute\_acc(true\_values, pred\_values), rmse.compute\_rmse(true\_values, pred\_values), auc.compute\_auc(true\_values, pred\_values)
Computes accuracy, rmse, and AUC respectively, on test data using model from training data.

**Parameters:**
* true\_values: an array of 1's and 2's depending on whether the student answered incorrectly or correctly at each step.
* pred\_values: an array of real numbers from [0, 1] as probabilities to whether the student will answer correctly at each step.

