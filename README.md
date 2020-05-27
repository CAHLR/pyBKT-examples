# pyBKT-examples
A collection of utility and example files demonstrating the usage of pyBKT, a python implementation of different variants of Bayesian Knowledge Tracing algorithms to model student learning rates of particular skills. Examples and utility files created by Frederic Wang (fredwang@berkeley.edu) using Professor Zachary Pardos' (zp@berkeley.edu) work from the pyBKT library (https://github.com/CAHLR/pyBKT). Example data sets are from Assistments (https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect) and Cognitive Tutors (https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp).

# Installation
Dependencies: numpy, python, sklearn.

Install the pyBKT library:
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

# Example Files Included

**simple:** Uses the simple BKT model, calculates the overall prior, learn, forget, guess, and slip rates for all data related to the skill provided.

**kt\_idem:** Uses the kt\_idem model, which incorporates multiple guess and slip rates.

**item\_learning\_effect:** Uses the item\_learning\_effect model, incorporates multiple learning and forget rates.

**item\_order\_effect:** Uses the item\_order\_effect model, incorporates different learning and forget rates for each unique pair of consecutive items.

**kt_pps:** Uses the kt\_pps model, calculates different prior rates based on whether the student answers correctly on their first attempt.

**test_defaults:** Uses custom column specifications for multiple learn and guess rates.

**test\_predict\_onestep:** Predicts the probability of at each timestep of the student answering the question correctly given a hand built model.

**test_url:** Retrieves data from the web instead of locally.

**crossvalidate_comparison:** Compares accuracy and RMSE of the different supported models of pyBKT on the same data set.

**crossvalidate_simple:** Performs crossvalidation using one model and prints out all intermediate data and calculations.

**hand\_specified\_model3:** Sees how well the model performs when data is built using specified parameters.

# Utility Functions
### data\_helper.convert\_data(url, skill\_name, defaults=None, multiguess=False, multilearn=False, multiprior=False, multipair=False)
Converts a given csv or txt file into a data structure that can be passed into pyBKT's modeling and utility functions such as EM\_Fit, crossvalidate, predict\_one\_step.

**Parameters:**
* URL: either a local filepath or URL to read data from, in txt or csv format.
* skill\_name: name of specific skill to gather data.
* defaults: optional argument, defaults to None, dictionary specifying custom columns for skill name, user id, etc. If data set is not assistments or cognitive tutor, all columns must be set here. All column name keys: 
    - 'order_id', specifies the ordering of data (usually defaults to time).
    - 'skill_name', specifies column to search for inputted skill\_name.
    - 'correct', specifies column that determines if a student answers correctly.
    - 'user_id', specifies column to differentiate between different students.
    - 'multilearn', specifies column for item\_learning\_effect.
    - 'multiprior', specifies column for kt_pps.
    - 'multipair', specifies column for item\_order\_effect.
    - 'multiguess', specifies column for kt_idem.
* df: optional argument, if not None uses passed in data frame instead of retrieving from URL/file path
* save\_df: optional argumment, defaults to False, also returns the data frame retrieved from URL/file path along with the data structure.
* multiguess: optional argument, defaults to False, uses kt_idem model if True. 
* multilearn: optional argument, defaults to False, uses item\_learning\_effect model if True.
* multiprior: optional argument, defaults to False, uses kt_pps model if True.
* multipair: optional argument, defaults to False, uses item\_order\_effect model if True.


### crossvalidate.crossvalidate(data, folds=5, verbose=False, seed=0)
Performs crossvalidation on data, returning a tuple containing accuracy and RMSE.

**Parameters:**
* data: input data, in the format of that returned by data\_helper.convert\_data.
* folds: optional argument, defaults to 5, specifying how many folds the crossvalidation runs on.
* verbose: optional argument, defaults to False, prints out learned model, RMSE, and accuracy for each iteration if True.
* seed: optional argument, defaults to 0, specifying the seed in which the data is partitioned into by k-fold crossvalidation.

### check\_data.check\_data(data)
Checks input data structure to see if it's dimensions are valid for pyBKT functionality.

**Parameters:**
* data: input data, hopefully in the format of that returned by data\_helper.convert\_data.

### accuracy.compute\_acc(true\_values, pred\_values, verbose=False) and rmse.compute\_rmse(true\_values, pred\_values, verbose=False)
Computes accuracy and rmse, respectively, on test data using model from training data.

**Parameters:**
* true\_values: an array of 1's and 2's depending on whether the student answered incorrectly or correctly at each step.
* pred\_values: an array of real numbers from [0, 1] as probabilities to whether the student will answer correctly at each step.
* verbose: optional argument, defaults to False, prints out computed results if True.