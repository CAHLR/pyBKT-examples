This dataset is provided from the following publication. Please make appropriate reference to this work if you utilize this dataset in public work.

Pardos, Z. A., Heffernan, N. T. (Under review) Modeling Individualization in a Bayesian Networks Implementation of Knowledge Tracing. In Proceedings of the 18th International Conference on User Modeling, Adaptation and Personalization. Big Island of Hawaii.

(datset, paper and up to date citation can be found at: http://users.wpi.edu/~zpardos)

Details of the dataset (from section 3.1 of paper):

Our dataset consisted of student responses to problem sets that satisfied the following constraints:
- Items in the problem set must have been given in a random order
- A student must have answered all items in the problem set in one day
- There are at least four items in the problem set of the exact same skill tagging
- Data is from The ASSISTment System gathered from Fall 2008 to Spring 2010

Filename convention example: G4.225-exact.txt
- 'G' is for GLOP which means Groups of Learning OPportunities
- '4' indicates the number of items in the problem set
- '225' is the GLOP id of the problem set
- '-exact' specifies that the items in the problem set all have the exact same skill tagging.
   all the problem sets in this dataset are of type -exact although we have a dataset
   with more problem sets that relaxes this contraint.

Data column description: user_id response_1 response_2 ... response_N
- 0 corresponds to an incorrect first attmpet of the main problem, 1 is a correct first attempt
- responses of a student are in chronological order
- each student complted an average of 3 of the 42 problem sets in this dataset
- each problem set has on average of 312 students

2/23/2010
author Zach Pardos (zpardos@wpi.edu)
SQL queries by Matt Dailey