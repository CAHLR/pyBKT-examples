# pyBKT-examples
A collection of utility and example files evaluating the validity of pyBKT, a python implementation of different variants of Bayesian Knowledge Tracing algorithms to model student learning rates of particular skills. Examples and utility files created by Frederic Wang (fredwang@berkeley.edu) using Professor Zachary Pardos' (zp@berkeley.edu) work from the pyBKT library (https://github.com/CAHLR/pyBKT). Example data sets are from Piech (https://github.com/chrispiech/DeepKnowledgeTracing), Assistments (https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect) and Cognitive Tutors (https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp).

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

## Standard KT Model, KT+Forgets (BasicForgets.py) 
In general, the predictive accuracy of the models generated by pyBKT are in line with what others have observed about BKT. For instance, in [1], they have found that by fitting the basic BKT model on the train/test split provided by from the 2009-2010 ASSISTments data set for each skill, discarding all students with only one trial, an AUC \footnote{Note that while works [2] have shown that the accepted metric of model comparison for BKT is RMSE , AUC comparisons are used for consistency with referenced publications.} of 0.73 was obtained, where pyBKT creates a model with an AUC of 0.76. Similarly, when [1] added the forgets parameter into their model, they achieved an AUC of 0.83, exactly the same AUC we achieved using pyBKT.

## Item Difficulty Effect Model (KtIdem.py)
### Cognitive Tutor Data (Low Trials/Classes Ratio)

|                   Skill                  |  Responses/Template  |  KT AUC  |  IDEM AUC  |  KT RMSE |  IDEM RMSE |
|:-----------------------------------------------:|:----------:|:-----:|:-----:|:-----:|:-----:|
| [SkillRule: Isolate positive; x+a=b, positive]  |    1.57    | 0.669 | 0.588 | 0.457 | 0.525 |
| simplify-fractions-sp                           |    2.42    | 0.607 | 0.645 | 0.442 | 0.458 |
| Identify no more factors                        |    5.09    | 0.518 | 0.679 | 0.369 | 0.398 |
| Calculate percent out of context                |    3.91    | 0.565 | 0.531 | 0.410 | 0.490 |
| Enter improper fraction from given model        |    10.76   | 0.627 | 0.554 | 0.453 | 0.508 |
| Enter numerator of percent change with variable |    5.87    | 0.650 | 0.690 | 0.280 | 0.318 |
| Compare fractions from contextual problem       |    6.90    | 0.564 | 0.527 | 0.484 | 0.547 |
| combine-like-terms-sp                           |    1.81    | 0.723 | 0.639 | 0.425 | 0.488 |
| Identify fraction using number line             |    16.26   | 0.707 | 0.702 | 0.394 | 0.405 |
| Calculate sum with negative integer             |    48.24   | 0.491 | 0.546 | 0.233 | 0.235 |
| [SkillRule: Isolate negative; x+a=b, negative]  |    1.34    | 0.632 | 0.518 | 0.469 | 0.563 |
| Calculate percent from given decimal            |    11.80   | 0.605 | 0.719 | 0.416 | 0.405 |

The alternative models supported by pyBKT have also shown to be in line with other published references. For instance, using the item difficulty effect model on twelve randomly selected skills in Cognitive Tutor’s 2006-2007 Bridge to Algebra, with different problems treated as different guess and slip classes, we got no significant change in AUC compared to the regular model, and perform even worse in terms of RMSE. Yet upon closer inspection, there is a trend between higher ratios of data per problem and larger increases of AUC for the KT-IDEM model (r=0.300), closely following the results demonstrated in [3]. Unfortunately, due to the relatively low average trials per classes ratio of this data set, the RMSE using the standard Knowledge Tracing model was much lower than the RMSE of the KT-IDEM model for this data set, and KT-IDEM only performs better than regular BKT in one skill out of the twelve selected.

### ASSISTments Data (High Trials/Classes Ratio)

|                   Skill                  |  Responses/Template  |  KT AUC  |  IDEM AUC  |  KT RMSE |  IDEM RMSE |
|:----------------------------------------:|:----------:|:-----:|:-----:|:-----:|:-----:|
| Percent Of                               | 1636       | 0.886 | 0.919 | 0.339 | 0.315 |
| Addition and Subtraction Integers        | 2862       | 0.785 | 0.808 | 0.405 | 0.414 |
| Conversion of Fraction Decimals Percents | 1105       | 0.648 | 0.732 | 0.460 | 0.442 |
| Volume Rectangular Prism                 | 9745       | 0.973 | 0.984 | 0.133 | 0.131 |
| Venn Diagram                             | 3046       | 0.897 | 0.920 | 0.309 | 0.280 |
| Equation Solving Two or Fewer Steps      | 1018       | 0.612 | 0.671 | 0.465 | 0.457 |
| Volume Cylinder                          | 13730      | 0.967 | 0.966 | 0.214 | 0.214 |
| Multiplication and Division Integers     | 1298       | 0.650 | 0.601 | 0.387 | 0.427 |
| Area Rectangle                           | 2139       | 0.987 | 0.969 | 0.100 | 0.094 |
| Addition and Subtraction Fractions       | 1259       | 0.677 | 0.706 | 0.442 | 0.437 |

On the other hand, when using the KT-IDEM model on the 2009-2010 skill builder ASSISTments data set on the ten skills with the most trials, using different templates as the guess class, we achieve an average AUC increase of 0.019319, which is very close to the 0.021 average increase reported by [3]. These results are displayed in Table 5. Since the ASSISTments data has a very high average trials to classes ratio (as no skill has a ratio below 1000), the KT-IDEM model performs very well compared to the standard BKT model when using RMSE as the metric of comparison, being lower or equal in nine of the ten skills selected.

## Prior Per Student (KtPPS.py)
The implementation of the Prior Per Student model relies on adjustable learned priors using the first response heuristic to determine between which prior is used. To do so, we set the true prior P(L0) to 0 and created fake empty responses at the beginning of each student's responses, each with learn rate class based on correctness of first response, and thus the learn rates of these fake responses are used as the calculated priors in this model.

When running the Prior Per Student model on Pardos' 42 Problem Sets GLOPs data set using the adjustable parameter algorithm, we perform better on 28 out of 42 of the problem sets compared to standard BKT, which only performs better on 14 out of 42 of the problem sets. According to [4], by using the same heuristic but with a fixed ad-hoc guess and slip parameter (0.15 and 0.1, respectively) throughout the BKT algorithm, they were able to perform better with Prior Per Student on 30 out of 42 of the problem sets. The small difference in prediction accuracy of this model can be attributed to differences in the algorithm regarding fixed parameters, but the similar performance is still promising.

In general, the larger the difference of the learned priors, the more successful the model is in predicting, while small differences perform very similarly to the basic BKT model. Thus, while this model performs well on this specific GLOPs data set, it also generally leads to unimpressive results, those of which are similar to regular BKT, on publicly released ASSISTments and Cognitive Tutor data as the priors for the majority of skills are already very high. Thus, differences of first responses are likely to be attributed to mostly slip errors rather than individual prior knowledge.

## Item Order Effect (ItemOrderEffect.py)
|     Template Pair     |  30041-30046 |  30046-30041 |  30304-30328 |  30328-30304 |  52495-52497 |  52497-52495 |
|:-------------------:|:------:|:------:|:------:|:------:|:------:|:------:|
|  Learn Rate  | 0.2242 | 0.1116 | 0.2599 | 0.4274 | 0.2330 | 0.3763 |

Unfortunately, the data set referenced by the Item Order Effect paper [5] is no longer available, so we instead run tests on ASSISTments data using template id as classifiers. ASSISTments data tends to choose from two templates per student, so the item order effect can help us determine which order of templates we should pull problems from. Running the model on the Venn Diagram skill, we can see that certain orderings of templates result in much greater learning rates of a skill. From the large differences in learn rates stemming from different orderings of templates shown, we can see that it is generally favorable to a student's learning to order problems of template id 30041 before those of template id 30046, problems of template id 30328 before those of template id 30304, and problems of template id 52497 before those of template id 52495. Using five fold cross-validation to calculate RMSE for this model, we achieve an RMSE of 0.295, an improvement from the RMSE of the basic model, which was 0.309.

## Item Learning Effect (ItemLearningEffect.py)

|     Template ID     |  30041 |  30046 |  30328 |  30304 |  52495 |  52497 |
|:-------------------:|:------:|:------:|:------:|:------:|:------:|:------:|
| Learn Rate | 0.5690 | 0.0031 | 0.3897 | 0.1585 | 0.0221 | 0.0048 |

Similar to the Item Order Effect references, the data sets used in the Item Learning Effect paper [6] are also no longer available. To showcase the potential of the model, we fit the Item Learning Effect Model to the Venn Diagram skill in ASSISTments and observed the calculated learn rates of each template. From the model results, it is clear that template 30041 results in much greater learning compared to template 30046. A similar trend is seen for template 30328 compared to template 30304, and template 52495 compared to template 52497. By predicting student responses using five fold cross-validation and the Item Learning Effect model, we are able to achieve an AUC of 0.295, a similar improvement to that shown using the Item Order Effect model.



# References
1. Khajah, M., Lindsey, R. V., & Mozer, M. C. (2016). How deep is knowledge tracing?. arXiv preprint arXiv:1604.02416.
2. Pelánek, R. (2018). The details matter: methodological nuances in the evaluation of student models. User Modeling and User-Adapted Interaction, 28(3), 207-235.
3. Pardos, Z. A., & Heffernan, N. T. (2011, July). KT-IDEM: Introducing item difficulty to the knowledge tracing model. In International conference on user modeling, adaptation, and personalization (pp. 243-254). Springer, Berlin, Heidelberg.
4. Pardos, Z. A., & Heffernan, N. T. (2010, June). Modeling individualization in a bayesian networks implementation of knowledge tracing. In International Conference on User Modeling, Adaptation, and Personalization (pp. 255-266). Springer, Berlin, Heidelberg.
5. Pardos, Z. A., & Heffernan, N. T. (2009). Determining the Significance of Item Order in Randomized Problem Sets. International Working Group on Educational Data Mining.
6. Pardos, Z. A., & Heffernan, N. T. (2009, July). Detecting the Learning Value of Items In a Randomized Problem Set. In AIED (pp. 499-506).

