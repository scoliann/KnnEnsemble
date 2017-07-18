This project demonstrates how to adapt KNN algorithm to be used effectively in an ensemble.  Performance results of the ensemble vs. baseline KNN against 20 datasets from the UCI Machine Learning Repository are presented.

## Inspiration
While trying to predict changes in the price of Bitcoin, I found that the KNN classifier gave the best results.  Wanting to further improve the results led me to learning about ensemble techniques, especially the unique formulations that have been developed to improve KNN.

## Introduction
Improving the performance of KNN using an ensemble is no easy task.  The "typical" ensemble techniques of bagging and boosting do not work with KNN, as KNN is already a stable classifier.  Improving performance with an ensemble requires that the component classifiers are accurate and "diverse" (ie. have uncorrelated errors).  We must look beyond bagging and boosting to find new ways to add diversity to the component classifiers, while maintaining good individual classifier performance.

## Diversity Methods
1.  Bagging
2.  Perturbing the distance metric
3.  Perturbing the feature set
4.  Perturbing the K values

Zhou and Yu have shown that an algorithm using methods 1 - 3, FASBIR, achieves excellent results.  In addition, my code allows users to perturb the K values of component classifiers as well.

## Voting Methods
As Domeniconi and Yan have demonstrated, different voting methods yield different results.  In this experiment, I have implemented six different voting methods.  The names and descriptions for each of these can be found in the comments at the start of `VotingMethodsClassifier.py `.

## Datasets
All datasets are from the UCI Machine Learning Repository.  Links to each of the datasets can be found in the comments of `readInDatasets.py`.  A table detailing some important information about each of the datasets is listed below:

| Dataset           | Size | # Continuous Features | # Categorical Features | # Classes |
| ----------------- |:----:| ---------------------:| ----------------------:| ---------:|
| anneal            | 898  | 6 | 32 | 6 |
| auto              | 205  | 15 | 10 | 7 |
| balanceScale      | 625  | 4 | 0 | 3 |
| breastTissue      | 106  | 9 | 0 | 6 |
| creditApproval    | 690  | 6 | 9 | 2 |
| diabetes          | 768  | 8 | 0 | 2 |
| germanCredit      | 1000 | 7 | 13 | 2 |
| glass             | 214  | 9 | 0 | 7 |
| heart             | 270  | 7 | 6 | 2 |
| hepatitis         | 155  | 6 | 13 | 2 |
| imageSegmentation | 2310 | 19 | 0 | 7 |
| indianLiver       | 583  | 9 | 1 | 2 |
| ionosphere        | 351  | 0 | 34 | 2 |
| iris              | 150  | 4 | 0 | 3 |
| liverDisorders    | 345  | 6 | 0 | 2 |
| sonar             | 208  | 60 | 0 | 2 |
| soybean           | 683  | 0 | 35 | 19 |
| vehicle           | 846  | 18 | 0 | 4 |
| vote              | 435  | 0 | 16 | 2 |
| vowel             | 990  | 10 | 0 | 11 |

## Encoding Categorical Features
For classification problems one might wonder how to encode categorical features.  I believe there are two acceptable ways to do this.  Firstly, one could use one hot encoding.  This is the simplest method, and functionality for one hot encoding is part of many standard libraries.  Zhou and Yu, however, present a new distance metric called `Minkovdm`.  Upon close examination, it can be found that Minkovdm is nothing more than Minkowsky distance applied to a feature set where the categorical features are encoded in a special manner.  Therefore, the second method of encoding that I explored is what I am calling "VDM encoding".  For more information about Minkovdm, refer to the papers by Zhou and Yu in the References section.

## Testing Ensemble vs. Baseline
A test is done to compare the performance of the ensemble algorithm under different configurations vs. baseline KNN.  A version of the ensemble algorithm is tested for every voting method that I have implemented.  Furthermore, each ensemble algorithm is tested both perturbing and not perturbing on the K values of the component classifiers.  These two parameters were varied while all others were kept the same because Zhou and Yu already performed extensive testing on the other ensemble parameters.

Each classifier is run on each dataset using 10 by 10-Fold Stratified Cross Validation.  The results of each of the 100 folds is placed in a list.  Once all tests have been performed, the classifier that has the lowest error for each dataset has its cell colored blue.  For all other classifiers for each dataset, a Pairwise Two-Tailed T-Test is performed to determine which classifier results are statistically worse than the blue cell.  Cells that are statistically worse are colored red, and those that aren't are colored green.  Blue and green cells are referred to as giving the `significantly best performance`.  At the bottom of the table is a total of the number of blue and green cells per classifier.  This process is repeated with the datasets encoded using one hot encoding and VDM encoding.  The results can be seen in `GeneralizationErrorTable_Onehot.png` and `GeneralizationErrorTable_VDM.png`.

Tests are then performed to calculate the error reduction rates on each of the aforementioned tables.  The error reduction rate is the percent decrease in error that a classifier has compared to the baseline KNN.  At the bottom of the table is an average of the error reduction rate for each classifier.  The results can be seen in `ErrorReductionRateTable_Onehot.png` and `ErrorReductionRateTable_VDM.png`.

![alt text](https://github.com/scoliann/KnnEnsemble/blob/master/GeneralizationErrorTable_Onehot.png)

## Interpreting the Results
Several important results can be gleaned from examining `GeneralizationErrorTable_Onehot.png` and `GeneralizationErrorTable_VDM.png`:
1.  For both one hot and VDM encoding, the fasbir_5 algorithm (ie. the FASBIR algorithm with constant K value for component classifiers, and the `bordaCount_allK_bordaOrder` voting method) yielded the most instances of the `significantly best performance`.  This means that fasbir_5 was the best (or statistically indistinguishable from the best) algorithm for the most number of data sets for both types of encoding.
2.  VDM encoding is superior to one hot encoding.  In the "Totals" row for both files, one can plainly see that every algorithm achieved an equal or greater total instances of the `significantly best performance` compared to one hot encoding.
3.  Perturbing on K values is often beneficial.  For both types of encoding, one can see that for each algorithm, perturbing on the K values led to a greater number of `significantly best performance`.  The only exception is the fasbir_5 algorithm.

Several important results can also be gleaned from examining `ErrorReductionRateTable_Onehot.png` and `ErrorReductionRateTable_VDM.png`:
1.  For both one hot and VDM encoding, the fasbir_5 algorithm resulted in the greatest error reduction.
2.  VDM encoding is superior to one hot encoding.  In the "Average" row, the error reduction for is greater for every algorithm when VDM encoding is used vs. one hot encoding.
3.  For both VDM and one hot encoding, perturbing on K values proved beneficial for implementations of fasbir_1, fasbir_2, and fasbir_3.  However, perturbing on the K values led to slight decreases in error reduction for implementations of fasbir_4, fasbir_5, and fasbir_6.

## Demonstration of Potential
Should anyone look at the results presented in the previous section and feel that using a KNN ensemble is a fancy way to achieve minor improvements, I would like to share the following.  In the previously mentioned tables, all hyper-parameters except the voting method and perturbing the K values were kept constant.  There is no guarantee that the constant values used were in any way optimal for achieving the best results.  Therefore, as a demonstration of the vast potential that using a KNN ensemble has over the baseline, I used Scikit-Learn's `RandomSearchCV` to tune a few of the hyper-parameters for the fasbir_5 algorithm.  The results can be seen in `GeneralizationErrorTable_RandomizedSearchCV_Onehot.png` and `ErrorReductionRateTable_RandomizedSearchCV_Onehot.png`.

A brief tuning of some of the hyper-parameters yielded a classifier that achieved `significantly best performance` on all twenty datasets (with seventeen of those twenty being the true best performance).  Furthermore, and more importantly, the error reduction of the fasbir_5 algorithm with tuned hyper-parameters was 23.1% vs. the 9.9% of the un-tuned fasbir_5.  That is a 233.3% reduction in error rate!

Most importantly, this massive improvement was achieved without tuning the hyper-parameter for ensemble size (which certainly one of the most important features), and was done with an `n_iter` value of 10 for `RandomSearchCV`.  This `n_iter` value is also on the low side, but was chosen to speed up run time.

## Conclusions
The results of this study lead me to believe that for some voting methods in my algorithm, perturbing on K values can offer significant improvement.  This especially seems to be the case with the first three voting methods, which are all forms of majority voting.

My results demonstrate that different voting methods can lead to significantly better results compared to simple majority voting.  From my experiments, `bordaCount_allK_bordaOrder` was the optimal voting method.  This method works by returning a Borda Count for each of the K nearest neighbors for each component classifier.  The Borda Counts of the component classifiers are then totaled to determine a classification prediction.

The results show that significant gains can be made by tuning the hyper-parameters.  In the future, any application of this algorithm for my own purposes will use `RandomSearchCV` (as it runs faster than `GridSearchCV`) to optimize hyper-parameters for the problem on which I am working.

## Thoughts
Overall, I am very happy with this experiment.  I learned a great deal about ensemble classifiers, their fundamentals, and I feel that I understand the KNN classifier (ironically one of the "simplest" classifiers) on a very high level now.  Furthermore, the results of this experiment were both consistent with past work, and I was able to improve upon known methods by incorporating voting methods and K perturbation.  Additionally, the results numerically speaking were extremely good, with my simple example demonstrating an average error reduction improvement of 233.3% over diverse datasets.  Finally, my results have led me to realize that VDM encoding is worth experimenting with, as it has yielded better results than one hot encoding for categorical features, rather handedly.

## FAQ
1.  In an ensemble, why must component classifiers must be accurate and "diverse" (ie. have uncorrelated errors)?
**Answer**:  The component classifiers must be accurate because inaccurate component classifiers will yield an inaccurate ensemble.  The component classifiers must be diverse so that the erroneous classifications of a data points by one classifier are "corrected" by the majority correct classifications of the remaining classifiers.

2.  Why can't bagging alone improve KNN in an ensemble?
**Answer**:  Imagine an ensemble of 100 KNN, and 100 training sets for those KNN that have been formed using bagging.  The probability of any data point appearing one or more times in one of the training sets is 62.3%.  Therefore, it is expected that the data point will exist in ~62 of the 100 training sets.  In a KNN ensemble with majority voting, the classification of a point will only change if one or more of the K nearest points of the baseline KNN is present as a K nearest point in fewer than 50% of the component classifiers.  As the size of an ensemble increases, the number of data points that occur in fewer than 50% of the datasets decreases monotonically.  As such, a KNN ensemble using bagging will be an extremely close in performance to that of the baseline KNN.

3.  If bagging cannot improve the performance of an ensemble for KNN, then why is it included as a diversity method?
**Answer**:  Though bagging alone cannot improve a KNN ensemble, bagging combined with other diversity methods can.  This was explored by Zhou and Yu, who showed that bagging and perturbing the distance metric yielded better results in an ensemble than either of those diversity methods did individually.  For more information, their papers are listed in the References section.

## Quick Note
This work is based off of numerous academic papers that I have read over the past year.  A special shout out to Zhi-Hua Zhou and Yang Yu of Nanjing University for implementing many of the diversity methods that previous researchers have explored in one algorithm, FASBIR.  Their paper can be found [here](https://www.researchgate.net/publication/3414055_Ensembling_Local_Learners_Through_Multimodal_Perturbation).

## References
1. Zhou, Z.-H., and Y. Yu. "Ensembling Local Learners Through Multimodal Perturbation." IEEE Transactions on Systems, Man and Cybernetics, Part B (Cybernetics) 35.4 (2005): 725-35. Web.
2. Zhou, Zhi-Hua, and Yang Yu. "Adapt Bagging to Nearest Neighbor Classifiers." Journal of Computer Science and Technology 20.1 (2005): 48-54. Web.
3. Garcia-Pedrajas, Nicolas, and Domingo Ortiz-Boyer. "Boosting K-nearest Neighbor Classifier by Means of Input Space Projection." Expert Systems with Applications 36.7 (2009): 10570-0582. Web.
4. Gul, Asma, Aris Perperoglou, Zardad Khan, Osama Mahmoud, Miftahuddin Miftahuddin, Werner Adler, and Berthold Lausen. "Ensemble of a Subset of KNN Classifiers." Advances in Data Analysis and Classification (2016): n. pag. Web.
5. Neo, Toh Koon Charlie, and Dan Ventura. "A Direct Boosting Algorithm for the K-nearest Neighbor Classifier via Local Warping of the Distance Metric." Pattern Recognition Letters 33.1 (2012): 92-102. Web.
6. Domeniconi, Carlotta, and Bojun Yan. "On Error Correlation and Accuracy of Nearest Neighbor Ensemble Classifiers." Proceedings of the 2005 SIAM International Conference on Data Mining (2005): 217-26. Web.

