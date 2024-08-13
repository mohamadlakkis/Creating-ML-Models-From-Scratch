from Data_imports import *
from sklearn.neighbors import KNeighborsClassifier
from tools import *
path_label_2_TR = r"Problem_3\Data\label_2_train.txt"
path_label_3_TR = r"Problem_3\Data\label_3_train.txt"
path_test_data  = r"Problem_3\Data\test.txt"
X_train, Y_train = import_training_data(path_label_2_TR,path_label_3_TR)
X_test,Y_test = import_test_data(path_test_data)

'''
This Exercise is answering Question 2.8 From the book "The Elements of Statistical Learning Second Edition". In addition to that, I am building the CV Model from Scratch and comparing it to the one from Sickit Learn !
'''
#-------------KNN--------------
TEST_ERRORS_KNN = []
TRAIN_ERRORS_KNN = []
CV_SICKIT_ERRORS_KNN = []
MY_CV_ERRORS_KNN = []
number_of_k_folds = 10
K_values = [i for i in range(1,16)]
for k in K_values:    
    knn = KNeighborsClassifier(n_neighbors=k)
    train_error,test_error = compute_KNN_ERRORS(X_train, Y_train, X_test, Y_test,knn)
    cv_sickit_error = compute_CV_sickit_error_KNN(X_train, Y_train,knn,number_of_k_folds=number_of_k_folds)
    my_cv_error = compute_my_cv_error_KNN(X_train, Y_train, k, number_of_k_folds=number_of_k_folds)
    MY_CV_ERRORS_KNN.append(my_cv_error)
    CV_SICKIT_ERRORS_KNN.append(cv_sickit_error)
    TEST_ERRORS_KNN.append(test_error)
    TRAIN_ERRORS_KNN.append(train_error)  
plotting_KNN_ERRORS(K_values,TRAIN_ERRORS_KNN,TEST_ERRORS_KNN,CV_SICKIT_ERRORS_KNN,MY_CV_ERRORS_KNN)
'''
We are interested in the K where the test_error is the smallest, but sometimes we don't have access to a test_error from our test_data and it is best to approximate it using the K-fold CV, which we will care about the minimum also in order to conclude based on this training data what is the best K to use !
the test error initially decreases(OR WE CAN SAY STAY STABLE) as k increases from 1, suggesting that the model's ability to generalize is improving. However, after a certain point, the test error starts to increase slightly, which might indicate the beginning of underfitting. This turning point is where the model may have the best trade-off between bias and variance. As we will discuss in number 4 (see pdf)!
The cross-validation error seems to follow a trend similar to the test error but at a lower error rate. This suggests that the
cross-validation is a good estimator for the test error and could be used to select the optimal value of k (which corresponds to the minimum value).
We can also see that the CV error is generally stable, which is due to the good choice of K=10 folds if we choose K=n this might lead to unstability in this estimator
'''
#-------------Linear Regression--------------
'''
Notice that here we won't get a graph we will get only one point for each error, because we have only one prediction , Please note that here I didn't compute the CV error using my model since it is not efficient it took 1 hour and something to run so I played a little bit with a built in model and made the built in model works for our case which is using linear regression for classification, so I built from scratch the classifier as you can see in more details in the function linear_Errors_computation
'''
TRAIN_ERROR,TEST_ERROR,CV_ERROR = linear_Errors_computation(X_train,Y_train,X_test,Y_test,number_of_k_folds=10)
print(f"TRAIN ERROR Linear : {TRAIN_ERROR}")
print(f"TEST ERROR Linear : {TEST_ERROR}")
print(f"CV ERROR Linear : {CV_ERROR}")
'''
And here we see that, CV error is the closest approximation to the test error. But we can't really say much here since we don't have many points to compare maybe the complexity or flexibility we can compare this model with the K-models and see which one is better -> See conclusion !
'''

#-------------Conclusion--------------
'''We can see that in this case the use of KNN with small K is better than the use of linear regression because the error rate of both the CV and test error is smaller than the one in linear regression
Notice that we know from the Gauss-Markov Theorem, we know that when (X,y) is multivariate Normal with unknown parameters, the
theorem holds- in this case the OLS is infact better than any other unbiased estimator, not necessarily linear; but KNN does not fall under the class of estimators for which we would typically define unbiasedness in the same way as for linear models. Its properties cannot be directly compared with those of OLS estimators in terms of unbiasedness because they are different types of models (non-parametric vs. parametic) ! In addition notice that regarding the mixture of Gaussian distributionsthis distribution is different from the simple multivariate normal distribution discussed in notes_2 the Gauss-Markov theorem. The mixture model involves a more complex data generation process, which does not conform to the assumptions required for the theorem to hold(linearity). TO SUMMARIZE, even if we assume that KNN is in the same unbiasedness that is meant in the Gauss-markov theorem, the theorem may not hold in this case (i.e. blue may not apply!)-> which is confirmed by the results of the graph !
'''
