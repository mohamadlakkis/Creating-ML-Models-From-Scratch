from Data_imports import *
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from helper_functions import *
def compute_KNN_ERRORS(X_train,Y_train,X_test,Y_test,knn):
    knn.fit(X_train, Y_train)
    knn_predictions_TRAIN = knn.predict(X_train)
    knn_predictions_TEST = knn.predict(X_test)
    #Here we will also use a 0-1 loss function to compute the error rates !
    train_error = 1 - accuracy_score(Y_train, knn_predictions_TRAIN)
    test_error = 1 - accuracy_score(Y_test, knn_predictions_TEST)
    return train_error,test_error
def compute_CV_sickit_error_KNN(X_train, Y_train,knn,number_of_k_folds=10):
    #notice that the accuracy scores here gives us an array of accuracy values for each subset of size n/kfold, and then we will average them by using the .mean() function 
    #Here we will also use a 0-1 loss function to compute the error rates ! (which is indicated by the parameter scoring = 'accuracy')
    accuracy_scores = cross_val_score(knn, X_train, Y_train, cv=number_of_k_folds, scoring='accuracy')
    error_rates = 1 - accuracy_scores
    return error_rates.mean()
def compute_my_cv_error_KNN(X_train, Y_train, k, number_of_k_folds=10):
    knn = KNeighborsClassifier(n_neighbors=k)
    #dividing the array into validation sets of size n/num_of_folds
    ERROR = []
    n  = len(X_train)
    size = n//number_of_k_folds
    
    i = 0
    while i<n:
        start = i
        end = i+size
        if(i+size>=n):
            end = n-1
        i = i+size
        X_val = X_train[start:end]
        Y_val = Y_train[start:end]
        X_train_new = np.concatenate((X_train[:start],X_train[end:]))
        Y_train_new = np.concatenate((Y_train[:start], Y_train[end:]))
        #Same thing here test_error is from a 0-1 loss function 
        train_error, test_error = compute_KNN_ERRORS(X_train_new, Y_train_new, X_val, Y_val, knn)
        #Notice that instead of using the compute_KNN_ERRORS function, we can use our own function which is trivial to create we just check if y_predicted[i]!=Y_val[i], and then we get the average 
        ERROR.append(test_error)
    ERROR = np.array(ERROR)
    return ERROR.mean()
def plotting_KNN_ERRORS(K_values,TRAIN_ERRORS,TEST_ERRORS,CV_SICKIT_ERRORS,MY_CV_ERRORS):
    plt.figure(figsize=(10, 5))
    plt.plot(K_values, TRAIN_ERRORS, marker='o', linestyle='-', color='blue', label='Train Error')
    plt.plot(K_values, TEST_ERRORS, marker='s', linestyle='--', color='red', label='Test Error')
    plt.plot(K_values, CV_SICKIT_ERRORS, marker='*', linestyle='-.', color='green', label='CV Error')
    plt.plot(K_values, MY_CV_ERRORS, marker='^', linestyle=':', color='orange', label='My CV Error')
    plt.title('KNN Train and Test Errors for Different K Values')
    plt.xlabel('K Value')
    plt.ylabel('Error Rate')
    plt.xticks(K_values)  
    plt.legend()  
    plt.grid(True)  
    plt.show()
def linear_Errors_computation(X_train,Y_train,X_test,Y_test,number_of_k_folds):
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, Y_train)
    TEST_ERROR = helper_compute_errors_linear(lin_reg,X_test,Y_test)
    TRAIN_ERROR = helper_compute_errors_linear(lin_reg,X_train,Y_train)
    CV_ERROR = helper_compute_CV_error_linear(lin_reg, X_train, Y_train, number_of_k_folds)
    
    return TRAIN_ERROR,TEST_ERROR,CV_ERROR