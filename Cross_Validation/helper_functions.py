import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
def helper_compute_errors_linear(lin_reg,X,Y):
    '''THis function is to eliminate outliers since maybe the round function will give us numbers other than 0 and 1, so this way we are choosing a threshold ourselves to guarantee the mapping of f_hat to {0,1}'''
    y_pred = lin_reg.predict(X)
    Y_PRED_ROUNDED = np.round(y_pred)
    y_pred = []
    for num in Y_PRED_ROUNDED:
        if num>=3:
            y_pred.append(3)
        else:
            y_pred.append(2)
    #Error is from a 0-1 loss function
    ERROR = 1-accuracy_score(Y, y_pred)
    return ERROR
def helper_custom_accuracy(y_true, y_pred):

    '''This function is used to set the accuracy manually since linear regression is not a classifier so we need to create our own method of classification, this function will be used in the computing the Cv_error'''
    y_pred_adjusted = [3 if y >= 3 else 2 for y in np.round(y_pred)]
    return accuracy_score(y_true, y_pred_adjusted)
def helper_compute_CV_error_linear(lin_reg, X_train, Y_train, number_of_k_folds):
    custom_scorer = make_scorer(helper_custom_accuracy)
    cv_scores = cross_val_score(lin_reg, X_train, Y_train, cv=number_of_k_folds, scoring=custom_scorer)
    cv_error = 1 - cv_scores.mean()
    return cv_error