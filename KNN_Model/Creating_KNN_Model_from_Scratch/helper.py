import numpy as np
from sklearn.metrics import accuracy_score
def helper_compute_errors_linear(lin_reg,X,Y):
    '''THis function is to eliminate outliers since maybe the round function will give us numbers other than 0 and 1, so this way we are choosing a threshold ourselves to guarantee the mapping of f_hat to {0,1}'''
    y_pred = lin_reg.predict(X)
    Y_PRED_ROUNDED = np.round(y_pred)
    y_pred = []
    for num in Y_PRED_ROUNDED:
        if num>=1:
            y_pred.append(1)
        else:
            y_pred.append(0)
    #Error is from a 0-1 loss function
    ERROR = 1-accuracy_score(Y, y_pred)
    return ERROR