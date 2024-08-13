import matplotlib.pyplot as plt
import numpy as np
from knn_model import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sampling import *
from helper import *
from sklearn.linear_model import LinearRegression
def MY_error_0_1_Loss(PRED):
    count = 0
    n = len(PRED)
    for i,j,k in PRED:
        if i != j:
            count += 1
    return count/n
def Fig_2_2_from_MY_KNN(ALL_TR,BLUE_TRAIN,ORANGE_TRAIN,K=15,stepp=1,title="My KNN"):
    
    #Instead of sampling from our given distrbution now we will take small (x1,x2) values in order to make the boundary of points 
    a = np.arange(start=ALL_TR[:,0].min()-1,stop=ALL_TR[:,0].max()+1,step=stepp) 
    b = np.arange(start=ALL_TR[:, 1].min()-1, stop=ALL_TR[:, 1].max()+1, step=stepp)
    aa,bb = np.meshgrid(a,b)
    grid_points = np.column_stack((aa.ravel(), bb.ravel()))
    GRID_PRED = KNN_PRED(BLUE_TRAIN, ORANGE_TRAIN, grid_points,None, K)
    ORANGE = []
    BLUE = []
    for pred in GRID_PRED:
        if pred[1] == 0:
            ORANGE.append(pred[2])
        else:
            BLUE.append(pred[2])
    ORANGE = np.array(ORANGE)
    BLUE = np.array(BLUE)
    plt.figure(figsize=(10, 7))

    # Plot the boundary predicted points ( which will create the boundary which we will draw the curve in the contour function !)
    if ORANGE.size > 0:
        plt.scatter(ORANGE[:, 0], ORANGE[:, 1], color='orange',  s=2, marker='o')
    if BLUE.size > 0:
        plt.scatter(BLUE[:, 0], BLUE[:, 1], color='blue', s=2, marker='o')

    # Plot actual training points
    if ORANGE_TRAIN.size > 0:
        plt.scatter(ORANGE_TRAIN[:, 0], ORANGE_TRAIN[:, 1], color='orange', label='Training Orange', s=15, marker='x')
    if BLUE_TRAIN.size > 0:
        plt.scatter(BLUE_TRAIN[:, 0], BLUE_TRAIN[:, 1], color='blue', label='Training Blue', s=15, marker='x')
    prediction_grid = np.array([pred[1] for pred in GRID_PRED]).reshape(aa.shape)
    # Draw the decision boundary
    plt.contour(aa, bb, prediction_grid, levels=[0.5],colors='black', vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()
def Fig_from_sickit(ALL_TR, BLUE_TRAIN, ORANGE_TRAIN, knn,K=15, stepp=1,title="Sickit KNN"):

    a = np.arange(start=ALL_TR[:,0].min()-1, stop=ALL_TR[:,0].max()+1, step=stepp)
    b = np.arange(start=ALL_TR[:,1].min()-1, stop=ALL_TR[:,1].max()+1, step=stepp)
    aa, bb = np.meshgrid(a, b)
    grid_points = np.c_[aa.ravel(), bb.ravel()]
    #Predicting the entire points
    predictions = knn.predict(grid_points)
    prediction_grid = predictions.reshape(aa.shape)
    #Plotting the black cure (i.e. the boundary)
    plt.contour(aa, bb, prediction_grid, levels=[0.5], colors='black')
    # Add your training points to the plot
    plt.scatter(BLUE_TRAIN[:, 0], BLUE_TRAIN[:, 1], color='blue', label='Training Blue', s=15,marker='x')
    plt.scatter(ORANGE_TRAIN[:, 0], ORANGE_TRAIN[:, 1], color='orange', label='Training Orange', s=15,marker='x')
    #Adding the boundary points to the plot
    ORANGE = grid_points[predictions == 0]
    BLUE = grid_points[predictions == 1]
    if ORANGE.size > 0:
        plt.scatter(ORANGE[:, 0], ORANGE[:, 1], color='orange', s=2, marker='o')
    if BLUE.size > 0:
        plt.scatter(BLUE[:, 0], BLUE[:, 1], color='blue', s=2, marker='o')
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()
def Fig_2_4_from_sickit(X_train,Y_train,k_values,title="Sickit KNN",test_obs_number=10000):
    # I need to sample 10000 element randomly (to not always choose 5000 blue and 5000 orange)
    BLUE_NUMBER_OBS = np.random.randint(0, test_obs_number)
    ORANGE_NUMBER_OBS = test_obs_number - BLUE_NUMBER_OBS
    BLUE_TEST,ORANGE_TEST = SAMPLE(NUM_OF_OBS_BLUE=BLUE_NUMBER_OBS, NUM_OF_OBS_ORANGE=ORANGE_NUMBER_OBS)
    X_test = np.concatenate((BLUE_TEST, ORANGE_TEST))
    Y_test = np.concatenate((np.ones(len(BLUE_TEST)), np.zeros(len(ORANGE_TEST))))
    '''Linear Regression'''
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, Y_train)
    #Computing errors (Computing errors is trivial, the interesting accuracy work, will be in problem_3 when we are getting the K-FOld CV estimate for error [we will create our own accuracy criterion], but more on that later !)
    TEST_ERROR_LINEAR = helper_compute_errors_linear(lin_reg,X_test,Y_test)
    TRAIN_ERROR_LINEAR = helper_compute_errors_linear(lin_reg,X_train,Y_train)

    '''KNN'''
    
    # Calculate N/k which is the degrees of freedom for each k value
    N = len(Y_train) 
    dof = N / k_values
    TEST_ERROR = []
    TRAIN_ERROR = []
   
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, Y_train)
        knn_predictions_TRAIN = knn.predict(X_train)
        knn_predictions_TEST = knn.predict(X_test)
        #in the training and test error we are using a 0-1 loss, if there is match the loss is 0, if there is not match then loss is 1. We could have used other loss function but we will stick to these 
        train_error = 1 - accuracy_score(Y_train, knn_predictions_TRAIN)
        test_error = 1 - accuracy_score(Y_test, knn_predictions_TEST)
        TRAIN_ERROR.append(train_error)
        TEST_ERROR.append(test_error)
        
    
    
    fig, ax1 = plt.subplots()

    ax1.set_xscale('log')  
    ax1.set_xlabel('Degrees of Freedom - N/k')
    
    ax1.plot(dof, TRAIN_ERROR, 'o-', color='blue',label="TRAIN ERROR")
    ax1.plot(4,TRAIN_ERROR_LINEAR, 'o-', color='red', label="TRAIN ERROR (LINEAR REGRESSION)")
    ax1.plot(4, TEST_ERROR_LINEAR, 'x-', color='green', label = "TEST ERROR (LINEAR REGRESSION)")
    ax1.plot(dof, TEST_ERROR, 'o-', color='orange',label = "TEST ERROR")
    ax1.legend(loc = 'lower left')
    ax1.set_ylabel('Errors')
    ax2 = ax1.twiny()
    ax2.set_xscale('log') 
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(dof) 
    ax2.set_xticklabels(k_values) 
    ax2.set_xlabel('k - Number of Nearest Neighbors')
    ax1.set_xticks(dof)

    ax1.set_xticklabels([f'{N/k:0.2f}' for k in k_values])  
    plt.title(title)
    plt.show()