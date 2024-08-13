from sampling import *
from knn_model import *
from tools import *
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
'''The images can be found in the  "The Elements of
 Statistical Learning Second Edition Book" Chapter 2'''
np.random.seed(0)    
BLUE_TRAIN,ORANGE_TRAIN = SAMPLE(NUM_OF_OBS_BLUE=100,NUM_OF_OBS_ORANGE=100)
K = 15

'''FIG 2.2 from my KNN'''
ALL_TR = np.concatenate((BLUE_TRAIN, ORANGE_TRAIN))
Fig_2_2_from_MY_KNN(ALL_TR,BLUE_TRAIN,ORANGE_TRAIN,K,stepp=0.1,title="Fig_2_2 from MY KNN")


'''FIG 2.2 from Sickit KNN'''
X_train = np.concatenate((BLUE_TRAIN, ORANGE_TRAIN))
Y_train = np.concatenate((np.ones(len(BLUE_TRAIN)), np.zeros(len(ORANGE_TRAIN))))
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, Y_train)
knn_predictions = knn.predict(X_train)

Fig_from_sickit(ALL_TR, BLUE_TRAIN, ORANGE_TRAIN, knn,K=K, stepp=0.1,title="Fig_2_2 from Skickit")

#We can see that both the two graphs are the same, which means that our model that we created is the same

'''FIGURE 2.3 from Sickit KNN'''
K=1
X_train = np.concatenate((BLUE_TRAIN, ORANGE_TRAIN))
Y_train = np.concatenate((np.ones(len(BLUE_TRAIN)), np.zeros(len(ORANGE_TRAIN))))
knn = KNeighborsClassifier(n_neighbors=K)
knn.fit(X_train, Y_train)
knn_predictions = knn.predict(X_train)

Fig_from_sickit(ALL_TR, BLUE_TRAIN, ORANGE_TRAIN, knn,K=K, stepp=0.1,title="Fig_2_3 from Sickit")
# We clearly see that in this case overfitting will take place, high variance, low bias

'''FIGURE 2.4 from Sickit KNN'''
X_train = np.concatenate((BLUE_TRAIN, ORANGE_TRAIN))
Y_train = np.concatenate((np.ones(len(BLUE_TRAIN)), np.zeros(len(ORANGE_TRAIN))))
K = np.array([1, 3, 5, 7, 11, 21, 31, 45, 69, 101, 151,200])
Fig_2_4_from_sickit(X_train,Y_train,k_values=K,title="Fig_2_4 from Sickit")