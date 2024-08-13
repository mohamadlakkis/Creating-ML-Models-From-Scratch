import numpy as np
''''''
'''It returns PREDICTION ARRAY (actual,predicted)'''
INF = float('inf')
def KNN_PRED(BLUE_TR,ORANGE_TR,BLUE_TEST,ORANGE_TEST,K):
    # 1 denoting BLUE, 
    # 0 denoting ORANGE
    #Storing the data
    n_test = len(BLUE_TEST)
    n_train = len(BLUE_TR)

    #Blue: i<n, Orange: i>=n

    ALL_TR = np.concatenate((BLUE_TR, ORANGE_TR)) 
    if(ORANGE_TEST is None):
        ALL_TEST = BLUE_TEST
    else:
        ALL_TEST =  np.concatenate((BLUE_TEST, ORANGE_TEST)) 

    PREDICTION = [[0, 0, 0] for _ in range(len(ALL_TEST))] #(actual value, predicted value,the observation)
    for i,(x1,x2) in enumerate(ALL_TEST):
        
        D = []
        for j,(xx1,xx2)in enumerate(ALL_TR):
            D.append((np.sqrt((x1-xx1)**(2)+(x2-xx2)**2),j)) # Distance from i to j ( we need to keep track of j since we are labeling them in the ALL array based on the index)
        #Doing the KNN Process
            
        count_blue = 0
        count_orange = 0
        D.sort()
        for k in range(K):

            if(D[k][1]<n_train):
                count_blue += 1
            else:
                count_orange += 1
        #Seeing the actual value of this data is what 
        if i<n_test:
            PREDICTION[i][0] = 1
        else:
            PREDICTION[i][0] = 0
        #Seeing the predicted value of this data is what
        if(count_blue>count_orange):
            PREDICTION[i][1] = 1
        elif(count_orange>count_blue):
            PREDICTION[i][1] = 0
            
        else: # we will label it based on a flip of a coin !
            PREDICTION[i][1] = np.random.choice([0,1])
        PREDICTION[i][2] = ALL_TEST[i]
    return PREDICTION