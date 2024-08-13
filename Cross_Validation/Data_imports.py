import numpy as np
def import_training_data(path_label_2_TR,path_label_3_TR):
    file = open(path_label_2_TR, 'r')
    X_label_2_train = []
    for line in file:
        line = line.split(",")
        K = []
        for word in line:
            if word!="":
                if "\n"in word:
                    word  = word[0:-1]
                
                K.append(float(word))
        X_label_2_train.append(K)
    file = open(path_label_3_TR, 'r')
    X_label_3_train = []
    for line in file:
        line = line.split(",")
        K = []
        for word in line:
            if word!="":
                if "\n"in word:
                    word  = word[0:-1]
                
                K.append(float(word))
        X_label_3_train.append(K)
    X_label_2_train = np.array(X_label_2_train)
    X_label_3_train = np.array(X_label_3_train)
    X_Train = np.concatenate((X_label_2_train, X_label_3_train),axis=0)
    Y_label_2 = np.full((len(X_label_2_train),), 2)  
    Y_label_3 = np.full((len(X_label_3_train),), 3) 
    Y_train = np.concatenate((Y_label_2, Y_label_3),axis=0)
    return X_Train,Y_train
def import_test_data(path_test_data):
    file = open(path_test_data, 'r')
    Y_test = []
    X_test = []
    for line in file:
        if line[0]=="2" or line[0]=="3":
            Y_test.append(int(line[0]))
            CURR = line.split(" ")
            for i in range(1,len(CURR)):
                CURR[i] = CURR[i].split("\n")
                CURR[i] = float(CURR[i][0])
            CURR = CURR[1:]
            X_test.append(CURR)
    return np.array(X_test),np.array(Y_test)