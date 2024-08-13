import numpy as np
import matplotlib.pyplot as plt
'''This Function will return 2- 2D arrays the BLUE Labeled and the ORANGE labeled'''
def SAMPLE(NUM_OF_OBS_BLUE=100,NUM_OF_OBS_ORANGE=100,n1=10,n2=10,mu1=np.array([1,0]),mu2 = np.array([0,1]),cov1= np.array([[1, 0], [0, 1]]),cov2 = np.array([[1, 0], [0, 1]])):
    #generating the means from the first bivariate Gaussian distribution (BLUE)
    BLUE_MEAN = np.random.multivariate_normal(mu1, cov1, n1) 
    #generating the means from the second bivariate Gaussian distribution (ORANGE)
    ORANGE_MEAN = np.random.multivariate_normal(mu2, cov2, n2)
    #generating NUM_OF_OBS from BLUE & ORANGE
    BLUE_OBS = np.empty((NUM_OF_OBS_BLUE,2))
    ORANGE_OBS = np.empty((NUM_OF_OBS_ORANGE,2))
    for i in range(NUM_OF_OBS_BLUE):
        random_mu_blue = BLUE_MEAN[np.random.choice(n1)]
        
        
        CUR_OBS_BLUE = np.random.multivariate_normal(random_mu_blue, cov1/5)
        BLUE_OBS[i] = CUR_OBS_BLUE
    for i in range(NUM_OF_OBS_ORANGE):
        random_mu_orange = ORANGE_MEAN[np.random.choice(n2)]

        CUR_OBS_ORANGE = np.random.multivariate_normal(random_mu_orange, cov2/5)
        ORANGE_OBS[i] = CUR_OBS_ORANGE
    return BLUE_OBS, ORANGE_OBS