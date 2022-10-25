import numpy as np
import random
import sys

#import neural network pacakages
from sklearn.neural_network import MLPClassifier
from scipy.spatial import distance
from copy import deepcopy

import pandas as pd
from matplotlib import pyplot as plt #package for visualization
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

def load_data(train_csv='mnist_train.csv',test_csv='mnist_test.csv'):
    # load data
    print("load data")

    data_train = pd.read_csv(train_csv) # load MNIST training data in
    data_train = np.array(data_train)   # turn into array
    m, n = data_train.shape
    np.random.shuffle(data_train)
    Y_train = data_train[:, 0]
    X_train = data_train[:, 1:n]

    data_test = pd.read_csv(test_csv)   # validating data loaded in
    data_test = np.array(data_test)     # turned to array and transposed
    p, q = data_test.shape
    np.random.shuffle(data_test)

    y_test = data_test[:, 0]    # first row of data
    X_test = data_test[:, 1:q]  # rest of data

    #next two lines are taking 10,000 samples from MNIST
    X_train, X_val = X_train[:10000], X_train[10000:20000]
    y_train, y_val = Y_train[:10000], Y_train[10000:20000]

    print("load data complete")
    return X_train, X_val, X_test, y_train, y_val, y_test

class NonSpatial_Coev_GA:
    def __init__(self, hid_nodes):
        self.NNs = {} # set of models for evolution. Swaps based on training score during evolution. 
        self.NNs_copy = {}  # for reference during evolution. Does not change during swaps.
        self.MNIST = {} # track parasite score
        self.all_train_score = []
        self.all_val_score = []
        self.neighbors = []
        self.cos_sim = []
        self.entropy = []

############## 1. Initialize parasite (load data) ##############
    def birth_host(self, population, hid_nodes, X_train, y_train):
        """
        Produce population each individual containing model, training score and validation score attributes.
        """
        for ind in range(population):
            self.NNs[ind] = {"model": MLPClassifier(hidden_layer_sizes=(hid_nodes,), max_iter=1, alpha=1e-4,
                          solver='sgd', verbose=False, learning_rate_init=.1),
                        "train_score": 0,
                        "val_score": 0}
            self.NNs[ind]["model"].fit(X_train, y_train) # fit the network to initialize W and b
            # randomly initialize weights and biases
            self.NNs[ind]["model"].coefs_[0] = np.random.uniform(low=-1, high=1, size=(784, hid_nodes)) 
            self.NNs[ind]["model"].coefs_[1] = np.random.uniform(low=-1, high=1, size=(hid_nodes, 10))
    
    def birth_parasite(self, population, X_train, y_train):
        """
        from a list of MNIST digits, take out samples
        """
        return None

############## 3. Initialize host ##############
#sklearn models

############## 4.Run Non-spatial coevolution ##############
for i in range(generations):
    # 4.1 calculate fitness for host
    # 4.2 calculate fitness for parasite 
        # how do i calculate the score for parasites?
    # 4.2 select the best performing two individuals
    # 4.3 select samples
    # 4.4 breed and mutate host
    # 4.5 mutate samples (parasite)

for i in range(generations):

    model.calculator(X_train, y_train, X_val, y_val)
    model.selection()
    model.evolution(population, cv_switch, selection_percent)
    model.cosine_sim()
