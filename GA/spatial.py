#import neural network pacakages
from sklearn.neural_network import MLPClassifier

import numpy as np
import random 
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

def params():
    # input hyperparameters from the shell script
    generations = 10 #int(sys.argv[1]) #10
    dimension = 10 #int(sys.argv[2]) #10
    population = dimension ** 2
    hid_nodes = 10 #int(sys.argv[3]) #10
    selection_percent = 0.2 #int(sys.argv[4]) #20
    mut_rate = 0.5 #float(sys.argv[5]) #0.05
    print("Total arguments: ", len(sys.argv))
    print("generations: ", sys.argv[1])
    print("population: ", sys.argv[2])
    # print("hid_nodes: ", sys.argv[3])
    # print("select_percent: ", sys.argv[4])
    # print("mut_rate: ", sys.argv[5])

def load_data(train_csv='mnist_train.csv',test_csv='mnist_test.csv'):
    # load data
    print("load data")

    data_train = pd.read_csv(train_csv) #load MNIST training data in
    data_train = np.array(data_train) #turn into array
    m, n =data_train.shape
    np.random.shuffle(data_train)
    Y_train=data_train[:,0]
    X_train=data_train[:,1:n]

    data_test = pd.read_csv(test_csv) #validating data loaded in
    data_test = np.array(data_test) #turned to array and transposed
    p, q = data_test.shape
    np.random.shuffle(data_test)

    y_test =data_test[:,0] #first row of data
    X_test = data_test[:,1:q] #rest of data

    #next two lines are taking 10,000 samples from MNIST
    X_train, X_val = X_train[:10000], X_train[10000:20000]
    y_train, y_val = Y_train[:10000], Y_train[10000:20000]

    print("load data complete")
    return X_train, X_val, X_test, y_train, y_val, y_test

class Initializer:
    def __init__(self, hid_nodes):
        self.model = MLPClassifier(hidden_layer_sizes=(hid_nodes,), max_iter=1, alpha=1e-4,
                          solver='sgd', verbose=10, learning_rate_init=.1)
        self.train_score = 0
        self.val_score = 0
        self.neighbors = []

    def birth(self, population, hid_nodes, X_train, y_train):
        self.NNs = {}
        self.NNs_copy = {}
        for ind in population:
            for ind in range(population):
                self.NNs[ind] = {"model":self.model,
                            "train_score":self.train_score,
                            "val_score":self.val_score}
                self.NNs[ind]["model"].fit(X_train, y_train) # fit the network to initialize W and b
                # randomly initialize weights and biases
                self.NNs[ind]["model"].coefs_[0] = np.random.uniform(low=-1,high=1,size=(784,hid_nodes)) 
                self.NNs[ind]["model"].coefs_[1] = np.random.uniform(low=-1,high=1,size=(hid_nodes,10))

    def calculator(self):
        train_score = []
        val_score = []
        for ind in self.NNs:
            self.NNs[ind]["train_score"]= self.NNs[ind]["model"].score(X_train, y_train) # calculate the score
            self.NNs[ind]["val_score"]= self.NNs[ind]["model"].score(X_val, y_val)
            train_score.append(self.NNs[ind]["train_score"])
            val_score.append(self.NNs[ind]["val_score"])
        print("Max training score: ", np.amax(train_score))
        print("Max validation score: ", np.amax(val_score))
        self.train_score.append(np.amax(train_score))
        self.val_score.append(np.amax(val_score))

    def identify_neighbors(self, dimension,i,j):
        #take the score and compare
        score = self.NNs[i+4*j]["model"].score(X_train, y_train) 
        idx = (i+4*j)
        for k in range(-1,2):
            if score < self.NNs[(i+k)+4*(j+k)]["model"].score(X_train, y_train):
                score = self.NNs[(i+k)+4*(j+k)]["model"].score(X_train, y_train)
                idx = ((i+k)+4*(j+k))
        return idx

    def replace_neighbors(self,dimension):
        for i in range(dimension):
            for j in range(dimension):
                self.NNs[i+4*j] = self.NNs[self.identify_neighbors(dimension, i, j)]

    def mutation(self, idx, mut_rate):
        for i in range(2):
            self.NNs[idx]["model"].coefs_[i]
        # chose mutation sites
        # mutate
        mutate_idx = np.random.choice(child_coefs.size,size = int(mut_rate*prt1.size))
        for idx in mutate_idx:
            child_coefs[idx] += np.random.normal(loc=0.1)
            return None

def run():
    return None