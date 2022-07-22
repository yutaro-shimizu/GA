#import neural network pacakages
from sklearn.neural_network import MLPClassifier

import numpy as np
import random 
import pandas as pd

import sys
import warnings
warnings.filterwarnings('ignore')

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
        self.NNs = {} #  set of models for evolution. Swaps based on training score during evolution. 
        self.NNs_copy = {} # for reference during evolution. Does not change during swaps.
        self.all_train_score = []
        self.all_val_score = []
        self.neighbors = []

    def birth(self, population, hid_nodes, X_train, y_train):
        for ind in range(population):
            self.NNs[ind] = {"model":MLPClassifier(hidden_layer_sizes=(hid_nodes,), max_iter=1, alpha=1e-4,
                          solver='sgd', verbose=10, learning_rate_init=.1),
                        "train_score":0,
                        "val_score":0}
            self.NNs[ind]["model"].fit(X_train, y_train) # fit the network to initialize W and b
            # randomly initialize weights and biases
            self.NNs[ind]["model"].coefs_[0] = np.random.uniform(low=-1,high=1,size=(784,hid_nodes)) 
            self.NNs[ind]["model"].coefs_[1] = np.random.uniform(low=-1,high=1,size=(hid_nodes,10))

    def calculator(self, X_train, y_train, X_val, y_val):
        train_score = []
        val_score = []
        for ind in self.NNs:
            self.NNs[ind]["train_score"]= self.NNs[ind]["model"].score(X_train, y_train) # calculate the score
            self.NNs[ind]["val_score"]= self.NNs[ind]["model"].score(X_val, y_val)
            train_score.append(self.NNs[ind]["train_score"])
            val_score.append(self.NNs[ind]["val_score"])
        self.NNs_copy = self.NNs # copy original including the scores
        print("Max training score: ", np.amax(train_score))
        print("Max validation score: ", np.amax(val_score))
        self.all_train_score.append(np.amax(train_score))
        self.all_val_score.append(np.amax(val_score))

    def identify_max_neighbor(self,i,j,dim,neigh_size): #fix the modulus equation
        #take the score and compare
        score = self.NNs_copy[i*dim+j]["train_score"]
        idx = i*dim+j
        llim = -int(neigh_size/2)
        rlim = int(neigh_size/2)
        for k in range(llim,rlim+1):
            for l in range(llim,rlim+1):
                if score < self.NNs_copy[((i+k)%dim)*dim+(j+l)%dim]["train_score"]:
                    score = self.NNs_copy[((i+k)%dim)*dim+(j+l)%dim]["train_score"]
                    idx = ((i+k)%dim)*dim+(j+l)%dim
                    print("swapped")
        return idx

    def mutate(self, idx, mut_rate):
        for i in range(2):
            coef_size = self.NNs[idx]["model"].coefs_[i].size
            shape = self.NNs[idx]["model"].coefs_[i].shape
            mutate_idx = np.random.choice(coef_size,size = int(mut_rate*coef_size))
            for loci in mutate_idx:
                self.NNs[idx]["model"].coefs_[i].flat[loci] += np.random.normal(loc=0.05)
            self.NNs[idx]["model"].coefs_[i].reshape(shape)
        return None
    
    def replace_neighbors(self,dim,neigh_size,mut_rate=0.05):
        for i in range(dim):
            for j in range(dim):
                idx = i*dim+j
                self.NNs[idx] = self.NNs_copy[self.identify_max_neighbor(i,j,dim,neigh_size)]
                self.mutate(idx, mut_rate)
        self.NNs_copy = self.NNs

def run():

    ######### 1.Set Hyperparameters #########
    generations = 10 #int(sys.argv[1]) #10
    dimension = 4 #int(sys.argv[2]) #10
    population = dimension ** 2
    hid_nodes = 10 #int(sys.argv[3]) #10
    mut_rate = 0.05 #float(sys.argv[4]) #0.05
    neighbor_size = 3
    
    ######### 2.Load Data #########
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    ######### 3.Initialize Populatioin #########
    initializer = Initializer(hid_nodes)
    initializer.birth(population,hid_nodes,X_train,y_train)

    ######### 4.Run Spatial Evolution #########
    for i in range(generations):
        initializer.calculator(X_train, y_train, X_val, y_val)
        initializer.replace_neighbors(dimension,neighbor_size,mut_rate)
        print(initializer.NNs)

    return None

if __name__ == "__main__":
    run()