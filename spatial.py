#import neural network pacakages
from re import A
from scipy.misc import face
from scipy.spatial import distance
from sklearn.neural_network import MLPClassifier
from copy import deepcopy

import numpy as np
import random 
import pandas as pd
import matplotlib.pyplot as plt

import sys
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

class Spatial_GA:
    def __init__(self, hid_nodes):
        self.NNs = {}  # set of models for evolution. Swaps based on training score during evolution. 
        self.NNs_copy = {}  # for reference during evolution. Does not change during swaps.
        self.all_train_score = []
        self.all_val_score = []
        self.neighbors = []
        self.diversity = []

    def birth(self, population, hid_nodes, X_train, y_train):
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

    def calculator(self, X_train, y_train, X_val, y_val):
        """
        Calculate max score for each generation. Store max score in the array.
        """
        train_score = []
        val_score = []
        for ind in self.NNs:
            self.NNs[ind]["train_score"]= self.NNs[ind]["model"].score(X_train, y_train) # calculate the score
            self.NNs[ind]["val_score"]= self.NNs[ind]["model"].score(X_val, y_val)
            train_score.append(self.NNs[ind]["train_score"])
            val_score.append(self.NNs[ind]["val_score"])
        self.NNs_copy = deepcopy(self.NNs) # copy original including the scores
        print("Max training score: ", np.amax(train_score))
        print("Max validation score: ", np.amax(val_score))
        self.all_train_score.append(np.amax(train_score))
        self.all_val_score.append(np.amax(val_score))

        return val_score
    
    def plot_growth(self, val_score, dim, i):

        COLOUR = 'white'
        plt.rcParams['text.color'] = COLOUR
        plt.rcParams['axes.labelcolor'] = COLOUR
        plt.rcParams['xtick.color'] = COLOUR
        plt.rcParams['ytick.color'] = COLOUR

        plt.figure(facecolor="black")
        im = plt.imshow(np.reshape((val_score),(dim,dim)),vmin=0.1,vmax=0.8)
        plt.colorbar(im)
        plt.savefig(f'./Figures/figure{i}.png', transparent=True)

    def plot_final(self):

        COLOUR = 'white'
        plt.rcParams['text.color'] = COLOUR
        plt.rcParams['axes.labelcolor'] = COLOUR
        plt.rcParams['axes.edgecolor'] = COLOUR
        plt.rcParams['axes.facecolor'] = 'black'
        plt.rcParams['xtick.color'] = COLOUR
        plt.rcParams['ytick.color'] = COLOUR

        plt.figure(facecolor="black")
        plt.plot(self.all_train_score, label = "training")
        plt.plot(self.all_val_score, label = "validation")
        plt.xlabel("Generations")
        plt.ylabel("Max Accuracy")
        plt.legend()
        plt.savefig('./Figures/saptial_final.png', transparent=True)

        plt.figure(facecolor="black")
        plt.plot(self.diversity)
        plt.xlabel("Generations")
        plt.ylabel("Cosine Similarity")
        plt.savefig('./Figures/spatial_diversity.png', transparent=True)

    def identify_max_neighbor(self, i, j, dim, neigh_size): 
        score = self.NNs_copy[i * dim + j]["train_score"]
        idx = i * dim + j
        llim = - int(neigh_size / 2)
        rlim = int(neigh_size / 2)

        """
        ################### add probabilistic replacement###########
        """
        for k in range(llim,rlim+1):
            for l in range(llim,rlim+1):
                if score < self.NNs_copy[((i + k) % dim) * dim + (j + l) % dim]["train_score"]:
                    score = self.NNs_copy[((i + k) % dim) * dim + (j + l) % dim]["train_score"]
                    idx = ((i + k) % dim) * dim + (j + l) % dim
        return idx

    def mutate(self, idx, mut_rate=0.5):
        """
        In each layer, mutate weights at random cites with probability "mut_rate"
        ---
        idx: int

        """
        for i in range(2):
            coef_size = self.NNs[idx]["model"].coefs_[i].size
            shape = self.NNs[idx]["model"].coefs_[i].shape
            mutate_idx = np.random.choice(coef_size, size=int(mut_rate * coef_size))
            for loci in mutate_idx:
                self.NNs[idx]["model"].coefs_[i].flat[loci] += np.random.normal(loc=0.005)
            self.NNs[idx]["model"].coefs_[i].reshape(shape)
        return None
    
    def probe_neighbors(self,dim,neigh_size):
        """
        - Check every cell and apply probabilistic replacement and mutation
        - Mutation happens for each layer
        """
        for i in range(dim): # for every row
            for j in range(dim): #for every column
                idx = i*dim+j # convert row and column notations to an index 
                self.NNs[idx] = deepcopy(self.NNs_copy[self.identify_max_neighbor(i,j,dim,neigh_size)])
                self.mutate(idx)

    def cosine_sim(self):
        current_div = []
        for ind1 in self.NNs:
            for ind2 in self.NNs:
                if ind1 >= ind2:
                    continue
                dist = distance.cosine(np.concatenate((np.ravel([self.NNs[ind1]["model"].coefs_[0]]),
                                                    np.ravel([self.NNs[ind1]["model"].coefs_[1]]))), 
                                    np.concatenate((np.ravel([self.NNs[ind2]["model"].coefs_[0]]), 
                                                    np.ravel([self.NNs[ind2]["model"].coefs_[1]]))))
                current_div.append(dist)
        div_score = np.mean(current_div)
        self.diversity.append(div_score)
        print("Cosine Similarity: ", div_score,"\n")

def run():

    ######### 1.Set Hyperparameters #########
    generations = int(sys.argv[1]) #10
    dimension = int(sys.argv[2]) #10
    div_switch = int(sys.argv[3])
    population = dimension ** 2
    hid_nodes = 10 #int(sys.argv[3]) #10
    mut_rate = 0.5 #float(sys.argv[4]) #0.05
    neighbor_size = 3

    print("Total arguments: ", len(sys.argv))
    print("generations: ", sys.argv[1])
    print("population: ", sys.argv[2])
        
    ######### 2.Load Data #########
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    ######### 3.Initialize Populatioin #########
    spaceGA = Spatial_GA(hid_nodes)
    spaceGA.birth(population,hid_nodes,X_train,y_train)

    ######### 4.Run Spatial Evolution #########
    for i in range(generations):
        print("\ncurrent generation: ", i)
        val_score = spaceGA.calculator(X_train, y_train, X_val, y_val)
        ######### Plot growth per 10 iterations #########
        if i%10 == 0:
            spaceGA.plot_growth(val_score, dimension, i)
        spaceGA.probe_neighbors(dimension, neighbor_size)
        if div_switch:
            spaceGA.cosine_sim()

    ######### 5.Plot Result #########
    spaceGA.plot_final()
    return None

if __name__ == "__main__":
    run()