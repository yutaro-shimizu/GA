#import neural network pacakages
from re import A
from scipy.misc import face
from scipy.spatial import distance
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from scipy.special import kl_div
from scipy.stats import entropy
from copy import deepcopy

import numpy as np
import random 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import sys
import warnings
warnings.filterwarnings('ignore')

def load_data(train_csv='mnist_train.csv',test_csv='mnist_test.csv'):
    """
    Helper function to load MNIST handwritten digits (Deng, 2012). This function has two purposes:
    1. load the data
    2. Divide the datasaet into two parts: training and test dataset.

    Training dataset is useed to trian the model and test dataset is used to check the accuracy.
    The dataset comes with images and correct labels of the digits.
    
    Deng, L., 2012. The mnist database of handwritten digit images for machine learning research. IEEE Signal Processing Magazine, 29(6), pp. 141–142.
    """

    # load data
    print("\nload data")

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
    X_train, X_val = X_train[:50000], X_train[50111:50223]
    y_train, y_val = Y_train[:50000], Y_train[50111:50223]

    print("load data complete")
    return X_train, X_val, X_test, y_train, y_val, y_test

class Spatial_GA:
    def __init__(self, hid_nodes):
        """
        Initialize key attributes for the model:
        NNs: host algorithm, content is initialized in the birth method.
        NNS_copy: copy of the host. Referenced during selection, and mutation for repopulation purpose.  
        all_train_score: keeps global max train accuracy
        all_val_score: keeps global max validation accuracy
        all_parasite_score: keeps global MNIST score
        cos_sim: keeps global genotype score
        entropy: keeps global phenotype score
        """

        self.NNs = {}  # set of models for evolution. Swaps based on training score during evolution. 
        self.NNs_copy = {}  # for reference during evolution. Does not change during swaps.
        self.all_train_score = []
        self.all_val_score = []
        self.neighbors = []
        self.cos_sim = []
        self.entropy = []

    ############## 1. Initialize host algorithm ##############
    def birth(self, population, hid_nodes, X_train, y_train):
        """
        Produce population each individual containing model, training score and validation score attributes.
        Line 94 and 95 represent the genome of neural networks as genotype

        """
        for ind in range(population):
            self.NNs[ind] = {"model": MLPClassifier(hidden_layer_sizes=(hid_nodes,), max_iter=1, alpha=1e-4,
                          solver='sgd', verbose=False, learning_rate_init=.1),
                        "train_score": 0,
                        "val_score": 0,
                        "parasite_X_train": None,
                        "parasite_y_train": None,
                        "parasite_score": 0}                        
            self.NNs[ind]["model"].fit(X_train, y_train) # fit the network to initialize W and b
            # randomly initialize weights and biases
            self.NNs[ind]["model"].coefs_[0] = np.random.uniform(low=-1, high=1, size=(784, hid_nodes)) 
            self.NNs[ind]["model"].coefs_[1] = np.random.uniform(low=-1, high=1, size=(hid_nodes, 10))

            ### 1.2 populate parasite (MNIST handwritten digits) ###
            indices = []
            counter = 0
            for num in np.unique(y_train, return_counts = True)[1]:
                # for each grid, load 110 digit
                indices.extend(np.random.randint(counter, counter + num, 11))
                counter += num
            self.NNs[ind]["parasite_X_train"] = X_train[indices]
            self.NNs[ind]["parasite_y_train"] = y_train[indices]

    ############## 2.Run Non-spatial coevolution ##############
    # 2.1 calculate fitness for host
    def fitness(self, X_train, y_train, X_val, y_val):
        """
        Calculate max score for the host and paraste in each generation.
        Also output confusion matrix of host for phenotype measure.

        Fitness for host NN algorithm (line 113 and line 114): correct classification percentage
        """
        train_score = []
        val_score = []
        cf_matrix = []

        for ind in self.NNs:
            self.NNs[ind]["train_score"]= self.NNs[ind]["model"].score(self.NNs[ind]["parasite_X_train"], self.NNs[ind]["parasite_y_train"]) # calculate the score
            self.NNs[ind]["val_score"]= self.NNs[ind]["model"].score(X_val, y_val)
            train_score.append(self.NNs[ind]["train_score"])
            val_score.append(self.NNs[ind]["val_score"])

            ## output confusion matrix and compute relative entropy
            y_val_pred = self.NNs[ind]["model"].predict(X_val)
            cf_matrix.append(confusion_matrix(y_val,y_val_pred))

        self.NNs_copy = deepcopy(self.NNs) # copy original including the scores
        print("Max training score: ", np.amax(train_score))
        print("Max validation score: ", np.amax(val_score))
        self.all_train_score.append(np.amax(train_score))
        self.all_val_score.append(np.amax(val_score))

        return val_score, cf_matrix
    
    def identify_max_neighbor(self, i, j, dim, neigh_size, rou_switch): 
        """
        Select top perfoming individuals.

        This is a probabilistic replacement where the top performing individuals are proportionately selected.
        The elitist strategy is from Mitchell (2006), howerver recent suggestions consider diveristy Mouret, J. B. (2020). 
        """
        score = self.NNs_copy[i * dim + j]["train_score"]
        idx = i * dim + j
        llim = - int(neigh_size / 2)
        rlim = int(neigh_size / 2)

        indices_lst = []
        sum_fit = 0
        fit_lst = []
        
        for k in range(llim,rlim+1):
            for l in range(llim,rlim+1):
                ### roulette wheel selection ###
                if rou_switch:
                    fit_lst.append(self.NNs_copy[((i + k) % dim) * dim + (j + l) % dim]["train_score"])
                    indices_lst.append(((i + k) % dim) * dim + (j + l) % dim)
                ### deterministic replacement###
                else:
                    if score < self.NNs_copy[((i + k) % dim) * dim + (j + l) % dim]["train_score"]:
                        score = self.NNs_copy[((i + k) % dim) * dim + (j + l) % dim]["train_score"]
                        idx = ((i + k) % dim) * dim + (j + l) % dim
        ### roulette wheel selection ###
        if rou_switch:
            sum_fit = sum(fit_lst)
            idx = np.random.choice(indices_lst, 1, p = fit_lst / sum_fit)[0] # returns index as nd array
        return idx

    def mutate(self, idx, host_mut_rate=0.5, host_mut_amount=0.005):
        """
        In each layer, mutate weights at random cites with probability "mut_rate",  Mitchell (2006).
        ---
        idx: int

        line 171 -line 173: select mutation location
        line 174 - line 176: mutate mut_amount sampled from a normal distribution
        """
        for i in range(2):
            coef_size = self.NNs[idx]["model"].coefs_[i].size
            shape = self.NNs[idx]["model"].coefs_[i].shape
            mutate_idx = np.random.choice(coef_size, size=int(host_mut_rate * coef_size))
            for loci in mutate_idx:
                self.NNs[idx]["model"].coefs_[i].flat[loci] += np.random.normal(loc=host_mut_amount)
            self.NNs[idx]["model"].coefs_[i].reshape(shape)
        return None
    
    def probe_neighbors(self, dim, neigh_size, rou_switch, host_mut_rate, host_mut_amount):
        """
        - Check every cell and apply probabilistic replacement and mutation
        - Mutation happens for each layer
        """
        for i in range(dim): # for every row
            for j in range(dim): #for every column
                idx = i*dim+j # convert row and column notations to an index 
                self.NNs[idx]["model"] = deepcopy(self.NNs_copy[self.identify_max_neighbor(i, j, dim, neigh_size, rou_switch)]["model"])
                self.mutate(idx, host_mut_rate, host_mut_amount)
   
    ############## 3.Measure phenotype, genotype and store result ##############
    def entropy_calculator(self, cf_matrix):
        """
        Compute KL-divergence (distance between metrices) to characterize phenotype
        normalize each row of the confusion matrix for KL-Divergence calculation
        """
        entropy_periter = []

        for i in range(len(cf_matrix)):
            for j in range(len(cf_matrix) - 1):
                for row in range(10):
                    # add 1e-4 to avoid overflow (division by 0 for log)
                    entropy_periter.append(kl_div(normalize(1e-4+cf_matrix[i], axis=1, norm='l1')[row],
                                                normalize(1e-4+cf_matrix[j+1], axis=1, norm='l1')[row]))


        mean_KL = np.average(entropy_periter)
        self.entropy.append(mean_KL)
        print("KL-Divergence: ", mean_KL)

    def cosine_sim(self):
        """
        Vectorize weights of each host NN into a single genome and meausre the angular difference to capture genotype diveristy.
        """
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
        self.cos_sim.append(div_score)
        print("Cosine Similarity: ", div_score,"\n")

    def store_result(self, hyp_params):
        now = datetime.now().strftime("%y%m%d%H%M%S")

        d = {"train_score":self.all_train_score,
        "val_score":self.all_val_score,
        "cos_sim":self.cos_sim,
        "rel_ent":self.entropy,
        "hyp_params":hyp_params}
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))

        df.to_csv(f"./results/result_spatial{now}.csv")
        print(f"result stored as: result_spatial{now}.csv")

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

def run():

    ######### 1.Set Hyperparameters #########
    """
    Take in all hyperparameters to initiqalize the genetic population. Additional parasite hyperparameters.
    Major hyperparameters are from Mitchell (2006). Reference CP193 submission for references.
    """
    generations = int(input("Enter generations: ")) #10
    dimension = int(input("Enter dimension: ")) #10
    rou_switch = bool(input("Enter roulette_switch (True/False): "))
    population = dimension ** 2
    hid_nodes = 10 #int(sys.argv[4]) #10
    neighbor_size = 3
    host_mut_rate = float(input("FOR HOST Enter mutation RATE (default 0.5): "))
    host_mut_amount = float(input("FOR HOST Enter mutation AMOUNT (default 0.005): "))
    print("\nGenerations: ", generations)
    print("Population: ", population)
    print("Roulette Selection: ", rou_switch)
    print("Host mutation rate: ", host_mut_rate, "\nHost mutation amount: ", host_mut_amount)
        
    ######### 2.Load Data #########
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    ######### 3.Initialize Populatioin #########
    spaceGA = Spatial_GA(hid_nodes)
    spaceGA.birth(population,hid_nodes,X_train,y_train)

    ######### 4.Run Spatial Evolution #########
    for i in range(generations):
        print("\ncurrent generation: ", i)
        val_score, cf_matrix = spaceGA.fitness(X_train, y_train, X_val, y_val)
        spaceGA.entropy_calculator(cf_matrix)
        spaceGA.probe_neighbors(dimension, neighbor_size, rou_switch, host_mut_rate, host_mut_amount)
        spaceGA.cosine_sim()

    ######### 5.Store Result #########
    spaceGA.store_result([generations, 
                        dimension,
                        rou_switch,
                        population,
                        hid_nodes,
                        neighbor_size,
                        host_mut_rate,
                        host_mut_amount])
    return None

if __name__ == "__main__":
    run()