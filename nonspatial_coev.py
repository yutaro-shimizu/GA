from cProfile import label
from re import A
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from scipy.special import kl_div
from scipy.misc import face
from scipy.spatial import distance
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
        self.mnist_score = [] # track parasite score
        self.all_train_score = []
        self.all_val_score = []
        self.neighbors = []
        self.cos_sim = []
        self.entropy = []

    ############## 1. Initialize host and parasite ##############
    def birth(self, population, hid_nodes, X_train, y_train):
        """
        Produce host and parasite. 
        The pair is pacakged into a dictionary containing the following attributes:
            - MLPClassifier model
            - training score 
            - validation score 

            and
            - parasites (10 digits from each class)
        """
        for ind in range(population):
            self.NNs[ind] = {"model": MLPClassifier(hidden_layer_sizes=(hid_nodes,), 
                                                    max_iter=1, 
                                                    alpha=1e-4,
                                                    solver='sgd', 
                                                    verbose=False, 
                                                    learning_rate_init=.1),
                            "train_score": 0,
                            "val_score": 0,
                            "parasite_X_train": None,
                            "parasite_y_train": None,
                            "parasite_score": 0}

            ### 1.1 populate host (Neural Network Classifiers) ###
            self.NNs[ind]["model"].fit(X_train, y_train) # fit the network to initialize W and b
            # randomly initialize weights and biases
            self.NNs[ind]["model"].coefs_[0] = np.random.uniform(low=-1, high=1, size=(784, hid_nodes)) 
            self.NNs[ind]["model"].coefs_[1] = np.random.uniform(low=-1, high=1, size=(hid_nodes, 10))

            ### 1.2 populate parasite (MNIST handwritten digits) ###
            indices = []
            counter = 0
            for num in np.unique(y_train, return_counts = True)[1]:
                # for each class of digit, randomly pick 10 images with replacement
                indices.extend(np.random.randint(counter, counter + num, 10))
                counter += num
            self.NNs[ind]["parasite_X_train"] = X_train[indices]
            self.NNs[ind]["parasite_y_train"] = y_train[indices]

    ############## 2.Run Non-spatial coevolution ##############
    # 2.1 calculate fitness for host
    def fitness (self, X_train, y_train, X_val, y_val, population):
        """
        Calculate max score for each generation. Store max score in the array.
        Also calculate confusion matrix.
        """
        train_score = []
        val_score = []
        parasite_score = []
        cf_matrix = []

        for ind in self.NNs:
            current_mlp = self.NNs[ind]
            # calculate training score
            current_mlp["train_score"]= current_mlp["model"].score(current_mlp["parasite_X_train"], 
                                                                   current_mlp["parasite_y_train"]) 
            # calculate validation score
            current_mlp["val_score"]= current_mlp["model"].score(X_val, y_val)

            train_score.append(current_mlp["train_score"])
            val_score.append(current_mlp["val_score"])
            

            ## output confusion matrix
            y_train_pred = current_mlp["model"].predict(current_mlp["parasite_X_train"])
            cf_matrix.append(confusion_matrix(current_mlp["parasite_y_train"], y_train_pred))

            ### compute parasite score ###
            true_result = current_mlp["parasite_y_train"] == y_train_pred
            current_mlp["parasite_score"]  = 1 - (sum(true_result) / population)

            parasite_score.append(current_mlp["parasite_score"])

        self.NNs_copy = deepcopy(self.NNs) # copy original including the scores
        print("Max training score: ", np.amax(train_score))
        print("Max validation score: ", np.amax(val_score))
        print("Max parasite score: ", np.amax(parasite_score))
        self.all_train_score.append(np.amax(train_score))
        self.all_val_score.append(np.amax(val_score))
        self.all_parasite_score.append(np.amax(parasite_score))

        return val_score, cf_matrix

    # 2.2 select the best performing parasites
    def selection(self):
        return None

    # 2.3 mutation both host and parasite
    def mutation_host(self, idx, mut_rate=0.5, mut_amount=0.005):
        """
        In each layer, mutate weights at random cites with probability "mut_rate"
        ---
        idx: int

        """
        for i in range(2):
            # randomly select mutation sites
            coef_size = self.NNs[idx]["model"].coefs_[i].size
            shape = self.NNs[idx]["model"].coefs_[i].shape
            mutation_sites = np.random.choice(coef_size, size=int(mut_rate * coef_size))\

            # mutate values
            for loci in mutation_sites:
                self.NNs[idx]["model"].coefs_[i].flat[loci] += np.random.normal(loc=mut_amount)
            self.NNs[idx]["model"].coefs_[i].reshape(shape)
        return None
    
    def mutation_parasite(self, idx, mut_rate=0.5, mut_amount=10):
        for image in self.NNs[idx]["parasite_X_train"]:
            # randomly select mutation sites
            shape = image.shape[0]
            mutation_sites = np.random.choice(shape, size=int(mut_rate * shape))

            # mutate values
            for loci in mutation_sites:
                image[loci] += np.random.normal(loc=mut_amount)
        return None

    # 2.4 combine methods to coevolve the population
    def coevolution(self):
        return None

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

        return None

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
        self.cos_sim.append(div_score)
        print("Cosine Similarity: ", div_score,"\n")
        return None

    def store_result(self, hyp_params):
        now = datetime.now().strftime("%y%m%d%H%M%S")

        d = {"train_score":self.all_train_score,
        "val_score":self.all_val_score,
        "mnist_score":self.mnist_score,
        "cos_sim":self.cos_sim,
        "rel_ent":self.entropy,
        "hyp_params":hyp_params}
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
        df.to_csv(f"./results/result_nonspatial_coev{now}.csv")
        print(f"result stored as: result_nonspatial_coev{now}.csv")
        return None

def run():
    ######### 1.Set Hyperparameters #########
    generations = int(input("Enter generations: ")) #10
    population = int(input("Enter dimension: ")) #10
    rou_switch = 0 #int(sys.argv[3])
    hid_nodes = 10 #int(sys.argv[4]) #10
    host_mut_rate = float(input("FOR HOST Enter mutation RATE (default 0.5): "))
    host_mut_amount = float(input("FOR HOST Enter mutation AMOUNT (default 0.005): "))
    parasite_mut_rate = float(input("FOR PARASITE Enter mutation RATE: "))
    parasite_mut_amount = float(input("FOR PARASITE Enter mutation AMOUNT: "))
    print("\nGenerations: ", generations)
    print("Population: ", population)
    print("Host mutation rate: ", host_mut_rate, "\nHost mutation amount: ", host_mut_amount)
    print("Parasite mutation rate: ", parasite_mut_rate, "\nParasite mutation amount: ", parasite_mut_amount)
        
    ######### 2.Load Data #########
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    ######### 3.Initialize Population(host, and parasite) #########
    model = NonSpatial_Coev_GA(hid_nodes)

    model.birth_host(population, hid_nodes, X_train, y_train)
    model.birth_parasite()
    
    ######### 4. Run Non-spatial ***Co-***Evolution #########
    for i in range(generations):
        print("\ncurrent generation: ", i)
        val_score, cf_matrix = model.host_score(X_train, y_train, X_val, y_val)
        model.parasite_score()

        model.host_selection(X_train, y_train, X_val, y_val)
        model.parasite_selection()

        model.host_mutation()
        model.parasite_mutation(population, cv_switch, selection_percent)
        
        model.entropy_calculator()
        model.cosine_sim()

    model.store_result([generations,
                    dimension,
                    rou_switch,
                    population,
                    hid_nodes,
                    host_mut_rate,
                    host_mut_amount,
                    parasite_mut_rate,
                    parasite_mut_amount,
                    neighbor_size])
    return None

if __name__=="__main__":
    run()