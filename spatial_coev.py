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
import os
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

class Spatial_Coev_GA():
    def __init__(self):
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

        self.NNs = {} # set of models for evolution. Swaps based on training score during evolution. 
        self.NNs_copy = {}  # for reference during evolution. Does not change during swaps.
        self.all_train_score = []
        self.all_val_score = []
        self.all_parasite_score = []
        self.neighbors = []
        self.cos_sim = []
        self.entropy = []
    
    ############## 1. Initialize host ##############
    def birth(self, population, hid_nodes, X_train, y_train):
        """
        Produce host and parasite. 
        The pair is pacakged into each dictionary containing:
            - MLPClassifier model
            - training score 
            - validation score attributes

            and
            - parasites (10 digits from each class)
        
        Line 113 and 114 represent the genome of neural networks as genotype
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
                # for each grid, load 110 digit
                indices.extend(np.random.randint(counter, counter + num, 11))
                counter += num
            self.NNs[ind]["parasite_X_train"] = X_train[indices]
            self.NNs[ind]["parasite_y_train"] = y_train[indices]

    ############## 2.Run Spatial coevolution ##############
    # 2.1 calculate fitness for host
    def fitness (self, X_train, y_train, X_val, y_val, population):
        """
        Calculate max score for the host and paraste in each generation.
        Also output confusion matrix of host for phenotype measure.

        Fitness for host NN algorithm (line 144 and line 147): correct classification percentage
        Fitness for host MNIST algoirithm (line 158 and line 159): misclassifiation percentage
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
            current_mlp["parasite_score"]  = 1 - (sum(true_result) / len(true_result))

            parasite_score.append(current_mlp["parasite_score"])

        self.NNs_copy = deepcopy(self.NNs) # copy original including the scores
        print("Max training score: ", np.amax(train_score))
        print("Max validation score: ", np.amax(val_score))
        print("Max parasite score: ", np.amax(parasite_score))
        self.all_train_score.append(np.amax(train_score))
        self.all_val_score.append(np.amax(val_score))
        self.all_parasite_score.append(np.amax(parasite_score))

        return val_score, cf_matrix

    # 2.2 select the best performing individuals
    def selection(self, i, j, dim, neigh_size, rou_switch, host_or_parasite): 
        """
        Select top perfoming individuals.

        This is a probabilistic replacement where the top performing individuals are proportionately selected.
        The elitist strategy is from Mitchell (2006), howerver recent suggestions consider diveristy Mouret, J. B. (2020). 
        """
        score = self.NNs_copy[i * dim + j][host_or_parasite]
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
                    fit_lst.append(self.NNs_copy[((i + k) % dim) * dim + (j + l) % dim][host_or_parasite])
                    indices_lst.append(((i + k) % dim) * dim + (j + l) % dim)
                ### deterministic replacement###
                else:
                    if score < self.NNs_copy[((i + k) % dim) * dim + (j + l) % dim][host_or_parasite]:
                        score = self.NNs_copy[((i + k) % dim) * dim + (j + l) % dim][host_or_parasite]
                        idx = ((i + k) % dim) * dim + (j + l) % dim
        ### roulette wheel selection ###
        if rou_switch:
            sum_fit = sum(fit_lst)
            idx = np.random.choice(indices_lst, 1, p = fit_lst / sum_fit)[0] # returns index as nd array
        return idx

    # 2.3 mutation both host and parasite
    def mutation_host(self, idx, mut_rate=0.5, mut_amount=0.005):
        """
        In each layer, mutate weights at random cites with probability "mut_rate",  Mitchell (2006).
        ---
        idx: int

        line 219 -line 221: select mutation location
        line 224 - line 226: mutate mut_amount sampled from a normal distribution
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
    def coevolution(self, 
                    dim, 
                    neigh_size, 
                    rou_switch, 
                    host_mut_rate, 
                    host_mut_amount, 
                    parasite_mut_rate,
                    parasite_mut_amount):
        """
        - Check every cell and apply probabilistic replacement and mutation
        - Mutation happens for each layer
        """
        for i in range(dim): # for every row
            for j in range(dim): #for every column
                idx = i * dim + j # convert row and column notations to an index 

                # host neural network: probe max neighbor
                host_new_idx = self.selection(i, j, dim, neigh_size, rou_switch, "train_score")
                self.NNs[idx]["model"] = deepcopy(self.NNs_copy[host_new_idx]["model"])

                # parasite MNIST data: probe max neighbor
                parasite_new_idx = self.selection(i, j, dim, neigh_size, rou_switch, "parasite_score")
                self.NNs[idx]["parasite_X_train"] = deepcopy(self.NNs_copy[parasite_new_idx]["parasite_X_train"])
                self.NNs[idx]["parasite_y_train"] = deepcopy(self.NNs_copy[parasite_new_idx]["parasite_y_train"])

                self.mutation_host(idx, host_mut_rate, host_mut_amount)
                self.mutation_parasite(idx, parasite_mut_rate, parasite_mut_amount)

    ############## 3.Measure phenotype, genotype and store result ##############
    def entropy_calculator(self, cf_matrix):
        """
        Compute KL-divergence (distance between metrices) to characterize phenotype Mitchell 2006.
        normalize each row of the confusion matrix for KL-Divergence calculation
        """
        entropy_periter = []

        for i in range(len(cf_matrix)):
            for j in range(len(cf_matrix)):
                if i >= j:
                    continue
                for row in range(10):
                    # add 1e-4 to avoid overflow (division by 0 for log)
                    entropy_periter.append(kl_div(normalize(1e-4+cf_matrix[i], axis=1, norm='l1')[row],
                                                normalize(1e-4+cf_matrix[j], axis=1, norm='l1')[row]))

        mean_KL = np.average(entropy_periter)
        self.entropy.append(mean_KL)
        print("KL-Divergence: ", mean_KL)

        return None

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
        return None

    def init_path(self):
        now = datetime.now().strftime("%y%m%d%H%M%S")
        path = f"./results/spatial_coev{now}"
        os.mkdir(path)

        return path

    def store_result(self, hyp_params, path):
        d = {"train_score":self.all_train_score,
        "val_score":self.all_val_score,
        "mnist_score":self.all_parasite_score,
        "cos_sim":self.cos_sim,
        "rel_ent":self.entropy,
        "hyp_params":hyp_params}
        df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
        df.to_csv(path+"/result_spatial_coev.csv")
        return None

    def mnist_visualizer(self, path, iter):
        # plot images
        fig, axes = plt.subplots(2, 5)
        for i in range(10):
            image = self.NNs[0]['parasite_X_train'][i]
            label = self.NNs[0]['parasite_y_train'][i]
            ax = axes[i//5, i%5]
            ax.imshow(np.reshape(image,(28,28)), cmap='gray')
            ax.set_title('Label: {}'.format(label))
        plt.savefig(path+f"/spatial_coevo_{iter}.png")

def run():
   ######### 1.Set Hyperparameters #########
    """
    Take in all hyperparameters to initiqalize the genetic population. Additional parasite hyperparameters.
    Major hyperparameters are from Mitchell (2006). Reference CP193 submission for references.
    """
    generations = int(input("Enter generations: ")) #10
    dimension = int(input("Enter dimension: ")) #10
    rou_switch = 0 #int(sys.argv[3])
    population = dimension ** 2
    hid_nodes = 10 #int(sys.argv[4]) #10
    host_mut_rate = float(input("FOR HOST Enter mutation RATE (default 0.5): "))
    host_mut_amount = float(input("FOR HOST Enter mutation AMOUNT (default 0.005): "))
    parasite_mut_rate = float(input("FOR PARASITE Enter mutation RATE: "))
    parasite_mut_amount = float(input("FOR PARASITE Enter mutation AMOUNT: "))
    visualize_per = int(input("Visualize every how many iterations?"))
    neighbor_size = 3
    print("\nGenerations: ", generations)
    print("Population: ", population)
    print("Host mutation rate: ", host_mut_rate, "\nHost mutation amount: ", host_mut_amount)
    print("Parasite mutation rate: ", parasite_mut_rate, "\nParasite mutation amount: ", parasite_mut_amount)
        
    ######### 2.Load Data #########
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    ######### 3.Initialize Population(host, and parasite) #########
    model = Spatial_Coev_GA()
    model.birth(population, hid_nodes, X_train, y_train)
    
    ######### 4. Run Non-spatial *CoEvolution #########
    for i in range(generations):
        print("\ncurrent generation: ", i)
        val_score, cf_matrix = model.fitness(X_train, 
                                             y_train, 
                                             X_val, 
                                             y_val, 
                                             population)
        model.coevolution(dimension, 
                          neighbor_size, 
                          rou_switch, 
                          host_mut_rate, 
                          host_mut_amount,
                          parasite_mut_rate,
                          parasite_mut_amount)
        # model.entropy_calculator(cf_matrix)
        # model.cosine_sim()

        if i == 0:
            path = model.init_path()
        # if i % visualize_per == 0:
        #     model.mnist_visualizer(path, i)

    model.store_result([generations,
                        dimension,
                        rou_switch,
                        population,
                        hid_nodes,
                        host_mut_rate,
                        host_mut_amount,
                        parasite_mut_rate,
                        parasite_mut_amount,
                        neighbor_size], path)
    return None

if __name__ == "__main__":
    run()
    