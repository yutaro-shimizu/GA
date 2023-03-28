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

    #next two lines are taking 1,000 samples from MNIST
    X_train, X_val = X_train[:1000], X_train[1000:2000]
    y_train, y_val = Y_train[:1000], Y_train[1000:2000]

    print("load data complete")
    return X_train, X_val, X_test, y_train, y_val, y_test

class NonSpatial_GA:
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
    self.cos_sim = []
    self.entropy = []

  ############## 1. Initialize host algorithm ##############
  def birth(self, population, hid_nodes, X_train, y_train):
      """
      Produce population each individual containing model, training score and validation score attributes.
      Line 95 and 96 represent the genome of neural networks as genotype

      """
      for ind in range(population):
          self.NNs[ind] = {"model": MLPClassifier(hidden_layer_sizes=(hid_nodes,), max_iter=1, alpha=1e-4,
                        solver='sgd', verbose=False, learning_rate_init=.1),
                      "train_score": 0,
                      "val_score": 0}
          self.NNs[ind]["model"].fit(X_train, y_train) # fit the network to initialize W and b
          # randomly initialize weights and biases
          
          #self.NNs[ind]["model"].coefs_[0] = np.random.uniform(low=-1, high=1, size=(784, hid_nodes)) 
          #self.NNs[ind]["model"].coefs_[1] = np.random.uniform(low=-1, high=1, size=(hid_nodes, 10))

          # Xavier Initialization
          self.NNs[ind]["model"].coefs_[0] = np.random.random(size=(784, hid_nodes)) * np.sqrt(2/(784+10))/100
          self.NNs[ind]["model"].coefs_[1] = np.random.random(size=(hid_nodes, 10)) * np.sqrt(2/(784+10))/100

  ############## 2.Run Non-spatial evolution ##############
  # 2.1 calculate fitness for host algorithm
  def fitness(self, X_train, y_train, X_val, y_val):
      """
        Calculate max score for the host and paraste in each generation.
        Also output confusion matrix of host for phenotype measure.

        Fitness for host NN algorithm (line 115): correct classification percentage
      """
      train_score = []
      val_score = []
      cf_matrix = []

      for ind in self.NNs:
          self.NNs[ind]["train_score"]= self.NNs[ind]["model"].score(X_train, y_train) # calculate the score
          self.NNs[ind]["val_score"]= self.NNs[ind]["model"].score(X_val, y_val)
          train_score.append(self.NNs[ind]["train_score"])
          val_score.append(self.NNs[ind]["val_score"])

          ## output confusion matrix and compute relative entropy
          y_val_pred = self.NNs[ind]["model"].predict(X_val)
          cf_matrix.append(confusion_matrix(y_val,y_val_pred))

      print("Max training score: ", np.amax(train_score))
      print("Max validation score: ", np.amax(val_score))
      self.all_train_score.append(np.amax(train_score))
      self.all_val_score.append(np.amax(val_score))

      return val_score, cf_matrix
  
  # 2.2 select the best performing individuals
  def selection(self):
    """
    Select top perfoming individuals.
    This is a probabilistic replacement where the top performing individuals are proportionately selected.
    The elitist strategy is from Mitchell (2006), howerver recent suggestions consider diveristy Mouret, J. B. (2020). 
    """
    self.NNs = dict(sorted(self.NNs.items(), key = lambda NNs:(NNs[1]["train_score"], NNs[0]), reverse=True)) # sort the list for selection
    self.NNs = {i: v for i, v in enumerate(self.NNs.values())}  
    self.NNs_copy = deepcopy(self.NNs) # clone the population

  # 2.3 evolutionary mechanisms
  # crossover is turned off for controlled-intervention
  def crossover(self, num_selected, j, cv_switch, mut_rate=0.05):
    # # random selection of parents from top 20%
    prt1_idx = random.randint(0,num_selected - 1) 
    prt1 = self.NNs[prt1_idx]["model"].coefs_[j] #parent 1 w/ extracted weights and biases

    if cv_switch: # if cross over happens identify the second parent 
      prt2_idx = random.randint(0,num_selected - 1)
      prt2 = self.NNs[prt2_idx]["model"].coefs_[j] #parent 2 w/ extracted weights and biases

      # cross over takes place HERE
      locus = random.randint(1,len(self.NNs[prt1_idx]["model"].coefs_[j])-1)
      child_coefs = np.concatenate((prt1.flat[0:locus], prt2.flat[locus: ])) # vectorize prt2
    else:
      child_coefs = np.ravel(prt1)
    return child_coefs, prt1.shape
  
  def mutation(self, child_coefs, host_mut_rate=0.5, host_mut_amount=0.005):
    # mutation
    # randomly chose loci for mutation
    """
    In each layer, mutate weights at random cites with probability "mut_rate",  Mitchell (2006).
        ---
        idx: int
    line 167: select mutation location
    line 177 - 178: mutate mut_amount sampled from a normal distribution
    """
    mutate_idx = np.random.choice(child_coefs.size, int(host_mut_rate*child_coefs.size))
    for idx in mutate_idx:
      child_coefs[idx] += np.random.normal(loc=host_mut_amount)
    return child_coefs

  def inject_weights(self,children):
    # inject children's W and b to the NN objects
    for ind in self.NNs_copy:
      for layer in range(2): 
        self.NNs_copy[ind]["model"].coefs_[layer] = children[ind][layer]

    self.NNs, self.NNs_copy = deepcopy(self.NNs_copy), deepcopy(self.NNs) # overwrite the origial copy
    
  # 2.4 combine methods for evolution
  def evolution(self, population, cv_switch, selection_percent, host_mut_rate, host_mut_amount):
    num_selected = int(population * selection_percent)
    """
    Combine evolutionary operators. This method is swappable to any other operators.
    """
    children = [[] for i in range(population)] # sublist for each child
    for i in range(population): # reproduce children until full
      child = []
      for j in range(2): # cross over for each layer (input and hidden)
        child_coefs, shape = self.crossover(num_selected, j, cv_switch)
        child_coefs = self.mutation(child_coefs, host_mut_rate, host_mut_amount)

        child.append(child_coefs.reshape(shape)) # child of weights and biases
      children[i] = child # put child in the population of children
    self.inject_weights(children)

  ############## 3.Measure phenotype, genotype and store result ##############
  def entropy_calculator(self, cf_matrix):
        """
        Compute KL-divergence (distance between metrices) to characterize phenotype Mitchell (2006).
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
          df.to_csv(f"./results/result_nonspatial{now}.csv")

def run():
  ############## 1. import hyperparameters ##############
  """
    Take in all hyperparameters to initiqalize the genetic population. Additional parasite hyperparameters.
    Major hyperparameters are from Mitchell (2006). Reference CP193 submission for references.
  """
  generations = int(input("Enter generations: ")) #10
  population = int(input("Enter population: "))
  hid_nodes = 10 #int(sys.argv[4]) #10
  selection_percent = 0.2 #int(sys.argv[4]) #20
  cv_switch = bool(input("Enter crossover switch (True/False): "))
  host_mut_rate = float(input("FOR HOST Enter mutation RATE (default 0.5): "))
  host_mut_amount = float(input("FOR HOST Enter mutation AMOUNT (default 0.005): "))
  print("\nGenerations: ", generations)
  print("Population: ", population)
  print("Cross Over: ", cv_switch)
  print("Host mutation rate: ", host_mut_rate, "\nHost mutation amount: ", host_mut_amount)

  ############## 2. Load Data ##############
  X_train, X_val, X_test, y_train, y_val, y_test = load_data()

  ######### 3.Initialize Populatioin #########
  model = NonSpatial_GA(hid_nodes)
  model.birth(population, hid_nodes, X_train, y_train)

  ######### 4.Run Non-spatial Evolution #########
  for i in range(generations):
    print("\ncurrent generation: ", i)
    val_score, cf_matrix = model.fitness(X_train, y_train, X_val, y_val)
    model.entropy_calculator(cf_matrix)
    model.selection()
    model.evolution(population, cv_switch, selection_percent, host_mut_rate, host_mut_amount)
    model.cosine_sim()
  model.store_result([generations, 
                      population,
                      hid_nodes,
                      selection_percent,
                      cv_switch,
                      host_mut_rate,
                      host_mut_amount])
  return None

if __name__ == "__main__":
    run() 