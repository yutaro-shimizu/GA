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

class NonSpatial_GA:
  def __init__(self, hid_nodes):
        self.NNs = {}  # set of models for evolution. Swaps based on training score during evolution. 
        self.NNs_copy = {}  # for reference during evolution. Does not change during swaps.
        self.all_train_score = []
        self.all_val_score = []
        self.cos_sim = []
        self.entropy = []

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
          
          #self.NNs[ind]["model"].coefs_[0] = np.random.uniform(low=-1, high=1, size=(784, hid_nodes)) 
          #self.NNs[ind]["model"].coefs_[1] = np.random.uniform(low=-1, high=1, size=(hid_nodes, 10))

          # Xavier Initialization
          self.NNs[ind]["model"].coefs_[0] = np.random.random(size=(784, hid_nodes)) * np.sqrt(2/(784+10))/100
          self.NNs[ind]["model"].coefs_[1] = np.random.random(size=(hid_nodes, 10)) * np.sqrt(2/(784+10))/100

  def score_calculator(self, X_train, y_train, X_val, y_val):
      """
      Calculate max score for each generation. Store max score in the array.
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

  def selection(self):
    self.NNs = dict(sorted(self.NNs.items(), key = lambda NNs:(NNs[1]["train_score"], NNs[0]), reverse=True)) # sort the list for selection
    self.NNs = {i: v for i, v in enumerate(self.NNs.values())}  
    self.NNs_copy = deepcopy(self.NNs) # clone the population

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
  
  def mutation(self, child_coefs,mut_rate=0.5):
    # mutation
    # randomly chose loci for mutation
    """
    check child_coefs.size and prt1.size
    """
    mutate_idx = np.random.choice(child_coefs.size, int(mut_rate*child_coefs.size))
    for idx in mutate_idx:
      child_coefs[idx] += np.random.normal(loc=0.005)
    return child_coefs

  def inject_weights(self,children):
    # inject children's W and b to the NN objects
    for ind in self.NNs_copy:
      for layer in range(2): 
        self.NNs_copy[ind]["model"].coefs_[layer] = children[ind][layer]

    self.NNs, self.NNs_copy = deepcopy(self.NNs_copy), deepcopy(self.NNs) # overwrite the origial copy
    
  def evolution(self, population, cv_switch, selection_percent, mut_rate=0.05):
    num_selected = int(population * selection_percent)
    
    children = [[] for i in range(population)] # sublist for each child
    for i in range(population): # reproduce children until full
      child = []
      for j in range(2): # cross over for each layer (input and hidden)
        child_coefs, shape = self.crossover(num_selected, j, cv_switch)
        child_coefs = self.mutation(child_coefs)

        child.append(child_coefs.reshape(shape)) # child of weights and biases
      children[i] = child # put child in the population of children
    self.inject_weights(children)

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
  # input hyperparameters from the shell script
  generations = int(sys.argv[1]) #10
  population = int(sys.argv[2]) #10
  hid_nodes = int(sys.argv[3]) #10
  selection_percent = 0.2 #int(sys.argv[4]) #20
  cv_switch = False #bool(sys.argv[5])
  print("generations: ", generations)
  print("population: ", population)

  ############## 2. load data ##############
  X_train, X_val, X_test, y_train, y_val, y_test = load_data()

  ######### 3.Initialize Populatioin #########
  model = NonSpatial_GA(hid_nodes)
  model.birth(population, hid_nodes, X_train, y_train)

  ######### 4.Run Non-spatial Evolution #########
  for i in range(generations):
    print("\ncurrent generation: ", i)
    val_score, cf_matrix = model.score_calculator(X_train, y_train, X_val, y_val)
    model.entropy_calculator(cf_matrix)
    model.selection()
    model.evolution(population, cv_switch, selection_percent)
    model.cosine_sim()
  model.store_result([generations, 
                      population, 
                      hid_nodes, 
                      selection_percent,
                      cv_switch])
  return None

if __name__ == "__main__":
    run() 

"""
# def plot(self):
  #   COLOUR = 'white'
  #   plt.rcParams['text.color'] = COLOUR
  #   plt.rcParams['axes.labelcolor'] = COLOUR
  #   plt.rcParams['axes.edgecolor'] = COLOUR
  #   plt.rcParams['axes.facecolor'] = 'black'
  #   plt.rcParams['xtick.color'] = COLOUR
  #   plt.rcParams['ytick.color'] = COLOUR

  #   plt.figure(facecolor="black")
  #   plt.plot(self.all_train_score, label = "training")
  #   plt.plot(self.all_val_score, label = "validation")
  #   plt.legend()
  #   plt.xlabel("Generations")
  #   plt.ylabel("Max accuracy")
  #   plt.legend()
  #   plt.savefig('./Figures/nonspatial_final.png', transparent=True)

  #   plt.figure(facecolor="black")
  #   plt.plot(self.diversity)
  #   plt.xlabel("Generations")
  #   plt.ylabel("Cosine Similarity")
  #   plt.savefig('./Figures/nonspatial_diversity.png', transparent=True)
  """