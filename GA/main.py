#import neural network pacakages
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml

# genetic algorithm
import numpy as np
import random

from matplotlib import pyplot as plt #package for visualization
import warnings

warnings.filterwarnings('ignore')

def load_data():
  # load data
  X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
  X = X / 255 #scale the data

  # use the traditional train/test split
  X_train, X_test = X[:60000], X[60000:]
  y_train, y_test = y[:60000], y[60000:]


####### 1. Initialization #######
# setup lists for: NNs and scores
def initialization():
  """
  Having an original and a copy is essential so not to overwrite while evolution
  """
  NNs = {}
  NNs_copy = {}

  # initialize NN and score 0 for each individual
  for ind in range(population):
    NNs[ind] = [MLPClassifier(hidden_layer_sizes=(10,), max_iter=1, alpha=1e-4,
                            solver='sgd', verbose=10, learning_rate_init=.1),0]
    NNs[ind][0].fit(X_train, y_train) # fit the network to initialize W and b

# ####### 2. Calculate fitness #######
def calculate():
  print("Calculate fitness")
  score = [] # for display purpose, store scores
  for ind in NNs:
    NNs[ind][1]= NNs[ind][0].score(X_train, y_train) # calculate the score
    score.append(NNs[ind][1])
  print("Mean score: ", np.mean(score))
  NNs_copy = NNs # clone the population

####### 3. Select top 20% #######
def selection():
  print("Evolution begins")
  lst = dict(sorted(NNs.items(), key = lambda NNs:(NNs[1][1], NNs[0]), reverse=True)) # sort the list for selection

####### 4. Evolve top 20% #######
def mutation():
  length = int(population * percent)

  children = [[] for i in range(population)] # sublist for each child
  for i in range(population): # reproduce children until full

    child = [[] for i in range(length+1)] # each child w custom number of hidden layers
    
    # cross over
    for j in range(layers + 1): # cross over for each layer

      # random selection of parents from top 20%
      prt1_idx = random.randint(0,length - 1) 
      prt2_idx = random.randint(0,length - 1)

      prt1 = NNs[prt1_idx][0].coefs_[j] #parent 1 w/ extracted weights and biases
      prt2 = NNs[prt2_idx][0].coefs_[j] #parent 2 w/ extracted weights and biases
      
      # cross over takes place HERE: current locus: 1/2
      child_coefs = np.concatenate((
          np.concatenate(prt1[0:int(len(prt1)/2)]).flat,  # vectorize prt1
          np.concatenate(prt2[int(len(prt1)/2): ]).flat)) # vectorize prt2

      # mutation
      # randomly chose a loci for mutation
      mutate_idx = np.random.choice(len(child_coefs),size = int(percent*population))
      child_coefs[mutate_idx] += random.choice([-1,1])*increment

      child[j] = child_coefs.reshape(prt1.shape) # child of weights and biases

    children[i] = child # put child in the population of children


####### 5. Calculate final score #######
def score():
  lst = dict(sorted(NNs.items(), key = lambda NNs:(NNs[1][1], NNs[0]), reverse=True)) # sort the list
  final_score = NNs[0][0].score(X_test, y_test) #fit the best model
  print(final_score)

# training vs validation%%!
# parameters
# time

def main():
  load_data()
  initialization()
  for gen in range(generations=100):
    calculate()
    selection()
    mutation()
    print("Evolution complete\n")

    # inject children's W and b to the NN objects
    for ind in NNs_copy:
      for layer in range(layers + 1): 
        NNs_copy[ind][0].coefs_[layer] = children[ind][layer]

    NNs, NNs_copy = NNs_copy, NNs # overwrite the origial copy

  score()


if __name__=="__main__":
  main()
