#import neural network pacakages
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml

# genetic algorithm
import numpy as np
import random

from matplotlib import pyplot as plt #package for visualization
import warnings

warnings.filterwarnings('ignore')

# load data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255 #scale the data

# use the traditional train/test split
X_train, X_val, X_test = X[:10000], X[10000:20000], X[20000:30000]
y_train, y_val, y_test = y[:10000], y[10000:20000], y[20000:30000]

# # stochastic gradient descent

# ### check the size of layers: 1 hidden layer w 50 nodes
# mlp = MLPClassifier(hidden_layer_sizes=(10, ), max_iter=10, alpha=1e-4, 
#                     solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)
# mlp.fit(X_train, y_train)

# print(f"Training set score: {mlp.score(X_train, y_train):.3f}")
# print(f"Test set score: {mlp.score(X_test, y_test):.3f}")

# plt.show()

generations = 10
population = 10
hid_nodes = 10
selection_percent = 0.2 
mut_rate = 0.05
mut_increment = 0.01

####### 1. Initialization #######
# setup lists for: NNs and scores
"""
Having an original and a copy is essential so not to overwrite while evolution
"""
NNs = {}
NNs_copy = {}

# initialize NN and score 0 for each individual
for ind in range(population):
  NNs[ind] = {"model":MLPClassifier(hidden_layer_sizes=(hid_nodes,), max_iter=1, alpha=1e-4,
                          solver='sgd', verbose=10, learning_rate_init=.1),
                          "score":0}
  NNs[ind]["model"].fit(X_train, y_train) # fit the network to initialize W and b

# start training
for gen in range(generations):
  print("Current generation: ", gen)

  # ####### 2. Calculate fitness #######
  print("Calculate fitness🧮")
  score = [] # for display purpose, store scores
  for ind in NNs:
    NNs[ind]["score"]= NNs[ind]["model"].score(X_train, y_train) # calculate the score
    score.append(NNs[ind]["score"])
  print("Mean score: ", np.mean(score))
  NNs_copy = NNs # clone the population

  ####### 3. Select top 20% #######
  print("Evolution begins🦠👩‍🏭")
  lst = dict(sorted(NNs.items(), key = lambda NNs:(NNs[1]["score"], NNs[0]), reverse=True)) # sort the list for selection

  ####### 4. Evolve top 20% #######
  num_selected = int(population * selection_percent)

  children = [[] for i in range(population)] # sublist for each child
  for i in range(population): # reproduce children until full

    child = [] # each child w custom number of hidden layers
    
    # cross over
    for j in range(2): # cross over for each layer

      # random selection of parents from top 20%
      prt1_idx = random.randint(0,num_selected - 1) 
      prt2_idx = random.randint(0,num_selected - 1)

      prt1 = NNs[prt1_idx]["model"].coefs_[j] #parent 1 w/ extracted weights and biases
      prt2 = NNs[prt2_idx]["model"].coefs_[j] #parent 2 w/ extracted weights and biases
      
      # cross over takes place HERE

      locus = random.randint(1,len(prt1)-1)

      child_coefs = np.concatenate((prt1.flat[0:locus], prt2.flat[locus: ])) # vectorize prt2

      # mutation
      # randomly chose loci for mutation
      mutate_idx = np.random.choice(len(child_coefs),size = int(mut_rate*len(prt1)))
      for idx in mutate_idx:
        child_coefs[idx] += random.choice([-1,1])*mut_increment

      child.append(child_coefs.reshape(prt1.shape)) # child of weights and biases

    children[i] = child # put child in the population of children
  print("Evolution complete🦠✨")

  # inject children's W and b to the NN objects
  for ind in NNs_copy:
    for layer in range(2): 
      NNs_copy[ind]["model"].coefs_[layer] = children[ind][layer]

  NNs, NNs_copy = NNs_copy, NNs # overwrite the origial copy

####### 5. Calculate final score #######
lst = dict(sorted(NNs.items(), key = lambda NNs:(NNs[1]["score"], NNs[0]), reverse=True)) # sort the list
final_score = NNs[0]["model"].score(X_test, y_test) #fit the best model
print(final_score)

# training vs validation%%!
# parameters
# time