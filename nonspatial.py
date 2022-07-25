#import neural network pacakages
from sklearn.neural_network import MLPClassifier

# genetic algorithm
import numpy as np
import random
import pandas as pd
import sys

import timeit
# import cProfile
# import pstats

from matplotlib import pyplot as plt #package for visualization
import warnings

warnings.filterwarnings('ignore')

tic_all = timeit.default_timer()

# input hyperparameters from the shell script
generations = int(sys.argv[1]) #10
population = int(sys.argv[2]) #10
hid_nodes = 10 #int(sys.argv[3]) #10
selection_percent = 0.2 #int(sys.argv[4]) #20
mut_rate = 0.5 #float(sys.argv[5]) #0.05
print("Total arguments: ", len(sys.argv))
print("generations: ", sys.argv[1])
print("population: ", sys.argv[2])
# print("hid_nodes: ", sys.argv[3])
# print("select_percent: ", sys.argv[4])
# print("mut_rate: ", sys.argv[5])

# load data
print("load data")

data_train = pd.read_csv('mnist_train.csv') #load MNIST training data in
data_train = np.array(data_train) #turn into array
m, n =data_train.shape
np.random.shuffle(data_train)
Y_train=data_train[:,0]
X_train=data_train[:,1:n]

data_test = pd.read_csv('mnist_test.csv') #validating data loaded in
data_test = np.array(data_test) #turned to array and transposed
p, q = data_test.shape
np.random.shuffle(data_test)

y_test =data_test[:,0] #first row of data
X_test = data_test[:,1:q] #rest of data

#next two lines are taking 10,000 samples from MNIST
X_train, X_val = X_train[:10000], X_train[10000:20000]
y_train, y_val = Y_train[:10000], Y_train[10000:20000]

print("load data complete")

# # stochastic gradient descent

# ### check the size of layers: 1 hidden layer w 50 nodes
# mlp = MLPClassifier(hidden_layer_sizes=(10, ), max_iter=10, alpha=1e-4, 
#                     solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)
# mlp.fit(X_train, y_train)

# print(f"Training set score: {mlp.score(X_train, y_train):.3f}")
# print(f"Test set score: {mlp.score(X_test, y_test):.3f}")

# plt.show()


####### 1. Initialization #######
# setup lists for: NNs and scores
"""
Having an original and a copy is essential so not to overwrite while evolution
"""
NNs = {}
NNs_copy = {}

# store training and validation scores
training_score = []
validation_score = []

# initialize NN and score 0 for each individual
for ind in range(population):
  NNs[ind] = {"model":MLPClassifier(hidden_layer_sizes=(hid_nodes,), max_iter=1, alpha=1e-4,
                          solver='sgd', verbose=10, learning_rate_init=.1),
                          "train_score":0,
                          "val_score":0}
  NNs[ind]["model"].fit(X_train, y_train) # fit the network to initialize W and b
# randomly initialize weights and biases
  NNs[ind]["model"].coefs_[0] = np.random.uniform(low=-1,high=1,size=(784,hid_nodes)) 
  NNs[ind]["model"].coefs_[1] = np.random.uniform(low=-1,high=1,size=(hid_nodes,10)) 

# start training
for gen in range(generations):
  print("Current generation: ", gen)

  # ####### 2. Calculate fitness #######
  tic_fitness = timeit.default_timer()
  # profiler = cProfile.Profile()
  # profiler.enable()

  train_score = [] # for display purpose, store scores
  val_score = []

  for ind in NNs:
    NNs[ind]["train_score"]= NNs[ind]["model"].score(X_train, y_train) # calculate the score
    NNs[ind]["val_score"]= NNs[ind]["model"].score(X_val, y_val)
    train_score.append(NNs[ind]["train_score"])
    val_score.append(NNs[ind]["val_score"])
  print("Max training score: ", np.amax(train_score))
  print("Max validation score: ", np.amax(val_score))
  training_score.append(np.amax(train_score))
  validation_score.append(np.amax(val_score))
  # profiler.disable()
  # stats = pstats.Stats(profiler).sort_stats("tottime")

  toc_fitness = timeit.default_timer()
  print('fitness time: ',toc_fitness-tic_fitness)

  ####### 3. Select top 20% #######
  tic_select = timeit.default_timer()
  NNs = dict(sorted(NNs.items(), key = lambda NNs:(NNs[1]["train_score"], NNs[0]), reverse=True)) # sort the list for selection
  NNs = {i: v for i, v in enumerate(NNs.values())}  
  NNs_copy = NNs # clone the population
  toc_select = timeit.default_timer()
  print('sorting time: ',toc_select-tic_select)

  ####### 4. Evolve top 20% #######
  tic_evo = timeit.default_timer()
  num_selected = int(population * selection_percent)

  children = [[] for i in range(population)] # sublist for each child
  for i in range(population): # reproduce children until full

    child = [] # each child w custom number of hidden layers
    
    # random selection of parents from top 20%
    # prt1_idx = random.randint(0,num_selected - 1) 
    # prt2_idx = random.randint(0,num_selected - 1)

    # p1 = NNs[prt1_idx]['score']
    # p2 = NNs[prt2_idx]['score']


    # cross over
    for j in range(2): # cross over for each layer

      # # random selection of parents from top 20%
      prt1_idx = random.randint(0,num_selected - 1) 
      prt2_idx = random.randint(0,num_selected - 1)

      prt1 = NNs[prt1_idx]["model"].coefs_[j] #parent 1 w/ extracted weights and biases
      prt2 = NNs[prt2_idx]["model"].coefs_[j] #parent 2 w/ extracted weights and biases
      
      # cross over takes place HERE
      locus = random.randint(1,len(NNs[prt1_idx]["model"].coefs_[j])-1)
      child_coefs = np.concatenate((prt1.flat[0:locus], prt2.flat[locus: ])) # vectorize prt2

      # mutation
      # randomly chose loci for mutation
      mutate_idx = np.random.choice(child_coefs.size,size = int(mut_rate*prt1.size))
      for idx in mutate_idx:
        child_coefs[idx] += np.random.normal(loc=0.1)

      child.append(child_coefs.reshape(prt1.shape)) # child of weights and biases

    children[i] = child # put child in the population of children
  toc_evo = timeit.default_timer()
  print("evolution time:", toc_evo - tic_evo)
  print("Evolution complete\n")

  # inject children's W and b to the NN objects
  for ind in NNs_copy:
    for layer in range(2): 
      NNs_copy[ind]["model"].coefs_[layer] = children[ind][layer]

  NNs, NNs_copy = NNs_copy, NNs # overwrite the origial copy

####### 5. Calculate final score #######
NNs = dict(sorted(NNs.items(), key = lambda NNs:(NNs[1]["train_score"], NNs[0]), reverse=True)) # sort the list for selection
NNs = {i: v for i, v in enumerate(NNs.values())}
final_score = NNs[0]["model"].score(X_test, y_test) #fit the best model
print(final_score)

toc_all = timeit.default_timer()
print("total time: ", toc_all - tic_all)
# stats.print_stats() #print the stats report for profiling

COLOUR = 'white'
plt.rcParams['text.color'] = COLOUR
plt.rcParams['axes.labelcolor'] = COLOUR
plt.rcParams['axes.edgecolor'] = COLOUR
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['xtick.color'] = COLOUR
plt.rcParams['ytick.color'] = COLOUR

plt.figure(facecolor="black")
plt.plot(training_score, label = "training")
plt.plot(validation_score, label = "validation")
plt.legend()
plt.xlabel("generations")
plt.ylabel("max accuracy of individual")
plt.legend()
plt.savefig('./Figures/final.png', transparent=True)