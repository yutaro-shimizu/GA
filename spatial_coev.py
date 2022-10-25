import numpy as np
import random
import sys

#import neural network pacakages
from sklearn.neural_network import MLPClassifier
from scipy.spatial import distance
from copy import deepcopy

import pandas as pd
from matplotlib import pyplot as plt #package for visualization
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

############## 1. Import hyperparameters ##############
############## 2. Initialize parasite (load data) ##############
# setup data
# setup score

############## 3. Initialize host ##############
#sklearn models

############## 4.Run Non-spatial coevolution ##############
for i in range(generations):
    # 4.1 calculate fitness for host
    # 4.2 calculate fitness for parasite 
        # how do i calculate the score for parasites?
    # 4.2 select the best performing two individuals
    # 4.3 select samples
    # 4.4 breed and mutate host
    # 4.5 mutate samples (parasite)

for i in range(generations):

    model.calculator(X_train, y_train, X_val, y_val)
    model.selection()
    model.evolution(population, cv_switch, selection_percent)
    model.cosine_sim()
