from turtle import shape
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt #package for visualization
import numpy as np
import pandas as pd

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

X_train, X_val, X_test, y_train, y_val, y_test = load_data()
# stochastic gradient descent

### check the size of layers: 1 hidden layer w 50 nodes
mlp = MLPClassifier(hidden_layer_sizes=(10, ), max_iter=10, alpha=1e-4, 
                    solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)
mlp.fit(X_train, y_train)
print(len(mlp.predict(X_val)))

print(f"Training set score: {mlp.score(X_train, y_train):.3f}")
print(f"Test set score: {mlp.score(X_test, y_test):.3f}")

plt.show()