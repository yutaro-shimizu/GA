from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt #package for visualization

# stochastic gradient descent

### check the size of layers: 1 hidden layer w 50 nodes
mlp = MLPClassifier(hidden_layer_sizes=(10, ), max_iter=10, alpha=1e-4, 
                    solver='sgd', verbose=10, random_state=1, learning_rate_init=.1)
mlp.fit(X_train, y_train)

print(f"Training set score: {mlp.score(X_train, y_train):.3f}")
print(f"Test set score: {mlp.score(X_test, y_test):.3f}")

plt.show()