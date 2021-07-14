"""
https://developer.aliyun.com/article/614411#
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(2)


def initialize_parameters(n_x, n_h, n_y):
    # Input weight matrix of shape (n_x, n_h)
    W1 = np.random.randn(n_x, n_h) * 0.01
    # Input bias vector of shape (n_h)
    b1 = np.zeros((n_h))
    # output weight matrix of shape (n_h, n_y, )
    W2 = np.random.randn(n_h, n_y) * 0.01
    b2 = np.zeros((n_y))  #bias vector of shape (n_y, 1)
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


def forward_propagation(X, parameters):
    #retrieve intialized parameters from dictionary
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # Implement Forward Propagation to calculate A2 (probability)
    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)  #tanh activation function
    Z2 = np.dot(A1, W2) + b2
    A2 = 1 / (1 + np.exp(-Z2))  #sigmoid activation function
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def predict(parameters, X):
    # retrieve intialized parameters from dictionary
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    # Implement Forward Propagation to calculate A2 (probability)
    Z1 = np.dot(X, W1) + b1
    A1 = np.tanh(Z1)  # tanh activation function
    Z2 = np.dot(A1, W2) + b2
    res = 1 / (1 + np.exp(-Z2))  # sigmoid activation function
    return res


def compute_cost(A2, Y, parameters):
    m = Y.shape[0]  # number of training examples
    # Retrieve W1 and W2 from parameters
    W1 = parameters['W1']
    W2 = parameters['W2']
    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(
        (1 - Y), np.log(1 - A2))
    cost = -np.sum(logprobs) / m
    return cost


def backward_propagation(parameters, cache, X, Y):
    # Number of training examples
    m = X.shape[0]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters['W1']
    W2 = parameters['W2']

    # Retrieve A1 and A2 from dictionary "cache".
    A1 = cache['A1']
    A2 = cache['A2']

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)  # 1
    dZ1 = np.multiply(np.dot(dZ2, W2.T), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    grads = {"dW1": dW1, "db1": db1[0, :], "dW2": dW2, "db2": db2[0, :]}
    # for key in grads:
    #     print(key, grads[key].shape)
    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    n_x = 2  # layer_sizes(X, Y)[0]
    n_y = 1  # layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2.
    # Inputs: "n_x, n_h, n_y".
    # Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    print("Layer Struct : ", W1.shape, W2.shape)

    # Loop (gradient descent)
    for i in range(0, num_iterations):  # num_iterations

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters, n_h


def plot_decision_boundary(parameters, input_x, input_y):
    # Set min and max values and give it some padding
    x_min, x_max = input_x[:, 0].min() - 0.25, input_x[:, 0].max() + 0.25
    y_min, y_max = input_x[:, 1].min() - 0.25, input_x[:, 1].max() + 0.25
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    model = lambda x: predict(parameters, x)

    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(input_x[:, 0], input_x[:, 1], c=input_y, cmap=plt.cm.Spectral)
    plt.title("Decision Boundary for hidden layer size " + str(6))
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.show()
    plt.tight_layout()
    plt.close()
    return None


def load_iris(show=False):
    iris = pd.read_csv('data/iris.csv')
    # Create numeric classes for species (0,1,2)
    iris.loc[iris['species'] == 'virginica', 'species_id'] = 0
    iris.loc[iris['species'] == 'versicolor', 'species_id'] = 1
    iris.loc[iris['species'] == 'setosa', 'species_id'] = 2
    iris = iris[iris['species_id'] != 2]
    # Create Input and Output columns
    X = iris[['petal_length', 'petal_width']].values
    Y = iris[['species_id']].values
    Y = Y.astype('uint8')
    if show:
        # Make a scatter plot
        plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], s=40, cmap=plt.cm.Spectral)
        plt.title("IRIS DATA | Blue - Versicolor, Red - Virginica ")
        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')
        plt.tight_layout()
        plt.show()
        plt.close()
    return X, Y


if __name__ == '__main__':

    input_x, input_y = load_iris()
    print(input_x.shape, input_y.shape)

    parameters, hh = nn_model(input_x,
                              input_y,
                              n_h=6,
                              num_iterations=10000,
                              print_cost=True)
    for key in parameters:
        print(key, parameters[key].shape)

    plot_decision_boundary(parameters, input_x, input_y)
