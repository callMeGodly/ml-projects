import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1
    """
    c = np.amax(np.dot(theta, np.transpose(X))/temp_parameter, axis=0)
    v = np.exp(np.dot(theta, np.transpose(X)/temp_parameter)-c)
    s = np.sum(v,axis=0)
    return v/s


def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.
    """
    k, d = theta.shape
    n = Y.shape[0]
    
    reg = lambda_factor/2 * np.sum(theta**2)

    # Clip prob matrix to avoid NaN instances then takes log
    clip_m = np.log(np.clip(compute_probabilities(X, theta, temp_parameter), 1e-15, 1-1e-15))
    
    # for [[y(i) == j]]
    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape = (k,n)).toarray()
    
    # Only add terms of log(matrix of prob) where M == 1
    et = (-1/n)*np.sum(clip_m[M == 1])  

    return et + reg

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent
    """
    k = theta.shape[0]
    n = Y.shape[0]

    M = sparse.coo_matrix(([1]*n, (Y, range(n))), shape = (k,n)).toarray()

    P = compute_probabilities(X, theta, temp_parameter)

    reg = lambda_factor * theta

    err = (-1/(temp_parameter * n) * ((M-P) @ X))

    theta = theta - alpha*(err+reg)

    return theta


def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.
    """
    train_y_mod3 = train_y % 3
    test_y_mod3 = test_y % 3
    
    return train_y_mod3, test_y_mod3


def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)
    """
    
    y_pred = get_classification(X, theta, temp_parameter) % 3
    return 1 - (np.mod(y_pred, 3) == Y).mean()

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
        cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
