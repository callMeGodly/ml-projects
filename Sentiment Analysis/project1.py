from filecmp import DEFAULT_IGNORES
from select import KQ_NOTE_LOWAT
from string import punctuation, digits
from turtle import update
import numpy as np
import random

def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """

    to_compare = label*(np.dot(feature_vector, theta) + theta_0);
    to_compare = 1 - to_compare if to_compare < 1 else 0 

    return to_compare


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    
    total_loss = 0
    feature_nums = feature_matrix.shape[0]
    for i in range(0, feature_nums):
        total_loss += hinge_loss_single(feature_matrix[i,:], labels[i], theta, theta_0)
    
    return total_loss/feature_nums



def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """

    if label * (np.dot(feature_vector, current_theta) + current_theta_0) <= 0:
        current_theta += label * feature_vector
        current_theta_0 += label
    
    return (current_theta, current_theta_0)


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """

    # initializes starting values
    theta = np.zeros(feature_matrix.shape[1]); theta_not = 0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            result = perceptron_single_step_update(feature_matrix[i,:], labels[i], theta, theta_not)
            theta, theta_not = result
            
    return (theta, theta_not)


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # intialize starting values
    theta = np.zeros(feature_matrix.shape[1]); theta_not = 0

    # keeps sum of returned perceptron steps
    theta_tracker = np.zeros(feature_matrix.shape[1]); theta_not_tracker = 0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            result = perceptron_single_step_update(feature_matrix[i,:], labels[i], theta, theta_not)
            theta, theta_not = result
            theta_tracker += theta; theta_not_tracker += theta_not

    # takes the average perceptron
    theta_tracker /= (feature_matrix.shape[0] * T); theta_not_tracker /= (feature_matrix.shape[0] * T)
            
    return (theta_tracker, theta_not_tracker)


def pegasos_single_step_update(feature_vector, label, L, eta, current_theta, current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    
    if label * (np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
        current_theta = (1 - eta*L)*current_theta + (eta*label)*feature_vector
        current_theta_0 = eta*label + current_theta_0
    else:
        current_theta = (1 - eta*L)*current_theta
        current_theta_0 = current_theta_0

    return (current_theta, current_theta_0)


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """

    # initialize starting values
    theta = np.zeros(feature_matrix.shape[1]); theta_not = 0
    s = 1 # records number of steps
    for t in range(1,T+1):
        for i in get_order(feature_matrix.shape[0]):
            eta = 1/(s ** 0.5) # update learning rate
            result = pegasos_single_step_update(feature_matrix[i,:], labels[i], L, eta, theta, theta_not)
            theta, theta_not = result
            s += 1

    return (theta, theta_not)


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    retval = np.zeros(feature_matrix.shape[0]);
    for i in range(feature_matrix.shape[0]):
        if np.dot(feature_matrix[i,:], theta) + theta_0 > 0:
            retval[i] = 1
        else:
            retval[i] = -1

    return retval


def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    train_val = classifier(train_feature_matrix, train_labels, **kwargs)
    retval_train = classify(train_feature_matrix, train_val[0], train_val[1])
    retval_val = classify(val_feature_matrix, train_val[0], train_val[1])
    retval = (accuracy(retval_train, train_labels), accuracy(retval_val, val_labels))
    return retval



def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()


def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input
    """
    sw = open("stopwords.txt", "r")
    l = sw.read().split("\n")
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in l:
                dictionary[word] = len(dictionary)
    return dictionary


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    """
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] += 1
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
