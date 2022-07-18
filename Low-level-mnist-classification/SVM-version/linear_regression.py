import numpy as np

def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization
    """
    
    N, M = X.shape
    identity = lambda_factor * np.identity(M)
    XTY = np.dot(np.transpose(X), Y)
    XTX = np.dot(np.transpose(X), X)
    return np.dot(np.linalg.inv(XTX+identity), XTY)

def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
