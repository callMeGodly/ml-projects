import numpy as np

def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.
    """
    final = ((np.dot(X, Y.T)) + c) ** p
    return final;



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.
    """
    n, d = X.shape
    test1 = np.zeros(7)
    np.reshape(test1, (1,7))
    print(test1.shape)
    m = Y.shape[0]
    kernel_matrix = np.zeros((n,m))
    for i in range(n):
        d = X[i,:] - Y
        b = np.sum(d**2, axis=1)
        kernel_matrix[i,:] = np.exp(-gamma*b)
    
    return kernel_matrix

