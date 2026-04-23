

import numpy as np

def correlation_matrix(dataset):
    n = len(dataset)

    S = sum([np.outer(x, x.T) for x in dataset])

    return S/n
    # return sum([x @ x.T for x in dataset])


def covariance_matrix(dataset):

    R = correlation_matrix(dataset)

    return R - np.outer(np.mean((dataset)), np.mean(dataset).T)


def average(dataset):

    temporary = np.zeros(len(dataset[0]))
    for i in range(len(temporary)):
        for datapoint in dataset:
            temporary[i] += datapoint[i]

    n = len(dataset)

    mu = np.array(temporary)/n

    return mu


def PCA(X, subtract_average=False):
    # We assume mu_x = 0, so we subtract the mean from each dimension
    
    if subtract_average:
        mu = average(X)
        X = X - mu

    # covariance matrix 
    Sigma_x = covariance_matrix(X)

    # Lambda matrix
    eigenresult = np.linalg.eig(Sigma_x)
    eigenvalues = eigenresult[0]
    eigenvectors = eigenresult[1]
    #
    eigenvalues, eigenvectors = zip(*sorted(zip(eigenvalues, eigenvectors), key=lambda x: x[0]))
    eigenvalues = np.array(eigenvalues[::-1])
    eigenvectors = np.array(eigenvectors[::-1])
    Z = np.array([eigenvectors @ x for x in X])
    return eigenvalues, eigenvectors, Z