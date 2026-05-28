


"""
Principal Component Analysis (PCA) implementation for TEC data. 
This module includes both an incremental PCA implementation using scikit-learn's IncrementalPCA, as well as a personal 
implementation of PCA using numpy. 

The incremental PCA is designed to handle large datasets that may not fit into memory, 
while the personal implementation is a more traditional approach to PCA.

"""
import numpy as np
from incremental_pca_torch import IncrementalPCA

#----------------------incremental implementation----------------------
def weight_by_latitude(tec: np.ndarray) -> np.ndarray:
    """Weight TEC values by sqrt(latitude).
    Each TEC value is multiplied by sqrt(lat_index) for the corresponding latitude.
    """
    nlat = tec.shape[0]
    nlon = tec.shape[1]
    ntime = tec.shape[2]
    
    # Create latitude weight array: sqrt(0), sqrt(1), sqrt(2), ..., sqrt(nlat-1)
    # lats = 90 - np.arange(nlat)
    lats = np.arange(0, nlat, 1)
    lat_weights = np.sqrt(lats)
    # Create weight tensor with same shape as tec
    weight_tensor = np.tile(lat_weights[:, np.newaxis, np.newaxis], (1, nlon, ntime))
    # Apply weights
    weighted_tec = tec * weight_tensor
    return weighted_tec


def find_principal_components(tec: np.ndarray, number_of_components: int)->np.ndarray:
    """Find the principal components of the TEC data using Incremental PCA from scikit-learn.
    The TEC data is reshaped to a 2D array of shape (lat*lon, time) before applying PCA, and the 
    resulting components are reshaped back to the original image format."""
    
    # save shape for later
    original_shape = tec.shape
    
    # weight by latitude
    tec = weight_by_latitude(tec)

    # reshape to 2D array of shape (time, lat*lon)
    tec_columns = tec.reshape((-1, tec.shape[2])) # (lat*lon, time)
    tec_columns = tec_columns.T                   # (time, lat*lon)
 
    # initialize PCA
    ipca = IncrementalPCA(
        n_components=number_of_components, 
        batch_size=256, 
        device='cpu'  # Use 'cpu' if no GPU available
    )
    # fit on data
    ipca.fit(tec_columns)

    # Get principal components
    components = ipca.components_  # shape: (n_components, n_features)
    # transpose for compatibility
    components = components.T
    # convert to images
    components_images = components.reshape((original_shape[0], original_shape[1], number_of_components))

    # return both images and columns for futher analysis
    return components_images, components


def subtract_mean(tec:np.ndarray)->np.ndarray:
    mean = np.mean(tec, axis=2)

    for i in range(tec.shape[2]):
        tec[:, :, i] -=  mean
    return tec


def check_orthonomality(principal_components:np.ndarray)->None:
    print("Checking othonomality")

    # check shape of components. Reshape into columns if needed
    if len(principal_components.shape) == 3:
        principal_components_columns = np.reshape(
            principal_components, 
            (
                principal_components.shape[0]*principal_components.shape[1], 
                principal_components.shape[2]
            )
        )

    for i in range(principal_components_columns.shape[1]):
        string = f""
        for j in range(principal_components_columns.shape[1]):
            pc1 = principal_components_columns[:, i]
            pc2 = principal_components_columns[:, j]
            dp = np.dot(pc1, pc2)
            string += f"{dp:>10.2}"
        print(string)
        

def compute_time_coefficients(principal_components:np.ndarray, tec:np.ndarray)->np.ndarray:

    # principal_components = weight_by_latitude(principal_components)
    tec = weight_by_latitude(tec)
    # check shape of components. Reshape into columns if needed
    if len(principal_components.shape) == 3:
        principal_components = np.reshape(
            principal_components, 
            (
                principal_components.shape[0]*principal_components.shape[1], 
                principal_components.shape[2]
            )
        )

    # reshape images into columns
    tec = np.reshape(
        tec,
        (
            tec.shape[0]*tec.shape[1],
            tec.shape[2]
        )
    )
    coefficients = principal_components.T @ tec
    # coefficients = np.dot(principal_components.T, tec)
    return coefficients        



#----------------------personal implementation----------------------
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

