from __future__ import annotations

import multiprocessing as mp


"""
Principal Component Analysis (PCA) implementation for TEC data. 
This module includes both an incremental PCA implementation using scikit-learn's IncrementalPCA, as well as a personal 
implementation of PCA using numpy. 

The incremental PCA is designed to handle large datasets that may not fit into memory, 
while the personal implementation is a more traditional approach to PCA.

"""
import numpy as np
try:
    from incremental_pca_torch import IncrementalPCA
except ImportError:
    IncrementalPCA = None

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


def latitude_weights(
    nlat: int,
    lat_min: float | None = None,
    lat_max: float | None = None,
) -> np.ndarray:
    """Return latitude weights used by the TEC PCA routines.

    If latitude limits are supplied, use an area-style sqrt(cos(latitude))
    weighting.  Without latitude limits, keep the original row-index weighting
    for backwards compatibility with the existing northern-cap products.
    """
    if lat_min is None or lat_max is None:
        return np.sqrt(np.arange(0, nlat, 1))
    latitudes = np.linspace(lat_min, lat_max, nlat)
    return np.sqrt(np.clip(np.cos(np.deg2rad(latitudes)), 0.0, None))


def weighted_tec_columns(
    tec: np.ndarray,
    lat_min: float | None = None,
    lat_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Return latitude-weighted TEC columns and an observation mask.

    The returned data matrix has shape ``(lat*lon, time)``.  Missing values are
    replaced by zero only after the boolean observation mask has been built, so
    they can be assigned zero weight by sparse PCA solvers instead of being
    treated as real zero-valued TEC observations.
    """
    original_shape = tec.shape
    weights = latitude_weights(original_shape[0], lat_min=lat_min, lat_max=lat_max)[:, np.newaxis, np.newaxis]
    observed = np.isfinite(tec)
    weighted = np.where(observed, tec, 0.0) * weights
    return (
        weighted.reshape((-1, original_shape[2])),
        observed.reshape((-1, original_shape[2])),
        original_shape,
    )


def sparse_feature_mean(columns: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Mean of each spatial feature using only observed samples."""
    counts = observed.sum(axis=1)
    sums = np.sum(columns, axis=1)
    mean = np.zeros(columns.shape[0], dtype=float)
    valid = counts > 0
    mean[valid] = sums[valid] / counts[valid]
    return mean


def _sample_observed_columns(observed: np.ndarray, max_columns: int, random_state: int) -> np.ndarray:
    coverage = observed.sum(axis=0)
    usable = np.flatnonzero(coverage > 0)
    if usable.size <= max_columns:
        return usable
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(usable, size=max_columns, replace=False))


def _initial_sparse_components(
    residual_columns: np.ndarray,
    observed: np.ndarray,
    n_components: int,
    sample_columns: int,
    random_state: int,
) -> np.ndarray:
    """Initialize components from a sampled zero-filled SVD."""
    sample = _sample_observed_columns(observed, sample_columns, random_state)
    if sample.size == 0:
        raise ValueError("sparse PCA needs at least one observed TEC sample")
    sampled = residual_columns[:, sample].copy()
    sampled[~observed[:, sample]] = 0.0
    u, _, _ = np.linalg.svd(sampled, full_matrices=False)
    components = u[:, :n_components]
    if components.shape[1] < n_components:
        rng = np.random.default_rng(random_state)
        extra = rng.normal(size=(residual_columns.shape[0], n_components - components.shape[1]))
        components = np.hstack([components, extra])
    components, _ = np.linalg.qr(components)
    return components[:, :n_components]


def _solve_time_coefficients(
    components: np.ndarray,
    residual_columns: np.ndarray,
    observed: np.ndarray,
    ridge: float,
) -> np.ndarray:
    n_components = components.shape[1]
    coefficients = np.zeros((n_components, residual_columns.shape[1]), dtype=float)
    ridge_matrix = ridge * np.eye(n_components)
    for time_idx in range(residual_columns.shape[1]):
        mask = observed[:, time_idx]
        if np.count_nonzero(mask) < n_components:
            continue
        a = components[mask, :]
        b = residual_columns[mask, time_idx]
        coefficients[:, time_idx] = np.linalg.solve(a.T @ a + ridge_matrix, a.T @ b)
    return coefficients


_TIME_WORKER_COMPONENTS = None
_TIME_WORKER_RESIDUAL_COLUMNS = None
_TIME_WORKER_OBSERVED = None
_TIME_WORKER_RIDGE = None


def _init_time_worker(
    components: np.ndarray,
    residual_columns: np.ndarray,
    observed: np.ndarray,
    ridge: float,
) -> None:
    global _TIME_WORKER_COMPONENTS
    global _TIME_WORKER_RESIDUAL_COLUMNS
    global _TIME_WORKER_OBSERVED
    global _TIME_WORKER_RIDGE
    _TIME_WORKER_COMPONENTS = components
    _TIME_WORKER_RESIDUAL_COLUMNS = residual_columns
    _TIME_WORKER_OBSERVED = observed
    _TIME_WORKER_RIDGE = ridge


def _chunk_ranges(n_items: int, n_chunks: int) -> list[tuple[int, int]]:
    n_chunks = max(1, min(n_chunks, n_items))
    edges = np.linspace(0, n_items, n_chunks + 1, dtype=int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_chunks) if edges[i] < edges[i + 1]]


def _solve_time_coefficients_chunk(item: tuple[int, int]) -> tuple[int, int, np.ndarray]:
    start, stop = item
    coeffs = _solve_time_coefficients(
        _TIME_WORKER_COMPONENTS,
        _TIME_WORKER_RESIDUAL_COLUMNS[:, start:stop],
        _TIME_WORKER_OBSERVED[:, start:stop],
        _TIME_WORKER_RIDGE,
    )
    return start, stop, coeffs


def _solve_time_coefficients_parallel(
    components: np.ndarray,
    residual_columns: np.ndarray,
    observed: np.ndarray,
    ridge: float,
    n_jobs: int,
) -> np.ndarray:
    if n_jobs <= 1:
        return _solve_time_coefficients(components, residual_columns, observed, ridge)
    coefficients = np.zeros((components.shape[1], residual_columns.shape[1]), dtype=float)
    ranges = _chunk_ranges(residual_columns.shape[1], n_jobs)
    with mp.get_context("fork").Pool(
        processes=len(ranges),
        initializer=_init_time_worker,
        initargs=(components, residual_columns, observed, ridge),
    ) as pool:
        for start, stop, chunk in pool.imap_unordered(_solve_time_coefficients_chunk, ranges):
            coefficients[:, start:stop] = chunk
    return coefficients


def _solve_component_rows(
    coefficients: np.ndarray,
    residual_columns: np.ndarray,
    observed: np.ndarray,
    ridge: float,
) -> np.ndarray:
    n_components = coefficients.shape[0]
    components = np.zeros((residual_columns.shape[0], n_components), dtype=float)
    ridge_matrix = ridge * np.eye(n_components)
    for feature_idx in range(residual_columns.shape[0]):
        mask = observed[feature_idx, :]
        if np.count_nonzero(mask) < n_components:
            continue
        a = coefficients[:, mask].T
        b = residual_columns[feature_idx, mask]
        components[feature_idx, :] = np.linalg.solve(a.T @ a + ridge_matrix, a.T @ b)
    return components


def _solve_components(
    coefficients: np.ndarray,
    residual_columns: np.ndarray,
    observed: np.ndarray,
    ridge: float,
) -> np.ndarray:
    n_components = coefficients.shape[0]
    components = _solve_component_rows(coefficients, residual_columns, observed, ridge)
    components, r = np.linalg.qr(components)
    return components[:, :n_components], r[:n_components, :n_components]


_COMPONENT_WORKER_COEFFICIENTS = None
_COMPONENT_WORKER_RESIDUAL_COLUMNS = None
_COMPONENT_WORKER_OBSERVED = None
_COMPONENT_WORKER_RIDGE = None


def _init_component_worker(
    coefficients: np.ndarray,
    residual_columns: np.ndarray,
    observed: np.ndarray,
    ridge: float,
) -> None:
    global _COMPONENT_WORKER_COEFFICIENTS
    global _COMPONENT_WORKER_RESIDUAL_COLUMNS
    global _COMPONENT_WORKER_OBSERVED
    global _COMPONENT_WORKER_RIDGE
    _COMPONENT_WORKER_COEFFICIENTS = coefficients
    _COMPONENT_WORKER_RESIDUAL_COLUMNS = residual_columns
    _COMPONENT_WORKER_OBSERVED = observed
    _COMPONENT_WORKER_RIDGE = ridge


def _solve_components_chunk(item: tuple[int, int]) -> tuple[int, int, np.ndarray]:
    start, stop = item
    components = _solve_component_rows(
        _COMPONENT_WORKER_COEFFICIENTS,
        _COMPONENT_WORKER_RESIDUAL_COLUMNS[start:stop, :],
        _COMPONENT_WORKER_OBSERVED[start:stop, :],
        _COMPONENT_WORKER_RIDGE,
    )
    return start, stop, components


def _solve_components_parallel(
    coefficients: np.ndarray,
    residual_columns: np.ndarray,
    observed: np.ndarray,
    ridge: float,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_jobs <= 1:
        return _solve_components(coefficients, residual_columns, observed, ridge)
    n_components = coefficients.shape[0]
    components = np.zeros((residual_columns.shape[0], n_components), dtype=float)
    ranges = _chunk_ranges(residual_columns.shape[0], n_jobs)
    with mp.get_context("fork").Pool(
        processes=len(ranges),
        initializer=_init_component_worker,
        initargs=(coefficients, residual_columns, observed, ridge),
    ) as pool:
        for start, stop, chunk in pool.imap_unordered(_solve_components_chunk, ranges):
            components[start:stop, :] = chunk
    components, r = np.linalg.qr(components)
    return components[:, :n_components], r[:n_components, :n_components]


def _observed_rmse(
    components: np.ndarray,
    coefficients: np.ndarray,
    residual_columns: np.ndarray,
    observed: np.ndarray,
) -> float:
    sumsq = 0.0
    count = 0
    for time_idx in range(residual_columns.shape[1]):
        mask = observed[:, time_idx]
        if not np.any(mask):
            continue
        residual = residual_columns[mask, time_idx] - components[mask, :] @ coefficients[:, time_idx]
        sumsq += float(residual @ residual)
        count += int(np.count_nonzero(mask))
    return np.sqrt(sumsq / count) if count else np.nan


def find_sparse_principal_components(
    tec: np.ndarray,
    number_of_components: int,
    n_iterations: int = 8,
    ridge: float = 1e-6,
    sample_columns: int = 2048,
    random_state: int = 0,
    n_jobs: int = 1,
    lat_min: float | None = None,
    lat_max: float | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate TEC principal components with missing samples weighted to zero.

    This solves a weighted low-rank approximation problem using alternating
    least squares.  Finite TEC samples get weight one, and missing/non-finite
    samples get weight zero.  This allows the PCA basis to use sparse raw TEC
    maps directly instead of first filling gaps by interpolation.

    Returns ``(component_images, component_columns, time_coefficients, mean)``.
    The component images have shape ``(lat, lon, number_of_components)`` and the
    columns have shape ``(lat*lon, number_of_components)``.
    """
    original_shape = tec.shape
    columns, observed, _ = weighted_tec_columns(tec, lat_min=lat_min, lat_max=lat_max)
    mean = sparse_feature_mean(columns, observed)
    residual_columns = columns - mean[:, np.newaxis]
    residual_columns[~observed] = 0.0

    components = _initial_sparse_components(
        residual_columns,
        observed,
        number_of_components,
        sample_columns,
        random_state,
    )
    coefficients = np.zeros((number_of_components, original_shape[2]), dtype=float)
    last_rmse = np.nan
    n_jobs = max(1, int(n_jobs))
    for iteration in range(n_iterations):
        coefficients = _solve_time_coefficients_parallel(
            components,
            residual_columns,
            observed,
            ridge,
            n_jobs,
        )
        components, transform = _solve_components_parallel(
            coefficients,
            residual_columns,
            observed,
            ridge,
            n_jobs,
        )
        coefficients = transform @ coefficients
        if verbose:
            last_rmse = _observed_rmse(components, coefficients, residual_columns, observed)
            print(f"sparse PCA iteration {iteration + 1}/{n_iterations}: observed RMSE {last_rmse:.6g}")

    component_images = components.reshape((original_shape[0], original_shape[1], number_of_components))
    return component_images, components, coefficients, mean


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
    if IncrementalPCA is None:
        raise ImportError("incremental_pca_torch is required for dense PCA")
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


def compute_sparse_time_coefficients(
    principal_components: np.ndarray,
    tec: np.ndarray,
    mean: np.ndarray | None = None,
    ridge: float = 1e-6,
    n_jobs: int = 1,
    lat_min: float | None = None,
    lat_max: float | None = None,
) -> np.ndarray:
    """Project sparse TEC data onto components using only observed samples."""
    columns, observed, _ = weighted_tec_columns(tec, lat_min=lat_min, lat_max=lat_max)
    if len(principal_components.shape) == 3:
        principal_components = np.reshape(
            principal_components,
            (
                principal_components.shape[0]*principal_components.shape[1],
                principal_components.shape[2],
            ),
        )
    if mean is None:
        mean = sparse_feature_mean(columns, observed)
    residual_columns = columns - mean[:, np.newaxis]
    residual_columns[~observed] = 0.0
    return _solve_time_coefficients_parallel(
        principal_components,
        residual_columns,
        observed,
        ridge,
        max(1, int(n_jobs)),
    )



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
