
import numpy as np
from numpy import array, zeros, roll, cos, sin
from scipy.optimize import curve_fit


#----------------------centering----------------------
def center_midday(tec: np.ndarray) -> np.ndarray:
    """Shift the TEC images uniformly, only according to the time of day"""
    rolled_tec = zeros(tec.shape)
    for temporal_index in range(tec.shape[2]):
        number_of_indicies_to_roll = int(tec.shape[1]/288 * temporal_index)
        if number_of_indicies_to_roll == 360: number_of_indicies_to_roll = 0
        rolled_tec[:, :, temporal_index] = roll(tec[:, :, temporal_index], number_of_indicies_to_roll, axis=1)
    rolled_tec = roll(rolled_tec, 180, axis=1) # try to get peak in center
    return rolled_tec

def center_midnight(tec: np.ndarray) -> np.ndarray:
    """Shift the TEC images uniformly, only according to the time of day"""
    rolled_tec = zeros(tec.shape)
    for temporal_index in range(tec.shape[2]):
        number_of_indicies_to_roll = int(tec.shape[1]/288 * temporal_index)
        if number_of_indicies_to_roll == 360: number_of_indicies_to_roll = 0
        rolled_tec[:, :, temporal_index] = roll(tec[:, :, temporal_index], number_of_indicies_to_roll, axis=1)
    # rolled_tec = roll(rolled_tec, 180, axis=1) # try to get peak in center
    return rolled_tec



def _gaussian_2d(coords: tuple, amp: float, x0: float, y0: float, sigma_x: float, sigma_y: float, theta: float, offset: float) -> array:
    """2D Gaussian function with rotation.
    Used to fit the TEC images and find the center of the peak. The function is defined as:
    G(x, y) = offset + amp * exp(-((x_rot^2 / (2 * sigma_x^2)) + (y_rot^2 / (2 * sigma_y^2))))
    where (x_rot, y_rot) are the coordinates rotated by angle theta around the center (x0, y0).

    coords: tuple of (x, y) coordinate arrays"""
    x, y = coords
    cos_t = cos(theta)
    sin_t = sin(theta)

    x_shift = x - x0
    y_shift = y - y0

    x_rot = x_shift * cos_t + y_shift * sin_t
    y_rot = -x_shift * sin_t + y_shift * cos_t

    exp_term = (x_rot ** 2) / (2 * sigma_x ** 2) + (y_rot ** 2) / (2 * sigma_y ** 2)
    return offset + amp * np.exp(-exp_term)

def fit_gaussian_2d_slice(tec_slice: np.ndarray) -> dict:
    """Fit a rotated 2D Gaussian to a single TEC slice (2D array).
    Returns fitted parameters as a dict.
    """
    data = np.asarray(tec_slice)
    if data.ndim != 2:
        raise ValueError("tec_slice must be a 2D array")

    ny, nx = data.shape
    x = np.arange(nx)
    y = np.arange(ny)
    xv, yv = np.meshgrid(x, y)

    mask = np.isfinite(data)
    x_data = xv[mask]
    y_data = yv[mask]
    z_data = data[mask]

    if z_data.size == 0:
        raise ValueError("tec_slice has no finite values")

    offset0 = np.nanmin(z_data)
    amp0 = np.nanmax(z_data) - offset0
    y0_0, x0_0 = np.unravel_index(np.nanargmax(data), data.shape)
    sigma_x0 = max(1.0, nx / 6.0)
    sigma_y0 = max(1.0, ny / 6.0)
    theta0 = 0.0

    p0 = (amp0, x0_0, y0_0, sigma_x0, sigma_y0, theta0, offset0)
    bounds = (
        (0.0, 0.0, 0.0, 0.1, 0.1, -np.pi / 2, -np.inf),
        (np.inf, nx - 1.0, ny - 1.0, np.inf, np.inf, np.pi / 2, np.inf),
    )

    popt, _ = curve_fit(
        _gaussian_2d,
        (x_data, y_data),
        z_data,
        p0=p0,
        bounds=bounds,
        maxfev=1000,
    )

    return {
        "amp": float(popt[0]),
        "x0": float(popt[1]),
        "y0": float(popt[2]),
        "sigma_x": float(popt[3]),
        "sigma_y": float(popt[4]),
        "theta": float(popt[5]),
        "offset": float(popt[6]),
    }

def get_peaks(tec: np.ndarray) -> np.ndarray:
    """Find the peak of the TEC images by fitting a rotated 2D 
    Gaussian to each slice of the TEC data and extracting the 
    x0 parameter as the peak index."""
    peak_indices = []
    for t in range(tec.shape[2]):
        print(f"{t} / {tec.shape[2]}        ", end="\r")
        image = tec[:, :, t]
        try:
            popt = fit_gaussian_2d_slice(image)
            peak_index = popt["x0"]
        except RuntimeError:
            # if fit unsuccessful, use previous fit
            peak_index = popt["x0"]

        peak_indices.append(peak_index)

    peak_indices = array(peak_indices)
    return peak_indices

def center_concomic(tec: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Shift the TEC images according to the center of the peak, as 
    found by fitting a rotated 2D Gaussian to each slice of the TEC data."""
    # roll tec images according to the magnitude of the center
    rolled_tec = zeros(tec.shape)
    for t in range(tec.shape[2]):
        image = tec[:, :, t]
        rolled_tec[:, :, t] = roll(image, -centers[t], axis=1)
    
    # center peaks
    rolled_tec = roll(rolled_tec, 180, axis=1)
    return rolled_tec

