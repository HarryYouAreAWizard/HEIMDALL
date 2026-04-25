
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, zeros, save, load
from scipy import ndimage
from TEC.TEC import load_naive as load_single

file_seperator = "/"

#----------------------dataset builder----------------------
def load_all_in_folder()->dict:
    # datafolder = r"data"
    datafolder = r"/data/nonie"
    """Load all datasets in the data folder and return them as a dict with filename as key and dataset as value."""
    datafiles = os.listdir(datafolder)
    datasets = dict()
    for i, datafile in enumerate(datafiles):
        path = datafolder + file_seperator + datafile
        try:
            data = load_single(path, time_format="unix")
            datasets[datafile] = data
        except OSError as e:
            print(f"\nFollowing error occured:\n")
            print(e)
            print("\ncontinuing...\n")
            continue
    return datasets


def build_large_dataset(title="non_shifted_tec_", extract_northern=False, small_data=False)->None:
    """Build a large dataset by merging all available datasets in the data folder.
    The resulting dataset is saved as a numpy array in the master_data folder.  """
    # load all available datasets
    datasets = load_all_in_folder()

    if small_data:
        title += "sd_"
    else:
        small_data = len(datasets)

    # TODO shift individual datasets before merging
    # either do it here, or do it afterwards. It is of course nice to do it once and never think about it again

    # pick an example to determine size
    keys = list(datasets.keys())
    data_shape = datasets[keys[0]]["tec"].shape
    print(f"{data_shape = }")

    if extract_northern:
        master_dataset = zeros((30, data_shape[1], data_shape[2]*small_data))
    else:
        master_dataset = zeros((data_shape[0], data_shape[1], data_shape[2]*small_data))
    print(f"{master_dataset.shape = }")


    if extract_northern:
        for i, key in enumerate(datasets.keys()):
            # master_dataset[60+90:, :, i:i+data_shape[2]] = datasets[key]["tec"]
            master_dataset[:, :, i*data_shape[2]:i*data_shape[2]+data_shape[2]] = datasets[key]["tec"][60+90:, :, :]
            if i == small_data - 1: break
        # title += "northern"
        # np.save("..\\master_data\\non_shifted_tec_northern.npy", master_dataset)


    else:
        for i, key in enumerate(datasets.keys()):
            master_dataset[:, :, i*data_shape[2]:i*data_shape[2]+data_shape[2]] = datasets[key]["tec"]
            if i == small_data - 1: break
  
        title += "global"

    title += ".npy"
    print(f"Saving master data at /data/nonie/masterdata/{title}")
    # np.save(title, master_dataset)
    np.save(f"/data/nonie/masterdata/{title}", master_dataset)

#----------------------interpolation----------------------
def interpolate_tec(data:array)->array:
    """
    Spatial interpolation of tec images
    
    param: data: 3D array of shape (lat, lon, time)
    return: 3D array of shape (lat, lon, time) with nans filled by nearest neighbour interpolation
    """
    tec_data = array(data).copy()
    
    # Spatial interpolation using nearest neightbour fill
    for time_idx in range(tec_data.shape[2]):
        # pick image
        frame = tec_data[:, :, time_idx]    
        # find nans
        nan_mask = np.isnan(frame)
        
        if nan_mask.any():
            # Use distance_transform to find nearest non-NaN values
            indices = ndimage.distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
            # for every pixel with nan value, find the nearest pixel with non nan value and set it
            frame[nan_mask] = frame[tuple(indices[:, nan_mask])]
        
        # set data
        tec_data[:, :, time_idx] = frame
    return tec_data




