


"""
SWARM loading, plotting as well as some handling functions such as finding the closest approach to Tromsø, and plotting the trajectory near Tromsø.
"""
import numpy as np
import matplotlib.pyplot as plt

from numpy import array, linspace, sqrt
from matplotlib.pyplot import subplots, show 

import datetime

def load_single(swarm_path=None, time_format="unix"):
    if swarm_path == None:
        swarm_path = r"C:\Users\NoahH\OneDrive - UiT Office 365\Personal OneDrive copy\Tromsø\UiT Arctic University of Norway\Techniques for Investigating the Near-Earth Space Environment\HEIMDALL\data_gathering\src\data\SWARM\SW_OPER_FACATMS_2F+SW_FAST_FACATMS_2F_SW_OPER_EFIA_LP_1B_20260202T170000_20260202T235959_Filtered(1).csv"
    doc = open(swarm_path)
    data = dict()
    for i, line in enumerate(doc):
        if i == 0:
            keys = line.split(",")
            for key in keys:
                data[key] = []
        else:
            for j, key in enumerate(keys):
                try:
                    data[key].append(float(line.split(",")[j]))
                except ValueError:
                    data[key].append(line.split(",")[j])

    for key in keys:
        data[key] = array(data[key])

    # print(f"{data['Timestamp'] = }")
    if time_format=="unix":
        data["time"] = array([datetime.datetime.fromisoformat(TS).timestamp() for TS in data["Timestamp"]])
    return data

def load_multiple(time_format):
    # load data naively, using the single loader
    path = r"C:\Users\NoahH\OneDrive - UiT Office 365\Personal OneDrive copy\Tromsø\UiT Arctic University of Norway\Techniques for Investigating the Near-Earth Space Environment\HEIMDALL\data_gathering\src\data\SWARM\SW_OPER_FACATMS_2F+SW_FAST_FACATMS_2F_SW_OPER_EFIA_LP_1B_SW_OPER_FACBTMS_2F+SW_FAST_FACBTMS_2F_SW_OPER_EFIB_LP_1B_SW_OPER_FACCTMS_2F+SW_FAST_FACCTMS_2F_0ad83a882b22e6bb928a8e2966fe4d00_20260202T170000_20260202T235959_Filt.csv"
    data = load_single(path, time_format="unix")

    # construct new dict with keys being the spacecraft entries
    new_data = dict()
    for SC in set(data["Spacecraft"]):
        new_data[SC] = dict()
    
    # initialize lists in the new dictionary which matches the old dictionary lists, except for the spacecraft entry
    for key in data.keys():
        if key == "Spacecraft":
            continue
        else:
            for new_key in new_data.keys():
                new_data[new_key][key] = []

    # fill the lists with the data from the old dataset, and store it appropiately with respect to spacecraft
    for i in range(len(data["Spacecraft"])):
        SC = data["Spacecraft"][i]
        for key in data.keys():
            if key == "Spacecraft":
                continue
                           
            new_data[SC][key].append(data[key][i])
    
    # make everything an array
    for SC in new_data.keys():
        for key in new_data[SC].keys():
           new_data[SC][key] = array(new_data[SC][key])

    return new_data



def get_index_close(data):
    TRO_lat = 69.6492
    TRO_lon = 18.9553

    # data = load_single(time_format="unix")
    lat, lon, time = data["Latitude"], data["Longitude"], data["time"]
    # print(f"{data.keys() = }")

    lat = array([np.float64(e) for e in lat])
    lon = array([np.float64(e) for e in lon])
    distance_from_TRO = sqrt((lat-TRO_lat)**2 + (lon-TRO_lon)**2)

    closest_index = np.where(distance_from_TRO == min(distance_from_TRO) )
    # print(f"{closest_index = }")
    closest_index = closest_index[0][0]
    
    # print(f"{lat.shape = }")
    # index_width = 250
    return closest_index

def get_TRO_lon_index():
    '''Not used at the moment'''
    # TRO_lat = 69.6492
    TRO_lon = 18.9553

    data = load_single(time_format="unix")
    lat, lon, time = data["Latitude"], data["Longitude"], data["time"]
    # print(f"{data.keys() = }")

    # lat = array([np.float64(e) for e in lat])
    lon = array([np.float64(e) for e in lon])
    distance_from_TRO_lon = sqrt((lon-TRO_lon)**2)

    TRO_lon_index = np.where(distance_from_TRO_lon == min(distance_from_TRO_lon) )
    # print(f"{TRO_lon_index = }")
    TRO_lon_index = TRO_lon_index[0][0]
    
    # print(f"{lat.shape = }")
    # index_width = 250
    return TRO_lon_index


def plot_location(fig, ax, closest_index, index_width):
    TRO_lat = 69.6492
    TRO_lon = 18.9553

    data = load_single(time_format="unix")
    lat, lon, time = data["Latitude"], data["Longitude"], data["time"]
    # print(f"{data.keys() = }")

    lat = array([np.float64(e) for e in lat])
    lon = array([np.float64(e) for e in lon])

    
    # distance_from_TRO = sqrt((lat-TRO_lat)**2 + (lon-TRO_lon)**2)
    # closest_index = np.where(distance_from_TRO == min(distance_from_TRO) )
    # print(f"{closest_index = }")
    # closest_index = closest_index[0][0]
    
    # print(f"{lat.shape = }")
    # fig, ax=subplots()
    # index_width = 250
    ax.plot(
        lon[closest_index-index_width[0]:closest_index+index_width[1]],
        lat[closest_index-index_width[0]:closest_index+index_width[1]],
        label="SWARM ALPHA"
    )
    ax.scatter(
        lon[closest_index],
        lat[closest_index],
        color="forestgreen",
        label="Shortest distance"
    )
    lon_limited = lon[closest_index-index_width[0]: closest_index+index_width[1]]
    dist_lon = sqrt((lon_limited-TRO_lon)**2)
    # print(f"f{dist_lon = }")
    TRO_lon_index = np.where(dist_lon==min(dist_lon))[0][0]
    # print(f"{TRO_lon_index = }")
    # np.where(lat[TRO_lon_index][closest_index-index_width[0]: closest_index+index_width[1]])
    # ax.scatter(
    #     lon[TRO_lon_index],
    #     lat[TRO_lon_index],
    #     color="purple"

    # )
    closest     = datetime.datetime.fromtimestamp(time[closest_index]            )
    width_start = datetime.datetime.fromtimestamp(time[closest_index-index_width[0]])
    width_end   = datetime.datetime.fromtimestamp(time[closest_index+index_width[1]])
    ax.text(lon[closest_index-index_width[0]], lat[closest_index-index_width[0]], f"{width_start.hour}:{width_start.minute}")
    ax.text(lon[closest_index],             lat[closest_index],             f"{closest.hour}:{closest.minute}")
    ax.text(lon[closest_index+index_width[1]], lat[closest_index+index_width[1]], f"{width_end.hour}:{width_end.minute}")

    ax.scatter(TRO_lon, TRO_lat, label="Tromsø", color="purple")
    ax.legend()
    # ax.set_aspect("equal")


    ax.set_title(f"SWARM A Tromsø pass\n{datetime.datetime.fromtimestamp(time[closest_index])}")
    # ax.set_title("SWARM A Tromsø pass")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    # fig.tight_layout()    
    # fig.savefig("figures\\SWARM\\swarm position.png")


def plot_multiple_trajectories():
    TRO_lat = 69.6492
    TRO_lon = 18.9553

    data = load_single(time_format="unix")
    lat, lon, time = data["Latitude"], data["Longitude"], data["time"]
    # print(f"{data.keys() = }")

    lat = array([np.float64(e) for e in lat])
    lon = array([np.float64(e) for e in lon])
    distance_from_TRO = sqrt((lat-TRO_lat)**2 + (lon-TRO_lon)**2)

    threshold = 25 # degree
    close_indicies = np.where(distance_from_TRO < min(distance_from_TRO) + threshold )
    
    fig, ax=subplots()
    ax.scatter(
        lon[close_indicies],
        lat[close_indicies],
        label="SWARM ALPHA",
        s=1,
    )
    ax.scatter(TRO_lon, TRO_lat, label="Tromsø")

    # draw circle
    thetas = np.linspace(0, 2*np.pi, 1000)
    radius = threshold + min(distance_from_TRO)
    centerx = TRO_lon
    centery = TRO_lat
    xs = radius*np.cos(thetas) + centerx
    ys = radius*np.sin(thetas) + centery
    ax.plot(xs, ys, c="firebrick", label="Search field")

    ax.legend()
    ax.set_aspect("equal")
    ax.set_title(f"Search radius = {radius:.4} degrees")

    # ax.set_title(f"{datetime.datetime.fromtimestamp(time[closest_index])}")
    fig.suptitle("SWARM A trajectory near Tromsø")
    fig.supxlabel("Longitude")
    fig.supylabel("Latitude")
    fig.tight_layout()    

    # show()

    fig.savefig("figures\\SWARM\\swarm closest trajectories.png")


# plot_location()
# plot_multiple_trajectories()

def dataplot(data, sat, index_width):
    data = data[sat]
    # data = load_multiple(time_format="unix")[sat]
    index_close = get_index_close(data)
    # TRO_lon_index = get_TRO_lon_index()

    # print(f"{data.keys() = }")
    var_keys = [
        "N_ion",
        "N_elec",
        "T_elec",
        "FAC",
        "Vs"
    ]
    ts = data["time"][index_close-index_width[0]: index_close+index_width[1]]
    fig, axs=subplots(len(var_keys)//2+1, 2, figsize=(10,10))
    for ax, key in zip(axs.flatten(), var_keys):
        values = data[key][index_close-index_width[0]: index_close+index_width[1]]
        # print(f"{values.shape = }")
        # print(f"{type(values[0]) = }")
        ax.plot(
            ts,
            values)
        ax.set_ylabel(key)
        DT_start = datetime.datetime.fromtimestamp(ts[0])
        DT_end   = datetime.datetime.fromtimestamp(ts[-1])
        lon_start   = data["Longitude"]
        lon_end     = data["Longitude"]

        ax.text(ts[0],  np.min(values),  f"{DT_start.hour}:{DT_start.minute}")
        ax.text(ts[-1], np.min(values), f"{DT_end.hour}:{DT_end.minute}")

        ax.vlines(data["time"][index_close], min(values), max(values), color="forestgreen", label="Shortest distance")
        ax.legend()

    plot_location(fig, axs[2,1], closest_index=index_close, index_width=index_width)
    fig.supxlabel("Time")
    DT_mid = datetime.datetime.fromtimestamp(data["time"][index_close])
    fig.suptitle(
        f"SWARM {sat} Tromsø overpass at" + "\n" + 
        f"{DT_mid.hour}:{DT_mid.minute}:{DT_mid.second}"
    )
    fig.tight_layout()
    fig.savefig(f"figures\\SWARM\\some data {sat}.png")




# SWA, SWB, SWC, SWAC = load_multiple(time_format="unix")
# path = r"C:\Users\NoahH\OneDrive - UiT Office 365\Personal OneDrive copy\Tromsø\UiT Arctic University of Norway\Techniques for Investigating the Near-Earth Space Environment\HEIMDALL\data_gathering\src\data\SWARM\SW_OPER_FACATMS_2F+SW_FAST_FACATMS_2F_SW_OPER_EFIA_LP_1B_SW_OPER_FACBTMS_2F+SW_FAST_FACBTMS_2F_SW_OPER_EFIB_LP_1B_SW_OPER_FACCTMS_2F+SW_FAST_FACCTMS_2F_0ad83a882b22e6bb928a8e2966fe4d00_20260202T170000_20260202T235959_Filt.csv"
# data = load_single(path, time_format="unix")
# print(f"{len(data["dN_ion"]) = }")
# new_data = load_multiple(time_format="unix")
# print(f"{len(new_data["A"]["dN_ion"]) = }")
data = load_multiple(time_format="unix")
dataplot(data, "A", index_width = (100, 100))
dataplot(data, "B", index_width = (100, 100))
dataplot(data, "C", index_width = (100, 100))
dataplot(data, "-", index_width = (100, 100))

