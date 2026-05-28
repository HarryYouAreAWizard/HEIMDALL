


"""
EISCAT data loading and plotting

"""

import numpy as np
import matplotlib.pyplot as plt
import datetime

from numpy import array, linspace
from matplotlib.pyplot import subplots, show 
from scipy.io import loadmat

def plot_variable(fig, ax, variables, name, title, crange):
    T, H, variable = variables
    cmin, cmax = crange

    image = ax.pcolormesh(
        T,
        H,
        variable,
        cmap="plasma",
        shading="auto",
        vmin=cmin, vmax=cmax,
    )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(name, fontsize=15)
    # cbar.ax.tick_params(labelsize=12)
    # ax.set_xlabel("Time since start of experiment [min]", fontsize=15)
    # ax.set_ylabel("Altitude [km]", fontsize=15)
    # ax.set_title(title, fontsize=20)



def make_nice_figure(data, suptitle):
    electron_temperature = data["Te"]
    electron_density = data["ne"]
    ion_temperature = data["Ti"]
    ion_velocity = data["vi"] # probably
    h = data["h"] # s
    time = data["t"]

    start_time = time[0][0]
    ts = (time[0]-start_time)/60 # min
    hs = h[:, 0]
    T, H = np.meshgrid(ts, hs)

    fig, axs=subplots(4, 1, figsize=(8,10), sharex=False)
    fig.suptitle(suptitle, fontsize=25)
    plot_variable(fig, axs[0], (T, H, electron_density), r"$n_e$ [$\text{m}^{-3}$]", "Electron Density", (1e10,1e12))
    plot_variable(fig, axs[1], (T, H, electron_temperature), r"$T_e$ [K]", "Electron Temperature", (0, 4000))
    plot_variable(fig, axs[2], (T, H, ion_temperature), r"$T_i$ [K]", "Ion Temperature", (0, 3000))
    plot_variable(fig, axs[3], (T, H, ion_velocity), r"$v_i$ [$\frac{\text{m}}{\text{s}}$]", "Ion Velocity", (-200, 200))
    fig.supxlabel("Time since start of experiment [min]", fontsize=20)
    fig.supylabel("Altitude [km]", fontsize=20)

    fig.tight_layout()
    fig.savefig(r"figures\\EISCAT\\" + suptitle + ".png")

# beata_path = r"data\beata_20260202.mat"
# beata_data = loadmat(beata_path)
# make_nice_figure(beata_data, "Beata (UHF)")
# bella_path = r"data\bella_20260202.mat"
# bella_data = loadmat(bella_path)
# make_nice_figure(bella_data, "Bella (VHF)")

def load(path, time_format, slide_date=False):
    data = loadmat(path)

    electron_temperature = data["Te"]
    electron_density = data["ne"]
    ion_temperature = data["Ti"]
    ion_velocity = data["vi"] # probably
    h = data["h"]
    time = data["t"]

    if time_format=="unix":
        time = time[0]
        pass

    elif time_format=="datetime":
        timestamps = []
        for t in time:
            DT = datetime.datetime.fromtimestamp(int(t))

            timestamps.append(DT)

        time=timestamps
    else:
        print(f"Time format must be 'unix' or 'datetime'")

    # start_time = time[0][0]
    # ts = (time[0]-start_time)/60 # min
    # hs = h[:, 0]
    # T, H = np.meshgrid(ts, hs)    

    if slide_date:
        time_slided = []
        for t in time:
            # print(f"{t = }")
            DT = datetime.datetime.fromtimestamp(int(t))
            # DT.day -= 1
            DT_slided = datetime.datetime(
                DT.year,
                DT.month,
                DT.day-1,
                DT.hour,
                DT.minute,
                DT.second
            )
            t_slided = DT_slided.timestamp()
            time_slided.append(t_slided)
        time = time_slided

    return {
        "time":time,
        "altitude":h,
        "electron temperature":electron_temperature,
        "electron density":electron_density,
        "ion temperature":ion_temperature,
        "ion velocity":ion_velocity
    }
