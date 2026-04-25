

import numpy as np
from numpy import array, linspace
import datetime

import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show


from .src.ACE import load_MFI, load_SWE
from .src.EISCAT import load as load_EISCAT
from .src.TEC import load as load_TEC

from .src.EISCAT import plot_variable


current_directory = __file__[:-(len(__name__) + 3)]

def EISCAT(time_format, slide_date=False):
    path_beata = current_directory + r"data_gathering\src\data\EISCAT\beata_20260202.mat"
    path_bella = current_directory + r"data_gathering\src\data\EISCAT\bella_20260202.mat"
    
    data_beata = load_EISCAT(path_beata, time_format=time_format, slide_date=slide_date)
    data_bella = load_EISCAT(path_bella, time_format=time_format, slide_date=slide_date)
    return data_beata, data_bella

def ACE(time_format):

    MFI_path = current_directory + r"data_gathering\src\data\ACE\AC_K0_MFI_496241.csv"
    SWE_path = current_directory + r"data_gathering\src\data\ACE\AC_K0_SWE_496241.csv"
    # MFI_path = r"C:\Users\NoahH\OneDrive - UiT Office 365\Personal OneDrive copy\Tromsø\UiT Arctic University of Norway\Techniques for Investigating the Near-Earth Space Environment\HEIMDALL\data_gathering\data\ACE\AC_K0_MFI_496241.csv"
    
    data_MFI = load_MFI(MFI_path, time_format=time_format)
    data_SWE = load_SWE(SWE_path, time_format=time_format)

    return data_MFI, data_SWE

def TEC(time_format, lat_width_deg=2, lon_width_deg=2):
    TEC_path = current_directory + r"data_gathering\src\data\TEC\gps260202g.001.hdf5"

    data_tec = load_TEC(TEC_path, lat_width_deg=lat_width_deg, lon_width_deg=lon_width_deg, time_format=time_format)

    return data_tec


def plot_EISCAT(fig, axs, data_eiscat, time_format):


    if time_format == "unix": fig.supxlabel("Unix time [s]")
    elif time_format == "datetime": fig.supxlabel("Time")
    else: print("specify time format")
    fig.supylabel("Altitude [km]")
    fig.tight_layout()    

    T, H = np.meshgrid(data_eiscat["time"], data_eiscat["altitude"])
    colorranges = {
        "electron density":(1e10,1e12),
        "electron temperature":(0, 4000),
        "ion temperature":(0, 3000),
        "ion velocity":(-200, 200)
    }
    i = 0
    for key in data_eiscat.keys():
        if key == "time" or key=="altitude":
            continue
        else:
            plot_variable(fig=fig, ax=axs[i], variables=(T, H, data_eiscat[key]), 
                          name=key, title="Beata", crange=colorranges[key])
            i += 1

def plot_EISCAT_RGB(fig, ax, data_eiscat, variable_keys, logscale=False):

    # plot each variable as the rgb colors

    t = data_eiscat["time"]
    h = data_eiscat["altitude"]
    T, H = np.meshgrid(t, h)

    var1 = data_eiscat[variable_keys[0]]
    var2 = data_eiscat[variable_keys[1]]
    var3 = data_eiscat[variable_keys[2]]
    if logscale:
        var1 = np.log(var1)
        var2 = np.log(var2)
        var3 = np.log(var3)

    var1_max = np.nanmax(var1)    
    var2_max = np.nanmax(var2)
    var3_max = np.nanmax(var3)
    var1_normalized = np.clip(var1/var1_max, 0, 1) * 50
    var2_normalized = np.clip(var2/var2_max, 0, 1) * 50
    var3_normalized = np.clip(var3/var3_max, 0, 1) * 50
    
    rgb_array = np.stack(
        (
            var1_normalized,
            var2_normalized,
            var3_normalized
        ),
        axis=2
    )
    ax.imshow(
        rgb_array,
        origin="lower",
        aspect="auto",
        extent=[t.min(), t.max(), h.min(), h.max()]        
    )

    ax.legend(loc=(1, 0.75), title=f"R:{variable_keys[0]}\nG:{variable_keys[1]}\nB:{variable_keys[2]}")
    fig.supxlabel("Time [s]")
    fig.supylabel("Altitude [km]")



def main():
    time_format = "unix"
    data_beata, data_bella = EISCAT(time_format=time_format)
    data_mfi, data_swe = ACE(time_format=time_format)
    data_tec = TEC(time_format=time_format)


    time_start = datetime.datetime(2026, 2, 3, 22, 30, 0).timestamp()
    time_end   = datetime.datetime(2026, 2, 4, 0,  0,  0).timestamp()
    alt_start = 150 # km
    alt_end   = 400 # km

    fig_RGB, ax_RGB = subplots()
    fig_RGB.suptitle("Bella RGB, log scaled")
    plot_EISCAT_RGB(fig=fig_RGB, ax=ax_RGB, data_eiscat=data_bella, 
                    variable_keys=["ion temperature", "electron temperature", "ion velocity"], logscale=0)
    ax_RGB.set_ylim(alt_start, alt_end)
    ax_RGB.set_xlim(time_start, time_end)
    fig_RGB.tight_layout()

    # fig_RGB, ax_RGB = subplots()
    # fig_RGB.suptitle("Beata RGB, log scaled")
    # plot_EISCAT_RGB(fig=fig_RGB, ax=ax_RGB, data_eiscat=data_beata, 
    #                 variable_keys=["ion temperature", "electron temperature", "ion velocity"], logscale=0)
    # fig_RGB.tight_layout()

    fig_bella, axs_bella = subplots(len(data_bella.keys())-2, figsize=(8, 10))
    fig_bella.suptitle("Bella")
    plot_EISCAT(fig_bella, axs_bella, data_bella, time_format=time_format)
    for ax in axs_bella: ax.set_xlim(time_start, time_end), ax.set_ylim(alt_start, alt_end)


    # fig_beata, axs_beata = subplots(len(data_beata.keys())-2, figsize=(8, 10))
    # fig_beata.suptitle("Beata")
    # plot_EISCAT(fig_beata, axs_beata, data_beata, time_format=time_format)


    show()



