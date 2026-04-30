




import datetime
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


from src.ACE import load_MFI, load_SWE

def load():
    timeseries = np.load("time_series\\time_series_midday.npy")
    timeseries_time = np.load("time_series\\time.npy")
    SWE = load_SWE(r"src\data\ACE\AC_K0_SWE_496241.csv", "unix")
    MFI = load_MFI(r"src\data\ACE\AC_K0_MFI_496241.csv", "unix")
    return (
        timeseries,
        timeseries_time,
        SWE,
        MFI
    )

def scale_and_center_time(src_time, target_time):
    seconds_per_day = 60 * 60
    src_time = np.asarray(src_time, dtype=float) / (seconds_per_day)
    target_time = np.asarray(target_time, dtype=float) / (seconds_per_day)

    # Use the first MFI sample as the shared origin so both series are plotted
    # on the same relative time axis.
    origin = src_time[0]
    src_time = src_time - origin
    target_time = target_time - origin
    return src_time, target_time

def sort_around(src_time, target_time, target_var):
    # sort out symetricly around +- 10
    diff = np.abs(target_time - src_time[0])
    condition = diff < 10
    target_var = target_var[condition]
    target_time = target_time[condition]

    # remove trailing under 5
    diff = target_time - src_time[0]
    condition = diff > -5
    target_var = target_var[condition]
    target_time = target_time[condition]
    
    return target_time, target_var

def plot_raw(fig, ax, mfi_time, bz, timeseries_time, timeseries, title=""):

    ax_ = ax.twinx()
    tsp = ax.plot(timeseries_time, timeseries)
    bzp = ax_.plot(mfi_time, bz, c="tab:orange")
    ax.set_xlabel("Hours after ACE data start")
    # ax.set_ylim(-1, 2)
    ax.set_ylabel("Time serie", c="tab:blue")
    ax_.set_ylabel("Bz", c="tab:orange")
    # ax.legend()
    fig.suptitle(title)
    return tsp, bzp

def animate_raw_bz_and_time_series(fig, ax, mfi_time, bz, timeseries_time, timeseries, title=""):
    
    tsp, bzp = plot_raw(fig, ax, mfi_time, bz, timeseries_time, timeseries, title=title)

    indicies = range(1000)
    def update(i):
        print(f"{i} / {len(indicies)}", end="\r")
        bz_shifted= np.roll(bz, shift=-i)
        bzp[0].set_data(mfi_time, bz_shifted)
        return tsp

    anim = FuncAnimation(fig, update, frames=indicies, interval=50, blit=False)
    anim.save("figures\\gifs\\" + title + ".gif")
    return anim

def plot_timeseries(time, timeseries, title):
    fig, ax=plt.subplots()
    for ts in timeseries:
        ax.plot(time, ts)
    fig.savefig("figures\\" + title + ".png")

def plot_solar_wind_bz(time, bz, title):
    fig, ax=plt.subplots()
    ax.plot(time, bz)
    fig.savefig("figures\\" + title + "png")

def main():


    timeseries, timeseries_time, SWE, MFI = load()
    print(f"{SWE.keys() = }")
    print(f"{MFI.keys() = }")

    mfi_time = MFI["time"]
    mfi_datetimes = [datetime.datetime.fromtimestamp(t) for t in mfi_time]
    bz = MFI["Bz"]

    print(f"{timeseries_time.shape = }")
    print(f"{timeseries.shape = }")
    print(f"{mfi_time.shape = }")
    print(f"{bz.shape = }")

    origin_date = mfi_time[0].copy()
    start_date = datetime.datetime.fromtimestamp(origin_date)
    timeseries = timeseries[0, :]
    mfi_time, timeseries_time = scale_and_center_time(mfi_time, timeseries_time)
    timeseries_time, timeseries = sort_around(mfi_time, timeseries_time, timeseries)

    fig, ax=plt.subplots()
    title = "raw bz and components time series"
    plot_raw(fig, ax, mfi_time, bz, timeseries_time, timeseries, title=title)
    ax.set_title(f"{start_date.year} - {start_date.month} - {start_date.day}")
    fig.tight_layout()
    fig.savefig("figures\\" + title + ".png")
    plt.close()

    fig, ax=plt.subplots()
    title = "corr_delay"
    anim = animate_raw_bz_and_time_series(fig, ax, mfi_time, bz, timeseries_time, timeseries, title=title)
    

    # plot_timeseries(timeseries_time, timeseries, "raw timeseries")
    # plot_solar_wind_bz(mfi_time, bz, "raw bz")
    # print(f"{timeseries_time = }")
    # timeseries_datetimes = [datetime.datetime.fromtimestamp(t) for t in timeseries_time]

main()