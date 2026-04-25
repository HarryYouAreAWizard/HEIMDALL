

import numpy as np
import matplotlib.pyplot as plt
import datetime

from numpy import array, loadtxt
from matplotlib.pyplot import subplots, show

# MFI_path = r"data\ACE\AC_K0_MFI_496241.csv"
# SWE_path = r"data\ACE\AC_K0_SWE_496241.csv"



def TEC_timestamp_to_UNIX(timestamp):
    D, T = timestamp.split("T")
    year, month, day = D.split("-")
    hour, minute, second = T.split(":")
    year = int(year)
    month = int(month)
    day = int(day)
    hour = int(hour)
    minute = int(minute)
    second = int(float(second[:-1]))
    DT = datetime.datetime(
        year=year,
        month=month,
        day=day,
        hour=hour,
        minute=minute,
        second=second
    )
    time_UNIX = DT.timestamp()
    return time_UNIX

def TEC_timestamp_to_datetime(timestamp):
        D, T = timestamp.split("T")
        year, month, day = D.split("-")
        hour, minute, second = T.split(":")
        year = int(year)
        month = int(month)
        day = int(day)
        hour = int(hour)
        minute = int(minute)
        second = int(float(second[:-1]))
        DT = datetime.datetime(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second
        )
        return DT



def load_MFI(path, time_format):
    doc = open(path, mode="r")
    timestamps = []
    B_magnitude = []
    B_x = []
    B_y = []
    B_z = []
    for line_number, line in enumerate(doc):

        if line_number == 56:
            info_string = line

        if 56 < line_number:
            try:
                a, b, c, d, e = line.split(",")
                h, m ,s = [float(element[:2]) for element in a.split("T")[1].split(":")]
            except Exception: pass
            if h >= 16:
                timestamps.append(a)
                B_magnitude.append(float(b))
                B_x.append(float(c))
                B_y.append(float(d))
                B_z.append(float(e))

    if time_format=="unix":
        time_unix = []
        for timestamp in timestamps:
            time_unix.append(TEC_timestamp_to_UNIX(timestamp))
        timestamps = array(time_unix)

    elif time_format=="datetime":
        time_datetime = []
        for timestamp in timestamps:
            time_datetime.append(TEC_timestamp_to_datetime(timestamp))
        timestamps = array(time_datetime)

    else:
        print(f"Time format must be 'unix' or 'datetime'")

    return {
        "time":timestamps,
        "B magnitude":array(B_magnitude),
        "Bx":array(B_x),
        "By":array(B_y),
        "Bz":array(B_z)
    }

# (
#     timestamps,
#     B_magnitude,
#     B_x,
#     B_y,
#     B_z
# ) = load_MFI(MFI_path)



# fig, axs=subplots(2, 2)
# for ax, variable, title in zip(
#     axs.flatten(), 
#     [B_magnitude, B_x, B_y, B_z],
#     [r"$|B|$", r"$B_x$", r"$B_y$", r"$B_z$"]
# ):
#     times = np.linspace(-60*2, 60*6, len(variable))
#     ax.plot(times, variable)
#     ax.set_title(title)
#     # ax.set_xticks([e for i, e in enumerate(timestamps) if i%len(timestamps) == 0])

# fig.supylabel("Magnetic field [nT]")
# fig.supxlabel("Time since start of experiement [min]")
# fig.tight_layout()
# fig.savefig("figures\\ACE\\mag field.png")



def load_SWE(path, time_format):
    doc = open(path, mode="r")
    timestamps = []
    number_density = []
    bulk_speed = []
    start_line = 58
    for line_number, line in enumerate(doc):

        if line_number == start_line:
            info_string = line
            # print(line)
        if start_line < line_number:
            try:
                a, b, c = line.split(",")
                number_density.append(float(b))
                bulk_speed.append(float(c))
                timestamps.append(a)
            except Exception: pass
    
    if time_format=="unix":
        time_unix = []
        for timestamp in timestamps:
            time_unix.append(TEC_timestamp_to_UNIX(timestamp))
        timestamps = array(time_unix)

    elif time_format=="datetime":
        time_datetime = []
        for timestamp in timestamps:
            time_datetime.append(TEC_timestamp_to_datetime(timestamp))
        timestamps = array(time_datetime)

    else:
        print(f"Time format must be 'unix' or 'datetime'")

    return {
        "time":timestamps,
        "number density":array(number_density),
        "bulk speed":array(bulk_speed)
    }

# (
#     timestamps,
#     number_density,
#     bulk_speed
# ) = load_SWE(SWE_path)

# fig, axs=subplots(2, 1)
# for ax, variable, title in zip(
#     axs, 
#     [number_density, bulk_speed], 
#     [r"number density [$\text{cm}^{-3}$]", r"bulk speed [$\frac{\text{km}}{\text{s}}$]"]
# ):
#     times = np.linspace(-60*2, 60*6, len(variable))
#     ax.plot(times, variable)
#     ax.set_ylabel(title)
#     # ax.set_xticks([e for i, e in enumerate(timestamps) if i%len(timestamps) == 0])
# fig.supxlabel("Time since start of experiement [min]")
# fig.suptitle("ACE density and speed")
# fig.tight_layout()
# fig.savefig("figures\\ACE\\density and speed.png")