
import numpy as np
import matplotlib.pyplot as plt

from numpy import array, linspace
from matplotlib.pyplot import subplots, show 

import datetime
import h5py

TRO_lon = 180 + int(18.9553)
TRO_lat = 90 + int(69.6492)

def extract_TEC_timeseries(data, lat_width_deg, lon_width_deg):
    TEC_data = data["tec"]


    # print(f"{TEC_data.shape = }")
    # print(f"Taking data from box: {TRO_lat-lat_width_deg} : {TRO_lat+lat_width_deg}    x    {TRO_lon-lon_width_deg} : {TRO_lon + lon_width_deg}")
    TECs = []
    times = []
    for time_index in range(TEC_data.shape[2]):#range(time_expriment_start_min//5 - 60//5, time_expriment_end_min//5):#TEC_data.shape[2]):
        collector = []
        for lat in range(TRO_lat - lat_width_deg, TRO_lat + lat_width_deg):#range(TEC_data.shape[0]):
            for lon in range(TRO_lon - lon_width_deg, TRO_lon + lon_width_deg):#TEC_data.shape[1]):
                datapoint = TEC_data[lat, lon, time_index]
                if not np.isnan(datapoint):
                    collector.append(datapoint)


        # time = 5*time_index - time_expriment_start_min
        time = time_index
        times.append(time)
        average_TEC = np.mean(collector)
        TECs.append(average_TEC)

    TECs = array(TECs)
    times = array(times)
    return times, TECs


def plot_time_series(lat_widths, lon_widths, fig, ax, title="nameless",):
    for i, (lat_width, lon_width) in enumerate(zip(lat_widths, lon_widths)):#range(0, 10, 2):#[1, 2,3, 5, 10, 20, 55]:
        times, TECs = extract_TEC_timeseries(lat_width_deg=lat_width, lon_width_deg=lon_width)

        ax.plot(times, TECs, label=f"{lat_width} {lon_width}")

    # ax.hlines(0, time_expriment_start_min, time_expriment_end_min)
    # ax.text(time_expriment_start_min, 0, "Experiment time")
    ax.legend(title="Padding")
    ax.set_xlabel("Time since start of experiment [min]")
    ax.set_ylabel(r"TEC [$10^{16}$ $\text{m}^{-2}$]")


def extract_datetime_lines(doc):
    '''
    looking for following information

    (b'IBYRT               2026 Beginning year                                         ',)
    (b'IBDTT               0202 Beginning month and day                                ',)
    (b'IBHMT               0000 Beginning UT hour and minute                           ',)
    (b'IBCST               0000 Beginning centisecond                                  ',)
    (b'IEYRT               2026 Ending year                                            ',)
    (b'IEDTT               0203 Ending month and day                                   ',)
    (b'IEHMT               0000 Ending UT hour and minute                              ',)
    (b'IECST               0000 Ending centisecond                                     ',)
    '''
    year_start = None
    month_day_start = None
    hour_minute_start = None
    centisecond_start = None
    year_end = None
    month_day_end = None
    hour_minute_end = None
    centisecond_end = None
    start_line = 10
    line_counter = 0
    for line  in doc["Metadata"]["Experiment Notes"]:
        if line_counter < start_line:
            line_counter += 1
            continue

        # starts
        if str(line)[:8] == "(b'IBYRT":
            year_start = str(line)
   
        if str(line)[:8] == "(b'IBDTT":
            month_day_start = str(line)
        
        if str(line)[:8] == "(b'IBHMT":
            hour_minute_start = str(line)

        if str(line)[:8] == "(b'IBCST":
            centisecond_start = str(line)
        

        # endings
        if str(line)[:8] == "(b'IEYRT":
            year_end = str(line)
   
        if str(line)[:8] == "(b'IEDTT":
            month_day_end = str(line)
        
        if str(line)[:8] == "(b'IEHMT":
            hour_minute_end = str(line)

        if str(line)[:8] == "(b'IECST":
            centisecond_end = str(line)
        
    return (
            year_start,
            month_day_start,
            hour_minute_start,
            centisecond_start,
            year_end,
            month_day_end,
            hour_minute_end,
            centisecond_end
        )
    

def extract_datetime(doc):
    (
        year_start_line,
        month_day_start_line,
        hour_minute_start_line,
        centisecond_start_line,
        year_end_line,
        month_day_end_line,
        hour_minute_end_line,
        centisecond_end_line

    ) = extract_datetime_lines(doc=doc)
    year_start          = int(year_start_line.split()[1])
    year_end            = int(year_end_line.split()[1])
    month_start         = int(month_day_start_line.split()[1][0:2])
    day_start           = int(month_day_start_line.split()[1][2:4])
    month_end           = int(month_day_end_line.split()[1][0:2])
    day_end             = int(month_day_end_line.split()[1][2:4])
    hour_start          = int(hour_minute_start_line.split()[1][0:2])
    minute_start        = int(hour_minute_start_line.split()[1][2:4])
    hour_end            = int(hour_minute_end_line.split()[1][0:2])
    minute_end          = int(hour_minute_end_line.split()[1][2:4])
    centisecond_start   = int(centisecond_start_line.split()[1])
    centisecond_end     = int(centisecond_end_line.split()[1])

    DT_start = datetime.datetime(
        year=year_start,
        month=month_start,
        day=day_start,
        hour=hour_start,
        minute=minute_start,
        second=centisecond_start//10
    )
    DT_end = datetime.datetime(
        year=year_end,
        month=month_end,
        day=day_end,
        hour=hour_end,
        minute=minute_end,
        second=centisecond_end//10
    )

    return DT_start, DT_end


def get_times(doc):
    DT_start, DT_end = extract_datetime(doc)

    time_unix_start = DT_start.timestamp()
    time_unix_end = DT_end.timestamp()

    return time_unix_start, time_unix_end


def load(path, lat_width_deg, lon_width_deg, time_format):

    doc = h5py.File(path, "r")
    data = doc['Data']["Array Layout"]["2D Parameters"]

    times_indicies, TECs = extract_TEC_timeseries(data, lat_width_deg=lat_width_deg, lon_width_deg=lon_width_deg)
    time_unix_start, time_unix_end = get_times(doc)
    
    times_unix = linspace(time_unix_start, time_unix_end, len(times_indicies))

    if time_format=="datetime":
        times_datetime = []
        for time_unix in times_unix:
            time_datetime = datetime.datetime.fromtimestamp(time_unix)
            times_datetime.append(time_datetime)

        times = times_datetime
    
    elif time_format == "unix":
        times = times_unix
    
    else:
        print(f"Time format must be 'unix' or 'datetime'")  

    return {
        "time":times,
        "tec":TECs
    }
    

def load_naive(
        path = r"C:\Users\NoahH\OneDrive - UiT Office 365\Personal OneDrive copy\Tromsø\UiT Arctic University of Norway\Techniques for Investigating the Near-Earth Space Environment\HEIMDALL\data_gathering\src\data\TEC\gps260202g.001.hdf5",
        time_format="unix"
    ):
    gns_path = path
    # gns_path = r"data\gps260202g.001.hdf5"
    
    doc = h5py.File(gns_path, "r")
    data = doc['Data']["Array Layout"]["2D Parameters"]
    data_param = doc['Metadata']["Experiment Notes"]
    # print(array(data_param))

    # doc.close()
    time_unix_start, time_unix_end = get_times(doc)
    
    lats = np.linspace(0, data["tec"].shape[0], data["tec"].shape[0])
    lons = np.linspace(0, data["tec"].shape[1], data["tec"].shape[1])
    times_unix = linspace(time_unix_start, time_unix_end, data["tec"].shape[2])


    return {
        "time":times_unix,
        "lat":lats, 
        "lon":lons, 
        "tec":data["tec"]
    }



#----------------------------------global animation--------------------------------------

def make_animation(data, saveloc=None, title="", save=False):
    # data = load_naive(time_format="unix")

    # at every time step, we have a global map. The goal is to make a plot for every time step.
    # -> thus first step is to pick out a map


    TEC = data["tec"]
    print(f"{TEC.shape = }")

    import matplotlib.animation as animation

    fig, ax=subplots()
    fig.suptitle(title)
    image = ax.imshow(TEC[:, :, 0])
    time_unix = data["time"][0]
    DT = datetime.datetime.fromtimestamp(time_unix)
    ax.set_title(f"{DT.day},{DT.hour}")
    ax.set_aspect(360/30)
    
    def update(i):
        # only use every 50th entry
        # if not (i % 50 == 0):
        #     return
        image.set_data(TEC[:, :, i])
        time_unix = data["time"][i]
        DT = datetime.datetime.fromtimestamp(time_unix)
        ax.set_title(f"{DT.day},{DT.hour}")
        print(f"animating:   {i} / {TEC.shape[2]}     ", end="\r")
    anim = animation.FuncAnimation(fig, update, TEC.shape[2], interval=1)

    if save: 
        anim.save("figures\\month animation.gif", fps=60)

    # anim.save(saveloc + "\\" + "tec_anim.gif")
    show()
# make_animation()