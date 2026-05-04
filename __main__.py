
# general imports
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# specific imports
from numpy import array, reshape, zeros
from matplotlib.pyplot import subplots, imshow, show, close


# module imports
# from TEC.TEC import load_naive as load_single, make_animation
from principal_component_analysis import (
    find_principal_components, 
    check_orthonomality, 
    subtract_mean, 
    compute_time_coefficients
)
from dataset_handler import build_large_dataset, interpolate_tec
from centering import get_peaks, center_concomic, center_midday, center_midnight


# changes depending on system
# file_seperator = "\\"
file_seperator = "/"
figurefolder = "figures" + file_seperator
datafolder = "..\data"
datafolder = "/data/nonie/"


def time_from_gps_files()->np.ndarray:
    path = "/data/nonie/tec_data"
    list_of_files = os.listdir(path)
    dates = []
    for file in list_of_files:
        year = file[3:5]
        month = file[5:7]
        day = file[7:9]
        year = int(year) + 2000
        month = int(month)
        day = int(day)
        date = datetime.datetime(year, month, day)
        dates.append(date)
    dates = sorted(dates)
    date_start = dates[0].timestamp()
    date_end = dates[-1].timestamp()
    min_per_day = 60*24
    n_5_min_per_day = min_per_day//5
    time = np.linspace(date_start, date_end, len(dates)*n_5_min_per_day)
    print(f"{dates[0] = }{date_start = }")
    print(f"{dates[-1] = }{date_end = }")
    return time


#-------------------------------plot functions-------------------------------
def plot_geo(number_of_components):
    # calculate plot grid geometry
    for i in [2, 3, 5]:
        if number_of_components%i == 0:
            plot_size = i
            break
    return plot_size
    
def plot_components(fig: plt.Figure, axs: np.ndarray, tec_components: np.ndarray, title: str = "components")->None:
    """Plot the principal components of the TEC data as images."""
    vmin = np.nanmin(tec_components)
    vmax = np.nanmax(tec_components)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # plot resulting principal components
    for i, ax in enumerate(axs.flatten()):
        image = ax.imshow(tec_components[:, :, i], norm=norm)
        ax.set_aspect(360.0/30.0)
        ax.set_title(f"Component number {i}")
        ax.set_xticks([0, 180, 360])
        ax.set_yticks([0, 15, 30])
        fig.colorbar(image, ax=ax, shrink=0.8, label="TEC magnitude")
    
    fig.supxlabel("Longitude index")
    fig.supylabel("Latitude index")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"figures" + file_seperator + "{title}.png")

def plot_components_polar(fig: plt.Figure, axs: np.ndarray, tec_components: np.ndarray, title: str = "components", geo_labels=False)->None:
    """Plot the principal components of the TEC data as circular polar plots."""
    vmin = np.nanmin(tec_components)
    vmax = np.nanmax(tec_components)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    nlat, nlon, ncomp = tec_components.shape
    
    # Create polar coordinate grids
    lon = np.linspace(0, 360, nlon, endpoint=False)  # 0-360 degrees
    lat = np.linspace(0, nlat, nlat)  # 0-30 (or whatever your lat range is)
    
    theta = np.deg2rad(lon)  # Convert to radians
    radius = lat
    
    # Create meshgrids for polar coordinates
    THETA, RADIUS = np.meshgrid(theta, radius)
    
    # Plot each component
    for i, ax in enumerate(axs.flatten()):  
        data = tec_components[:, :, i]  
        mappable = ax.pcolormesh(THETA, RADIUS, data, norm=norm, cmap='viridis')

        # Replace degree labels with local-time style labels
        if geo_labels:
            ax.set_theta_zero_location('N')
            tick_angles_deg = [90, 180, 270]
            tick_labels = ["dawn", "midday", "dusk"]
            ax.set_xticks(np.deg2rad(tick_angles_deg))
            ax.set_xticklabels(tick_labels)
        else:
            ax.set_theta_zero_location('S')
        ax.set_theta_direction(1) # or -1 if you prefer clockwise
        ax.set_ylim(0, nlat)
        ax.set_title(f"Component number {i}")
        fig.colorbar(mappable, ax=ax, shrink=0.8, label="TEC magnitude", pad=0.1)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"figures" + file_seperator + "{title}.png", dpi=150)

def time_series_plot(fig:plt.figure, axs:plt.axes, time:np.ndarray, time_series:np.ndarray, title:str)->None:
    print(f"{time.shape = }")
    print(f"{time_series.shape = }")
    axs = axs.flatten()
    time = [datetime.datetime.fromtimestamp(t) for t in time]

    for i in range(time_series.shape[0]):
        axs[i].scatter(time, time_series[i], s=1)
        axs[i].set_xticks([time[i] for i in range(len(time)) if i%1000==0])
        axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"figures" + file_seperator + "{title}")

def component_and_timeseries_plot(fig, axs, tec_components, time, time_coefficients, title, *args, **kwargs):
    n_components = tec_components.shape[2]

    # Build an Nx2 layout with polar axes on the left and Cartesian axes on the right.
    fig.clear()
    left_axes = [fig.add_subplot(n_components, 2, 2*i + 1, projection='polar') for i in range(n_components)]
    right_axes = [fig.add_subplot(n_components, 2, 2*i + 2) for i in range(n_components)]

    plot_components_polar(fig, np.array(left_axes), tec_components, title="", **kwargs)
    time_series_plot(fig, np.array(right_axes), time, time_coefficients, title="")
    fig.suptitle(title)
    fig.tight_layout()
    
def make_polar_animation(tec, time, title, save=False, geo_labels=False):
    import matplotlib.animation as animation

    fig, ax = subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.suptitle(title)
    # setup first plot
    nlat, nlon, _ = tec.shape
    # Create polar coordinate grids
    lon = np.linspace(0, 360, nlon, endpoint=False)  # 0-360 degrees
    lat = np.linspace(0, nlat, nlat)  # 0-30 (or whatever your lat range is)
    theta = np.deg2rad(lon)  # Convert to radians
    radius = lat
    # Create meshgrids for polar coordinates
    THETA, RADIUS = np.meshgrid(theta, radius)

    mappable = ax.pcolormesh(
        THETA, 
        RADIUS, 
        tec[:, :, 0], 
        # norm=norm, 
        cmap='viridis', 
        shading='auto'
    )

    # heading  = "S" if geo_labels else "N"
    direction = -1 if geo_labels else 1 # 1 if you prefer clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(direction)   

    # Replace degree labels with local-time style labels
    if geo_labels:
        tick_angles_deg = [90, 180, 270]
        tick_labels = ["dawn", "midday", "dusk"]
    else:
        tick_angles_deg = [0, 90, 180, 270]
        tick_labels = [180, 270, 0, 90]
    ax.set_xticks(np.deg2rad(tick_angles_deg))
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, nlat)
    fig.colorbar(mappable, ax=ax, shrink=0.8, label="TEC magnitude", pad=0.1)

    DT = datetime.datetime.fromtimestamp(time[0])

    fig.suptitle(f"{DT.day},{DT.hour}")
    
    # ax.set_aspect(360/30)
    def update(i):
        # i *= 50
        # if not i%50 == 0:
            # return
        # pcolormesh stores values as a flattened array
        mappable.set_array(tec[:, :, i*2].ravel())
        ts = time[i*2]
        DT = datetime.datetime.fromtimestamp(ts)
        fig.suptitle(f"{DT.day}/{DT.month}-{DT.year}, {DT.hour}:{DT.minute}")
        print(f"animating:   {i} / {tec.shape[2]//2}     ", end="\r")

    anim = animation.FuncAnimation(fig, update, tec.shape[2]//2, interval=1)

    if save: 
        anim.save("figures" + file_seperator + "month_animation.gif", writer="pillow", fps=60)

    return anim

def make_polar_viewer(tec, time, title, geo_labels=False):
    import matplotlib.pyplot as animation
    from matplotlib.widgets import Slider

    nlat, nlon, nframes = tec.shape

    lon = np.linspace(0, 360, nlon, endpoint=False)
    lat = np.linspace(0, nlat, nlat)
    theta = np.deg2rad(lon)
    radius = lat
    THETA, RADIUS = np.meshgrid(theta, radius)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    fig.suptitle(title)

    mesh = ax.pcolormesh(THETA, RADIUS, tec[:, :, 0], cmap='viridis', shading='auto')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)
    if geo_labels:
        tick_angles_deg = [90, 180, 270]
        tick_labels = ["dawn", "midday", "dusk"]
        ax.set_xticks(np.deg2rad(tick_angles_deg))
        ax.set_xticklabels(tick_labels)

    ax.set_ylim(0, nlat)
    ax.set_title(f"{datetime.datetime.fromtimestamp(time[0]).day},{datetime.datetime.fromtimestamp(time[0]).hour}")
    fig.colorbar(mesh, ax=ax, shrink=0.8, label="TEC magnitude", pad=0.1)

    # slider axis
    slider_ax = fig.add_axes([0.15, 0.03, 0.7, 0.03])
    slider = Slider(slider_ax, "Frame", 0, nframes - 1, valinit=0, valstep=1)

    def update(frame):
        frame = int(frame)
        mesh.set_array(tec[:, :, frame].ravel())
        DT = datetime.datetime.fromtimestamp(time[frame])
        fig.suptitle(f"{DT.day}/{DT.month}-{DT.year}, {DT.hour}:{DT.minute}")
        fig.canvas.draw_idle()

    def on_slider_change(val):
        update(val)

    def on_scroll(event):
        current = int(slider.val)
        if event.button == 'up':
            next_frame = min(current + 1, nframes - 1)
        elif event.button == 'down':
            next_frame = max(current - 1, 0)
        else:
            return
        slider.set_val(next_frame)

    slider.on_changed(on_slider_change)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    update(0)
    plt.show()

def plot_single_polar(tec):
    fig, ax = subplots(subplot_kw=dict(projection='polar'))
    nlat, nlon = tec.shape
    # Create polar coordinate grids
    lon = np.linspace(0, 360, nlon, endpoint=False)  # 0-360 degrees
    lat = np.linspace(0, nlat, nlat)  # 0-30 (or whatever your lat range is)
    theta = np.deg2rad(lon)  # Convert to radians
    radius = lat
    # Create meshgrids for polar coordinates
    THETA, RADIUS = np.meshgrid(theta, radius)

    mappable = ax.pcolormesh(
        THETA, 
        RADIUS, 
        tec[:, :], 
        # norm=norm, 
        cmap='viridis', 
        shading='auto'
    )

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(1)  # or -1 if you prefer clockwise

    # Replace degree labels with local-time style labels
    tick_angles_deg = [90, 180, 270]
    tick_labels = ["dawn", "midday", "dusk"]

    ax.set_xticks(np.deg2rad(tick_angles_deg))
    ax.set_xticklabels(tick_labels)
    return fig, ax

def extract_image_at_18UTC(tec, time):

    for i in range(len(time)):
        DT = datetime.datetime.fromtimestamp(time[i])
        if DT.hour == 18:
            print(DT)
            fig, ax = plot_single_polar(tec[:, :, i])
            fig.suptitle(f"{DT.day}/{DT.month}-{DT.year}, {DT.hour}:{DT.minute}")
            fig.savefig(f"animation_breakdowns" + file_seperator + "{DT.day}.png")
            close()
            

#-------------------------------main-------------------------------
def main()->None:
    # flags
    refind_centers = 0
    rebuild_master_data = 0
    reinterpolate = 0
    rebuild_sets = 0
    do_pca = 1
    plot_principal_components = 1
    plot_time_series = 1
    plot_both = 1
    animate = 1
    extract_18UTC_images = 0
    make_single_day_global_animation = 0

    n_days = 15 * (24*60)//5 # for time series plot
    number_of_components = 9
    plot_size = plot_geo(number_of_components)

    time = time_from_gps_files()
    np.save("time_series/time.npy", time) # for external analysis

    #-------------dataset-------------
    if rebuild_master_data:
        build_large_dataset(title="raw_northern_tec", extract_northern=True, small_data=False)

    if reinterpolate or rebuild_sets:
        print(f"loading master data...")
        master_data = np.load("/data/nonie/masterdata" + file_seperator + "raw_northern_tec.npy", )
        tec = master_data

        #-------------interpolate-------------
        print(f"interpolating...")
        tec = interpolate_tec(tec)
        np.save("/data/nonie/masterdata" + file_seperator + "interpolated_tec.npy", tec)
    
    if rebuild_sets:

        #-------------centering-------------  
        print("centering")
        # do each set seperately, otherwise it gets too memory intense
        tec_md = center_midday(tec)
        tec_md = subtract_mean(tec_md)
        np.save("/data/nonie/masterdata" + file_seperator + "tec_midday.npy", tec_md)
    
    if do_pca:
        print("loading data")
        tec_md = np.load("/data/nonie/masterdata" + file_seperator + "tec_midday.npy")
        tec_raw = np.load("/data/nonie/masterdata" + file_seperator + "raw_northern_tec.npy")
        tec_int = np.load("/data/nonie/masterdata" + file_seperator + "interpolated_tec.npy")

        # find components
        print("performing PCA")    
        tec_md_components, _ = find_principal_components(tec_md, number_of_components)
        tec_int_components, _ = find_principal_components(tec_int, number_of_components)
        
        # check orthonormality of principal components
        print("Checking othonomality")
        check_orthonomality(tec_md_components)
        check_orthonomality(tec_int_components)

        # save components
        np.save("/data/nonie/masterdata/components_midday.npy", tec_md_components)
        np.save("/data/nonie/masterdata/components_geographic.npy", tec_int_components)
        
        print("Finding time series")    
        # pick out last five days
        tec_md_short = tec_md[:, :, -n_days:]
        tec_int_short = tec_int[:, :, -n_days:]
        time_coefficients_md = compute_time_coefficients(tec_md_components, tec_md)
        time_coefficients_int = compute_time_coefficients(tec_int_components, tec_int)
        
        # save time series
        np.save("/data/nonie/masterdata/time_series_midday.npy", time_coefficients_md)
        np.save("/data/nonie/masterdata/time_series_geographic.npy", time_coefficients_int)

        # save for external data analysis
        np.save("/HEIMDALL/HEIMDALL/time_series/time_series_midday.npy", time_coefficients_md)
        np.save("/HEIMDALL/HEIMDALL/components/components_midday.npy", tec_md_components)
        np.save("/HEIMDALL/HEIMDALL/components/components_geographic.npy", tec_int_components)

    if not do_pca:
        print("loading components and time series...")
        time_coefficients_md = np.load("/data/nonie/masterdata/time_series_midday.npy")
        time_coefficients_int = np.load("/data/nonie/masterdata/time_series_geographic.npy")
        tec_md_components = np.load("/data/nonie/masterdata/components_midday.npy")
        tec_int_components = np.load("/data/nonie/masterdata/components_geographic.npy")
        print(f"{time_coefficients_md.shape = }")
        print(f"{time_coefficients_int.shape = }")
        print(f"{tec_md_components.shape = }")
        print(f"{tec_int_components.shape = }")

        # save for external data analysis
        np.save("time_series/time_series_midday.npy", time_coefficients_md)
        np.save("components/components_midday.npy", tec_md_components)
        np.save("components/components_geographic.npy", tec_int_components)



    # # construct time. Based on known time period. Should be updated with new data
    # time_start = datetime.datetime(year=2026, month=1, day=3, hour=0, minute=0, second=0).timestamp()
    # time_end = time_start +  tec_md.shape[2] * 60 * 5 # 5 minute interval
    # time = np.linspace(time_start, time_end, tec_md.shape[2])

    #-------------plot components-------------
    if plot_principal_components:
        print(f"Plotting components...")
        fig_md, axs_md = subplots(number_of_components//plot_size, plot_size, figsize=(14, 14), 
                           subplot_kw=dict(projection='polar'))
        plot_components_polar(fig_md, axs_md, tec_md_components, "Nothern Components Midday", geo_labels=True)
        fig_md.savefig("figures/Nothern Components Midday.png")
        fig_int, axs_int = subplots(number_of_components//plot_size, plot_size, figsize=(14, 14), 
                    subplot_kw=dict(projection='polar'))
        plot_components_polar(fig_int, axs_int, tec_int_components, "Nothern Components Geographic", geo_labels=False)
        fig_int.savefig("figures/Nothern Components Geographic.png")

    #-------------plot time series-------------
    if plot_time_series:
        print(f"Plotting time series...")
        fig_md, axs_md=subplots(number_of_components//plot_size, plot_size, figsize=(10, 10))
        time_series_plot(fig_md, axs_md, time[-n_days:],
                         time_coefficients_md[:, -n_days:], title="Coefficients, Northern Midday Components")
        fig_md.savefig("figures/Coefficients, Northern Midday Components.png")

        fig_int, axs_int=subplots(number_of_components//plot_size, plot_size, figsize=(10, 10))
        time_series_plot(fig_int, axs_int, time[-n_days:], 
                         time_coefficients_int[:, -n_days:], title="Coefficients, Nothern Geographic Components")
        fig_int.savefig("figures/Coefficients, Northern Geographic Components.png")

    if plot_both:
        print(f"plotting overviews")
        n_rows = 4
        fig_md, axs_md = subplots(2, n_rows, figsize=(15, 15))
        fig_int, axs_int = subplots(2, n_rows, figsize=(15, 15))
        component_and_timeseries_plot(fig_md, axs_md,   tec_md_components[:, :, :n_rows],  time[-n_days:], time_coefficients_md[:n_rows,  -n_days:], title="Midday", geo_labels=True)
        component_and_timeseries_plot(fig_int, axs_int, tec_int_components[:, :, :n_rows], time[-n_days:], time_coefficients_int[:n_rows, -n_days:], title="Geographic", geo_labels=False)
        fig_md.savefig("figures" + file_seperator + "Overview Midday.png")
        fig_int.savefig("figures" + file_seperator + "Overview Geographic.png")


    #-------------animation-------------
    if animate:
        print("loading data")
        tec_md = np.load("/data/nonie/masterdata" + file_seperator + "tec_midday.npy")
        tec_raw = np.load("/data/nonie/masterdata" + file_seperator + "raw_northern_tec.npy")
        tec_int = np.load("/data/nonie/masterdata" + file_seperator + "interpolated_tec.npy")

        print("animating")
        length_idx = 1500
        starting_point = -5000
        tec_md =   tec_md[:, :, starting_point:starting_point + length_idx]
        tec_raw = tec_raw[:, :, starting_point:starting_point + length_idx]
        tec_int = tec_int[:, :, starting_point:starting_point + length_idx]
        
        # make_polar_viewer(tec_raw, time, "midday centered")
        anim = make_polar_animation(tec_md, time, title="Midday centered", save=True, geo_labels=True)
        anim.save("figures" + file_seperator + "gifs" + file_seperator + "Midday centered.gif", writer="pillow")
        del tec_md; del anim
        anim = make_polar_animation(tec_raw, time, title="Raw TEC", save=True)
        anim.save("figures" + file_seperator + "gifs" + file_seperator + "Raw.gif", writer="pillow")
        del tec_raw; del anim
        anim = make_polar_animation(tec_int, time, title="Interpolated centered", save=True)
        anim.save("figures" + file_seperator + "gifs" + file_seperator + "Interpolated.gif", writer="pillow")
        del tec_int; del anim
        # show()

    if extract_18UTC_images:
        tec_md = np.load("/data/nonie/masterdata" + file_seperator + "tec_midday.npy")
        extract_image_at_18UTC(tec_md, time)

    if make_single_day_global_animation:
        from TEC.TEC import load_naive as load_single
        from matplotlib.animation import FuncAnimation
        tec_campaing_day = load_single("/data/nonie/tec_data/gps260202g.002.hdf5", time_format="unix")
        print(f"{tec_campaing_day['tec'].shape = }")
        fig, ax=plt.subplots()
        im = ax.imshow(tec_campaing_day["tec"][:, :, 0])
        def update(frame):
            im.set_data(tec_campaing_day["tec"][:, :, frame])
            DT = datetime.datetime.fromtimestamp(tec_campaing_day["time"][frame])
            plt.title(f"{DT.day}/{DT.month}-{DT.year}, {DT.hour}:{DT.minute}")
        anim = FuncAnimation(fig, update, frames=tec_campaing_day["tec"].shape[2], interval=100)
        anim.save("figures" + file_seperator + "gifs" + file_seperator + "Single_day.gif", writer="pillow", fps=10)
    

if __name__ == "__main__":
    main()