
# general imports
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# specific imports
from numpy import array, reshape, zeros
from matplotlib.pyplot import subplots, imshow, show, close
from incremental_pca_torch import IncrementalPCA

# module imports
from TEC.TEC import load_naive as load_single, make_animation
from principal_component_analysis import PCA as myPCA # unused
from dataset_handler import build_large_dataset, interpolate_tec
from centering import get_peaks, center_concomic, center_midday, center_midnight

# changes depending on system
 # file_seperator = "\\"
# file_seperator = "/"

figurefolder = "figures" + file_seperator
datafolder = "..\data"
datafolder = "/data/nonie/"
#-------------------------------function definitions-------------------------------
#----------------------principal component analysis----------------------
def weight_by_latitude(tec: np.ndarray) -> np.ndarray:
    """Weight TEC values by sqrt(latitude).
    Each TEC value is multiplied by sqrt(lat_index) for the corresponding latitude.
    """
    nlat = tec.shape[0]
    nlon = tec.shape[1]
    ntime = tec.shape[2]
    # Create latitude weight array: sqrt(0), sqrt(1), sqrt(2), ..., sqrt(nlat-1)
    lats = 90 - np.arange(nlat)
    lat_weights = np.sqrt(lats)
    # Create weight tensor with same shape as tec
    weight_tensor = np.tile(lat_weights[:, np.newaxis, np.newaxis], (1, nlon, ntime))
    # Apply weights
    weighted_tec = tec * weight_tensor
    return weighted_tec


def find_principal_components(tec: np.ndarray, number_of_components: int)->np.ndarray:
    """Find the principal components of the TEC data using Incremental PCA from scikit-learn.
    The TEC data is reshaped to a 2D array of shape (lat*lon, time) before applying PCA, and the 
    resulting components are reshaped back to the original image format."""
    
    # save shape for later
    original_shape = tec.shape
    
    # weight by latitude
    tec = weight_by_latitude(tec)


    # reshape to 2D array of shape (time, lat*lon)
    # time, latxlon
    tec_columns = tec.reshape((-1, tec.shape[2]))
    # latxlon, time
    tec_columns = tec_columns.T
 
    # initialize PCA
    ipca = IncrementalPCA(
        n_components=number_of_components, 
        batch_size=256, 
        device='cpu'  # Use 'cpu' if no GPU available
    )
    # fit on data
    ipca.fit(tec_columns)

    # Transform data
    # X_transformed = ipca.transform(tec_columns)
    # Reconstruct data
    # X_reconstructed = ipca.inverse_transform(X_transformed)

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
    # fig, ax=subplots()
    # ax.imshow(mean, cmap="grey")
    # show()
    # plt.close()
    for i in range(tec.shape[2]):
        tec[:, :, i] -=  mean
    # tec = np.transpose(tec, axes=(1, 2, 0))
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


#----------------------plotting----------------------
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

        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)  # or -1 if you prefer clockwise

        # Replace degree labels with local-time style labels
        if geo_labels:
            tick_angles_deg = [90, 180, 270]
            tick_labels = ["dawn", "midday", "dusk"]
            ax.set_xticks(np.deg2rad(tick_angles_deg))
            ax.set_xticklabels(tick_labels)

        ax.set_ylim(0, nlat)
        ax.set_title(f"Component number {i}")
        fig.colorbar(mappable, ax=ax, shrink=0.8, label="TEC magnitude", pad=0.1)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"figures" + file_seperator + "{title}.png", dpi=150)

def time_series_plot(fig:plt.figure, axs:plt.axes, time:np.ndarray, time_series:np.ndarray, title:str)->None:
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
        # if not i%50 == 0:
            # return
        # pcolormesh stores values as a flattened array
        mappable.set_array(tec[:, :, i].ravel())
        ts = time[i]
        DT = datetime.datetime.fromtimestamp(ts)
        fig.suptitle(f"{DT.day}/{DT.month}-{DT.year}, {DT.hour}:{DT.minute}")
        print(f"animating:   {i} / {tec.shape[2]}     ", end="\r")

    anim = animation.FuncAnimation(fig, update, tec.shape[2], interval=1)

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
    animate = 0
    extract_18UTC_images = 0


    number_of_components = 9
    plot_size = plot_geo(number_of_components)

    #-------------dataset-------------
    if rebuild_master_data:
        build_large_dataset(title="raw_northern_tec", extract_northern=True, small_data=False)

    if reinterpolate or rebuild_sets:
        print(f"loading master data...")
        master_data = np.load("master_data" + file_seperator + "raw_northern_tec.npy", )
        tec = master_data

        #-------------interpolate-------------
        print(f"interpolating...")
        tec = interpolate_tec(tec)
        np.save("master_data" + file_seperator + "interpolated_tec.npy", tec)
    
    if rebuild_sets:

        #-------------centering-------------  
        print("centering")
        # do each set seperately, otherwise it gets too memory intense
        tec_md = center_midday(tec)
        tec_md = subtract_mean(tec_md)
        np.save("master_data" + file_seperator + "tec_midday.npy", tec_md)

    print("loading data")
    tec_md = np.load("master_data" + file_seperator + "tec_midday.npy")
    tec_raw = np.load("master_data" + file_seperator + "raw_northern_tec.npy")
    tec_int = np.load("master_data" + file_seperator + "interpolated_tec.npy")

    # construct time. Based on known time period. Should be updated with new data
    time_start = datetime.datetime(year=2026, month=1, day=3, hour=0, minute=0, second=0).timestamp()
    time_end = time_start +  tec_md.shape[2] * 60 * 5 # 5 minute interval
    time = np.linspace(time_start, time_end, tec_md.shape[2])

    #-------------find components-------------
    if do_pca:
        print("performing PCA")    
        tec_md_components, _ = find_principal_components(tec_md, number_of_components)
        tec_int_components, _ = find_principal_components(tec_int, number_of_components)
        
        # check orthonormality of principal components
        print("Checking othonomality")
        check_orthonomality(tec_md_components)
        check_orthonomality(tec_int_components)


    if do_pca:    
        print("Finding time series")
        time_coefficients_md = compute_time_coefficients(tec_md_components, tec_md)
        time_coefficients_int = compute_time_coefficients(tec_int_components, tec_int)

    #-------------plot components-------------
    if plot_principal_components:
        print(f"Plotting")
        fig_md, axs_md = subplots(number_of_components//plot_size, plot_size, figsize=(14, 14), 
                           subplot_kw=dict(projection='polar'))
        plot_components_polar(fig_md, axs_md, tec_md_components, "Nothern Components Midday", geo_labels=True)
        fig_int, axs_int = subplots(number_of_components//plot_size, plot_size, figsize=(14, 14), 
                    subplot_kw=dict(projection='polar'))
        plot_components_polar(fig_int, axs_int, tec_int_components, "Nothern Components Geographic", geo_labels=False)

    #-------------plot time series-------------
    if plot_time_series:
        fig_md, axs_md=subplots(number_of_components//plot_size, plot_size, figsize=(10, 10))
        time_series_plot(fig_md, axs_md, time, time_coefficients_md, title="Coefficients, Northern Midday Components")
        fig_int, axs_int=subplots(number_of_components//plot_size, plot_size, figsize=(10, 10))
        time_series_plot(fig_md, axs_md, time, time_coefficients_md, title="Coefficients, Nothern Geographic Components")

    if plot_both:
        n_rows = 4
        fig_md, axs_md = subplots(2, n_rows, figsize=(15, 15))
        fig_int, axs_int = subplots(2, n_rows, figsize=(15, 15))
        component_and_timeseries_plot(fig_md, axs_md, tec_md_components[:, :, :n_rows], time, time_coefficients_md[:n_rows, :], title="", geo_labels=True)
        component_and_timeseries_plot(fig_int, axs_int, tec_int_components[:, :, :n_rows], time, time_coefficients_int[:n_rows, :], title="", geo_labels=False)
        fig_md.savefig("figures" + file_seperator + "Overview Midday.png")
        fig_int.savefig("figures" + file_seperator + "Overview Geographic.png")


    #-------------animation-------------
    if animate:
        print("animating")
        length_idx = 1000
        tec_md = tec_md[:, :, :length_idx]
        tec_raw = tec_raw[:, :, :length_idx]
        tec_int = tec_int[:, :, :length_idx]
        
        # make_polar_viewer(tec_raw, time, "midday centered")
        anim = make_polar_animation(tec_md, time, title="Midday centered", save=False, geo_labels=True)
        anim.save("figures" + file_seperator + "gifs" + file_seperator + "Midday centered.gif", writer="pillow")
        del tec_md; del anim
        anim = make_polar_animation(tec_raw, time, title="Raw TEC", save=False)
        anim.save("figures" + file_seperator + "gifs" + file_seperator + "Raw.gif", writer="pillow")
        del tec_raw; del anim
        anim = make_polar_animation(tec_int, time, title="Interpolated centered", save=False)
        anim.save("figures" + file_seperator + "gifs" + file_seperator + "Interpolated.gif", writer="pillow")
        del tec_int; del anim
        # show()

    if extract_18UTC_images:
        extract_image_at_18UTC(tec_md, time)

if __name__ == "__main__":
    main()