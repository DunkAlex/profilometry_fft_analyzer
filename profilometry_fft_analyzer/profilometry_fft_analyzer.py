import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button
from scipy import ndimage, signal
import json
import re

# set base directory for file access
base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)

# define I/O directories
datafile_dir = os.path.join(base_dir, r'data_files')
default_corners_path = os.path.join(base_dir, 'default_corners.csv')
default_colors_path = os.path.join(base_dir, 'default_colors.csv')
default_calibration_path = os.path.join(base_dir, 'default_calibration.json')
default_color_path = os.path.join(base_dir, "default_color.csv")


# True = run as script: uses defaults/saved params, calculates metrics for all
# images in data_files - outputs as script_results.csv
script_mode = True

# editable constants
IMAGE_NAME = '2539AMPM005.000_7L6QB240SOH1_MS01_New_AFM_2_DIEX1_DIEY0_0612_122829_Section_Ch1.001.jpeg'
SCRIPT_OUTPUT_FILENAME = 'script_results.csv'
# default image settings - change between figure types
USE_DEFAULT_AREA = True
USE_DEFAULT_COLOR = True
# tuned parameter metrics
TUNED_PARAM_FILENAME = 'tuned_params.json'
USE_TUNING_RESULTS = True # if False, will use default - poor results
PLOT_DIAGNOSTICS = False

SUPPORTED_IMAGE_EXTENSIONS = (
    ".bmp",
    ".dib",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pxm",
    ".pnm",
    ".sr",
    ".ras",
    ".jpeg",
    ".jpg",
    ".jpe",
    ".jp2",
    ".png",
    ".webp",
    ".tiff",
    ".tif",
    ".pfm",
    ".exr",
)


def main():
    params = None
    if USE_TUNING_RESULTS:
        params = import_tuning_param()

    if params is not None:
        # pull parameters from import
        buffer = params['color_buffer']
        max_gap = params['max_gap_px']
        lambda_c = params['lambda_c']
        dx = params['dx_per_px']
        dy = params['dy_per_px']
    else:
        calib = load_default_calibration()
        # calibrate dimensions - units and scale
        if calib is not None:
            dx = calib['dx_per_px']
            dy = calib['dy_per_px']
        else:
            dx = 0.1
            dy = 0.1
        lambda_c = 2  # starting guess, in x-axis units
        buffer = 30
        max_gap = 2

    # assign cropped image
    cropped = import_and_crop(IMAGE_NAME, load_default_area_flag=USE_DEFAULT_AREA)
    # get rgb values of plot color
    rgb = select_plot_color(cropped, load_default_color_flag=USE_DEFAULT_COLOR)
    # extract trend fit to rgb
    x_vals, y_vals = extract_trend(cropped, rgb, buffer=buffer, max_gap=max_gap, interp_gaps=True)
    h, w = cropped.shape[:2]
    # review reformed trend
    plot_scatter(x_vals, y_vals, (h, w))

    # x-axis (spatial) and y-axis (height) can be calibrated to different
    # units/scales - dx only ever feeds spatial-frequency/filtering math,
    # dy only ever feeds the physical height conversion.
    y_phys = y_vals * dy
    # detrend removes linear offset
    y_detrended = detrend_profile(y_phys)
    # waviness = "ideal shape" (no noise, no bumps)
    waviness = compute_waviness(y_detrended, lambda_c, dx)

    # important roughness parameters
    results = compute_roughness_params(y_detrended, waviness)
    Ra = results["Ra"]
    Rq = results["Rq"]
    roughness = results["roughness_profile"]
    # ISO 4288: sampling length = lambda_c, averaged over up to 5 chunks
    Rz = compute_rz(roughness, dx, sampling_length_um=lambda_c)

    # output roughness results to terminal
    print(f"Ra = {Ra:.4f}, Rq = {Rq:.4f}, Rz = {Rz:.4f} (height units per y-axis calibration)")

    # output plots for viewing, good for trial/error
    if PLOT_DIAGNOSTICS == True:
        freqs, mag = diagnostic_spectrum(y_detrended, dx)
        plot_waviness_overlay(y_detrended, waviness, dx)
        plot_roughness_residual(roughness, dx, Ra=Ra, Rq=Rq)
        plot_frequency_scatter(freqs, mag, lambda_c=lambda_c)


def script():
    # run in script mode, assumes all import files exist
    params = None
    params = import_tuning_param()

    if params is not None:
        buffer = params['color_buffer']
        max_gap = params['max_gap_px']
        lambda_c = params['lambda_c']
        dx = params['dx_per_px']
        dy = params['dy_per_px']
    
    else:
        print('Parameters not found. Please calibrate or provide parameters before scripting.')
        return

    # extract all images from data_files
    images = get_image_names()

    names_list = []
    Ra_list = []
    Rq_list = []
    Rz_list = []
    height_list = []

    for image in images:
        # loop through images, calculate roughness, add to table and export
        directory, filename = os.path.split(image)
        filename = filename.split('.')
        filename=filename[0]

        cropped = import_and_crop(image, load_default_area_flag=True)
        rgb = select_plot_color(cropped, load_default_color_flag=True)
        x_vals, y_vals = extract_trend(cropped, rgb, buffer=buffer, max_gap=max_gap, interp_gaps=True)
        h, w = cropped.shape[:2]
        if y_vals is not None and len(y_vals) > 0:
            y_height, y1, y2 = compute_height(x_vals, y_vals)
            y_height = y_height * dy if y_height is not None else -1
        else:
            y_height = -1

        y_phys = y_vals * dy
        y_detrended = detrend_profile(y_phys)
        waviness = compute_waviness(y_detrended, lambda_c, dx)

        results = compute_roughness_params(y_detrended, waviness)
        Ra = results["Ra"]
        Rq = results["Rq"]
        roughness = results["roughness_profile"]

        # ISO 4288: sampling length = lambda_c, averaged over up to 5 chunks
        Rz = compute_rz(roughness, dx, sampling_length_um=lambda_c)

        print(f"Ra = {Ra:.4f}, Rq = {Rq:.4f}, Rz = {Rz:.4f}, Height = {y_height:.4f}")
        Ra_list.append(Ra)        
        Rq_list.append(Rq)
        Rz_list.append(Rz)
        height_list.append(y_height)
        names_list.append(filename)
    
    df = pd.DataFrame({
        'Name': names_list,
        'Ra': Ra_list,
        'Rq': Rq_list,
        'Rz': Rz_list,
        'Height': height_list
    })

    df.to_csv(SCRIPT_OUTPUT_FILENAME, index=False)
    print(f'Data logged to {SCRIPT_OUTPUT_FILENAME}')



def natural_key(text):
    # natural sort: file2 before file10
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]


def get_image_names(IMAGES_FILE_PATH=datafile_dir):
    try:
        image_files = [
            os.path.join(IMAGES_FILE_PATH, f)  # build full file path
            for f in os.listdir(IMAGES_FILE_PATH)  # iterate over files in USB root
            if f.lower().endswith(SUPPORTED_IMAGE_EXTENSIONS)  # filter only images
        ]

    except PermissionError:
        print('Access to file not granted. Make sure there is not an existing presentation in directory with same name as OUTPUT_PPT currently open')
        exit()

    if not image_files:
        print('No images found in specified directory.')
        exit()

    image_files.sort(key=lambda x: natural_key(os.path.basename(x)))

    print(f'Processing {len(image_files)} images...')
    # return list of filenames w/ directory attached
    return image_files


def import_tuning_param(base_dir=base_dir):
    # read turning parameters file, return dict of parameters
    tuning_param = os.path.join(base_dir, TUNED_PARAM_FILENAME)
    try:
        with open(tuning_param, 'r') as file:
            params = json.load(file)
        return params

    except FileNotFoundError:
        return None


def get_tuning():
    # extract tuning parameters and reformat
    params = import_tuning_param()
    if params is None:
        return None
    return {
        'buffer': params['color_buffer'],
        'max_gap': params['max_gap_px'],
        'lambda_c': params['lambda_c'],
        'dx': params['dx_per_px'],
        'dy': params['dy_per_px'],
    }


def plot_scatter(x, y, hxw):
    # scatter image plot reconstruction
    fig, ax = plt.subplots()
    if hxw is not None:
        h, w = hxw
        ax.set_ylim(0, h)
        ax.set_xlim(0, w)
    ax.set_title('Image plot reconstruction')
    ax.scatter(x, y, s=0.1)
    ax.plot(x, y, color='blue', linestyle='-', linewidth=1, label='Connecting Line', zorder=2)

    ax.set_ylabel("y-Pixel Value")
    ax.set_xlabel("x-Pixel Value")

    plt.show()

def plot_frequency_scatter(f, a, lambda_c=None):
    # scatter plot of frequencies - post fft
    fig, ax = plt.subplots()
    ax.set_title('Frequency Plot')
    ax.plot(f, a, color='blue', linewidth=1)
    if lambda_c is not None and lambda_c > 0:
        cutoff_freq = 1 / lambda_c
        ax.axvline(cutoff_freq, color='red', linestyle='--', linewidth=1,
                   label=f'λc cutoff = {cutoff_freq:.3f} (λc={lambda_c})')
        ax.legend()
    ax.set_yscale('log')
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Frequency (1/x-unit)")
    plt.show()


def get_image(image_name, display_plot=False, dir=datafile_dir):
    # find image, export rgb-formatted
    if len(image_name.split('.')) == 0:
        print('Make sure you include filetype (.jpeg,.png, etc.) to end of image_name')
        return

    image_loc = os.path.join(dir, image_name)
    image = cv2.imread(image_loc)

    if image is None:
        raise FileNotFoundError(f'Could not read image at {image_loc} - check path/format.')

    # Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if display_plot == True:
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

    return image_rgb


def show_image(image):
    # this exists because i'm lazy
    plt.imshow(image)
    plt.axis('off')
    plt.show()



def set_figure_bounds(image_rgb, save_path=default_corners_path):
    # interactive setting of figure bounds - allows image to be
    # cropped to plot area, which helps the trend recreation work
    h, w = image_rgb.shape[:2]

    def run_selection():
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ax.imshow(image_rgb)
        ax.axis('off')
        ax.set_title('Select two opposing corners of plot area, then Confirm or Redo')

        border = Rectangle((0, 0), w, h, linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(border)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # inverted, since imshow has row 0 at the top

        state = {"corners": [], "action": None}

        def on_button_press(event):
            if event.button != 1 or event.inaxes != ax:
                return
            if event.xdata is None or event.ydata is None:
                return

            x, y = event.xdata, event.ydata

            if not (0 <= x <= w and 0 <= y <= h):
                print(f"Click ({x:.1f}, {y:.1f}) outside valid image area - ignored.")
                return

            if len(state["corners"]) >= 2:
                return  # extra clicks ignored - use Redo to reselect

            state["corners"].append((x, y))
            ax.plot(x, y, "ro")
            ax.text(x, y, str(len(state["corners"])), color="blue")

            if len(state["corners"]) == 2:
                (x1, y1), (x2, y2) = state["corners"]
                x_min, x_max = sorted([x1, x2])
                y_min, y_max = sorted([y1, y2])
                box = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='lime', facecolor='none')
                ax.add_patch(box)

            fig.canvas.draw_idle()

        def on_scroll(event):
            # allows scroll-wheel zooming
            if event.xdata is None or event.ydata is None:
                return

            base_scale = 1.2
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata, ydata = event.xdata, event.ydata

            if event.button == 'up':
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                return

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])

            ax.set_xlim(xdata - new_width * (1 - relx), xdata + new_width * relx)
            ax.set_ylim(ydata - new_height * (1 - rely), ydata + new_height * rely)
            fig.canvas.draw_idle()

        def on_confirm(_event):
            if len(state["corners"]) != 2:
                print("Select exactly two corners before confirming.")
                return
            state["action"] = "confirm"
            plt.close(fig)

        def on_redo(_event):
            state["action"] = "redo"
            plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", on_button_press)
        fig.canvas.mpl_connect("scroll_event", on_scroll)

        ax_confirm = fig.add_axes([0.3, 0.05, 0.15, 0.075])
        ax_redo = fig.add_axes([0.55, 0.05, 0.15, 0.075])
        btn_confirm = Button(ax_confirm, 'Confirm')
        btn_redo = Button(ax_redo, 'Redo selection')
        btn_confirm.on_clicked(on_confirm)
        btn_redo.on_clicked(on_redo)
        fig._buttons = (btn_confirm, btn_redo)  # keep references alive

        plt.show()
        return state

    state = run_selection()

    if state["action"] == "confirm":
        corner_array = np.array(state["corners"])
        np.savetxt(save_path, corner_array, delimiter=",")
        print(f'Default corners updated to {save_path}')
        return state["corners"]
    else:
        print("Reselecting corners...")
        return set_figure_bounds(image_rgb, save_path=save_path)


def crop_corners(image_rgb, corners):
    # extracts corners and physcially crops image

    (x1, y1), (x2, y2) = corners

    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])

    x_min, x_max = int(round(x_min)), int(round(x_max))
    y_min, y_max = int(round(y_min)), int(round(y_max))

    h, w = image_rgb.shape[:2]
    x_min, x_max = max(0, x_min), min(w, x_max)
    y_min, y_max = max(0, y_min), min(h, y_max)

    cropped = image_rgb[y_min:y_max, x_min:x_max]

    return cropped


def load_default_corners(filepath=default_corners_path):
    # loads corners from file - use for repeated import
    # of same-format images

    if not os.path.isfile(filepath):
        return None

    try:
        corner_array = np.loadtxt(filepath, delimiter=",")
    except (OSError, ValueError):
        return None

    corner_array = np.atleast_2d(corner_array)

    if corner_array.shape != (2, 2):
        return None

    corners = [tuple(row) for row in corner_array]
    return corners


def import_and_crop(image_name, load_default_area_flag=True):
    # function handler for importing and cropping
    image_rgb = get_image(image_name)

    if load_default_area_flag:
        corners = load_default_corners()
        if corners is not None:
            return crop_corners(image_rgb, corners)

    corners = set_figure_bounds(image_rgb)
    return crop_corners(image_rgb, corners)




def load_default_color(filepath=default_color_path):
    # load default plot color (rgb)
    if not os.path.isfile(filepath):
        return None

    try:
        rgb_array = np.loadtxt(filepath, delimiter=",")
    except (OSError, ValueError):
        return None

    rgb_array = np.atleast_1d(rgb_array)

    if rgb_array.shape != (3,):
        return None

    return tuple(int(round(v)) for v in rgb_array)


def select_plot_color(cropped, load_default_color_flag=True, save_path=default_color_path):
    # physical selection of plot color w/ confirmation

    if load_default_color_flag:
        rgb = load_default_color(save_path)
        if rgb is not None:
            print(f"Using saved default color: {rgb}")
            return rgb

    h, w = cropped.shape[:2]

    def run_selection():
        fig = plt.figure(figsize=(8, 5))
        ax_img = fig.add_axes([0.05, 0.15, 0.6, 0.8])
        ax_img.imshow(cropped)
        ax_img.axis('off')
        ax_img.set_title('Select plot curve to preview color')

        border = Rectangle((0, 0), w, h, linewidth=1, edgecolor='black', facecolor='none')
        ax_img.add_patch(border)
        ax_img.set_xlim(0, w)
        ax_img.set_ylim(h, 0)

        ax_swatch = fig.add_axes([0.7, 0.55, 0.25, 0.35])
        ax_swatch.axis('off')
        swatch_im = ax_swatch.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
        ax_swatch.set_title('No color selected yet')

        state = {"rgb": None, "action": None}

        def on_button_press(event):
            if event.button != 1 or event.inaxes != ax_img:
                return
            if event.xdata is None or event.ydata is None:
                return

            x, y = event.xdata, event.ydata

            if not (0 <= x <= w and 0 <= y <= h):
                print(f"Click ({x:.1f}, {y:.1f}) outside valid image area - ignored.")
                return

            px = min(max(int(round(x)), 0), w - 1)
            py = min(max(int(round(y)), 0), h - 1)

            rgb = tuple(int(v) for v in cropped[py, px])
            state["rgb"] = rgb
            print(f"Pixel ({px}, {py}) -> RGB: {rgb}")

            swatch_im.set_data(np.full((10, 10, 3), rgb, dtype=np.uint8))
            ax_swatch.set_title(f"RGB: {rgb}")
            fig.canvas.draw_idle()

        def on_confirm(_event):
            if state["rgb"] is None:
                print("Click a pixel to select a color before confirming.")
                return
            state["action"] = "confirm"
            plt.close(fig)

        def on_redo(_event):
            state["action"] = "redo"
            plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", on_button_press)

        ax_confirm = fig.add_axes([0.7, 0.35, 0.25, 0.1])
        ax_redo = fig.add_axes([0.7, 0.2, 0.25, 0.1])
        btn_confirm = Button(ax_confirm, 'Confirm')
        btn_redo = Button(ax_redo, 'Redo selection')
        btn_confirm.on_clicked(on_confirm)
        btn_redo.on_clicked(on_redo)
        fig._buttons = (btn_confirm, btn_redo)

        plt.show()
        return state

    state = run_selection()

    if state["action"] == "confirm":
        np.savetxt(save_path, np.array(state["rgb"]), delimiter=",")
        print(f'Default color updated to {save_path}')
        return state["rgb"]
    else:
        print("Reselecting color...")
        return select_plot_color(cropped, load_default_color_flag=False, save_path=save_path)



def load_default_calibration(filepath=default_calibration_path):
    # load default sizing and unit calibration
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        required = {'dx_per_px', 'unit_x', 'dy_per_px', 'unit_y'}
        if not required.issubset(data):
            return None
        return data
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def save_default_calibration(dx_per_px, unit_x, dy_per_px, unit_y, filepath=default_calibration_path):
    # save calibration settings
    with open(filepath, 'w') as f:
        json.dump({
            'dx_per_px': dx_per_px,
            'unit_x': unit_x,
            'dy_per_px': dy_per_px,
            'unit_y': unit_y,
        }, f, indent=2)


def pick_calibration_points(image_rgb, axis_label="axis"):
    """Popup click-selector with in-window Confirm/Redo buttons: collects 2
    points and returns the pixel (Euclidean) distance between them.
    axis_label is cosmetic (shown in the window title) - e.g. "X" or "Y"."""
    h, w = image_rgb.shape[:2]

    def run_selection():
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        ax.imshow(image_rgb)
        ax.axis('off')
        ax.set_title(f'Click two points spanning a known {axis_label}-axis distance')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        state = {"points": [], "action": None}

        def on_click(event):
            if event.button != 1 or event.inaxes != ax:
                return
            if event.xdata is None or event.ydata is None:
                return
            x, y = event.xdata, event.ydata
            if not (0 <= x <= w and 0 <= y <= h):
                print(f"Click ({x:.1f}, {y:.1f}) outside valid image area - ignored.")
                return
            if len(state["points"]) >= 2:
                return
            state["points"].append((x, y))
            ax.plot(x, y, 'ro')
            ax.text(x, y, str(len(state["points"])), color='blue')
            if len(state["points"]) == 2:
                (x1, y1), (x2, y2) = state["points"]
                ax.plot([x1, x2], [y1, y2], color='lime', linewidth=1)
            fig.canvas.draw_idle()

        def on_confirm(_event):
            if len(state["points"]) != 2:
                print("Select exactly two points before confirming.")
                return
            state["action"] = "confirm"
            plt.close(fig)

        def on_redo(_event):
            state["action"] = "redo"
            plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_click)

        ax_confirm = fig.add_axes([0.3, 0.05, 0.15, 0.075])
        ax_redo = fig.add_axes([0.55, 0.05, 0.15, 0.075])
        btn_confirm = Button(ax_confirm, 'Confirm')
        btn_redo = Button(ax_redo, 'Redo selection')
        btn_confirm.on_clicked(on_confirm)
        btn_redo.on_clicked(on_redo)
        fig._buttons = (btn_confirm, btn_redo)

        plt.show()
        return state

    state = run_selection()

    if state["action"] == "confirm":
        (x1, y1), (x2, y2) = state["points"]
        return float(np.hypot(x2 - x1, y2 - y1))
    else:
        print("Reselecting calibration points...")
        return pick_calibration_points(image_rgb, axis_label=axis_label)


def largest_contiguous_run(matched_y, matched_dist, max_gap):
    """Given matched row indices (sorted ascending) and their distances,
    return only the values belonging to the largest contiguous run.
    A run continues as long as consecutive matched_y values are within
    `max_gap` of each other (allows tiny 1-2px internal gaps from
    antialiasing without breaking the run)."""

    if len(matched_y) == 0:
        return matched_y, matched_dist

    breaks = np.where(np.diff(matched_y) > max_gap)[0]
    run_starts = np.concatenate(([0], breaks + 1))
    run_ends = np.concatenate((breaks, [len(matched_y) - 1]))

    run_lengths = run_ends - run_starts + 1
    best = np.argmax(run_lengths)

    s, e = run_starts[best], run_ends[best]
    return matched_y[s:e + 1], matched_dist[s:e + 1]

def extract_trend(image, plot_rgb, buffer=30, max_gap=2, interp_gaps=True):
    # extract trend from image given buffer for allowed rgb tolerance -
    # assigns values based on intensity centroid if multiple values
    # occur within a px column
    x_vals, y_vals, _info = extract_trend_with_info(
        image, plot_rgb, buffer=buffer, max_gap=max_gap, interp_gaps=interp_gaps)
    return x_vals, y_vals


def extract_trend_with_info(image, plot_rgb, buffer=30, max_gap=2, interp_gaps=True):
    """
    Same extraction as extract_trend, but also returns an info dict the
    batch pipeline uses to score extraction quality:
        skipped_x     : list of column indices with no color match
        n_columns     : total column count of the crop
        max_gap_run   : longest consecutive run of skipped columns
        crop_height   : crop height in px (for normalizing jumpiness)
    """
    h, w = image.shape[:2]
    plot_rgb = np.array(plot_rgb, dtype=np.float64)

    diff = image.astype(np.float64) - plot_rgb
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # shape (h, w)

    y_indices = np.arange(h)

    x_vals = []
    y_vals = []
    skipped_x = []

    for x in range(w):
        col_dist = dist[:, x]
        mask = col_dist <= buffer

        if not np.any(mask):
            skipped_x.append(x)
            continue

        matched_y = y_indices[mask]
        matched_dist = col_dist[mask]

        matched_y, matched_dist = largest_contiguous_run(matched_y, matched_dist, max_gap=max_gap)

        weights = buffer - matched_dist
        weights = np.clip(weights, a_min=1e-6, a_max=None)

        centroid_y = np.sum(matched_y * weights) / np.sum(weights)

        x_vals.append(x)
        y_vals.append(centroid_y)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    if skipped_x:
        print(f"Note: {len(skipped_x)}/{w} columns had no color match "
              f"({'interpolated' if interp_gaps else 'left blank'}).")

    if interp_gaps and len(skipped_x) > 0 and len(x_vals) > 1:
        full_x = np.arange(w)
        y_vals = np.interp(full_x, x_vals, y_vals)
        x_vals = full_x

    # longest consecutive run of skipped columns
    max_gap_run = 0
    run = 0
    prev = None
    for sx in skipped_x:
        run = run + 1 if (prev is not None and sx == prev + 1) else 1
        max_gap_run = max(max_gap_run, run)
        prev = sx

    info = {
        "skipped_x": skipped_x,
        "n_columns": w,
        "max_gap_run": max_gap_run,
        "crop_height": h,
    }
    return x_vals, (h - 1) - y_vals, info


def lambda_c_to_sigma(lambda_c, dx):
    """
    lambda_c : cutoff wavelength in x-axis units
    dx       : sample spacing in the SAME x-axis units (after calibration)
    Returns sigma in samples, for use with gaussian_filter1d.

    ISO 16610-21 defines the Gaussian weighting function
        s(x) = 1/(alpha*lambda_c) * exp(-pi * (x / (alpha*lambda_c))^2)
    with alpha = sqrt(ln2/pi) = 0.4697, chosen so amplitude transmission at
    lambda_c is exactly 50%. That kernel's standard deviation is
    alpha*lambda_c/sqrt(2*pi), NOT alpha*lambda_c — using alpha*lambda_c
    directly as sigma over-smooths by a factor of sqrt(2*pi) (~2.5x) and
    moves the true 50% cutoff to ~2.5*lambda_c.
    """
    alpha = 0.4697  # ISO 16610-21 constant for the Gaussian weighting function
    sigma_physical = alpha * lambda_c / np.sqrt(2.0 * np.pi)
    sigma_samples = sigma_physical / dx
    return sigma_samples

def detrend_profile(y):
    # removes linear trend from provile
    return signal.detrend(y, type='linear')

def compute_waviness(y, lambda_c, dx, mode='reflect'):
    # applies filter and returns waviness - "ideal structure"
    sigma = lambda_c_to_sigma(lambda_c, dx)
    waviness = ndimage.gaussian_filter1d(y, sigma=sigma, mode=mode)
    return waviness


def plot_waviness_overlay(y_detrended, waviness, dx, x_label="Position (x-unit)"):
    # overlays waviness on raw trend
    x = np.arange(len(y_detrended)) * dx
    fig, ax = plt.subplots()
    ax.plot(x, y_detrended, color='gray', linewidth=0.7, alpha=0.7, label='Raw (detrended) profile')
    ax.plot(x, waviness, color='red', linewidth=1.5, label='Waviness (form)')
    ax.set_title('Waviness fit over raw profile')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Height (y-unit)')
    ax.legend()
    plt.show()

def compute_roughness_params(y, waviness):
    # calculates some roughness parameters
    roughness = y - waviness # removes main structure - what is left is roughness

    Ra = np.mean(np.abs(roughness))
    Rq = np.sqrt(np.mean(roughness**2))

    return {
        "roughness_profile": roughness,
        "Ra": Ra,
        "Rq": Rq,
    }

def plot_roughness_residual(roughness, dx, Ra=None, Rq=None):
    # plots just roughness from trend - no structure
    x = np.arange(len(roughness)) * dx
    fig, ax = plt.subplots()
    ax.plot(x, roughness, color='steelblue', linewidth=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    if Ra is not None:
        ax.axhline(Ra, color='green', linestyle='--', linewidth=0.8, label=f'Ra = {Ra:.3f}')
        ax.axhline(-Ra, color='green', linestyle='--', linewidth=0.8)
    if Rq is not None:
        ax.axhline(Rq, color='orange', linestyle=':', linewidth=0.8, label=f'Rq = {Rq:.3f}')
        ax.axhline(-Rq, color='orange', linestyle=':', linewidth=0.8)
    ax.set_title('Roughness residual')
    ax.set_xlabel('Position (x-unit)')
    ax.set_ylabel('Deviation (y-unit)')
    ax.legend()
    plt.show()


def compute_rz(roughness, dx, sampling_length_um, max_chunks=5):
    """
    Rz per ISO 4287/4288: the profile is divided into chunks one SAMPLING
    LENGTH long (the sampling length equals lambda_c), the peak-to-valley
    height is taken within each chunk, and Rz is the mean over (up to) 5
    chunks — the standard evaluation length is 5 sampling lengths.
    `sampling_length_um` is named for historical reasons - the value should
    be lambda_c in the same x-axis units as dx.

    If the profile is too short to fit even one chunk at the requested
    sampling length, falls back to a single chunk over the full profile
    instead of raising - keeps batch runs from dying on short/tightly-
    cropped images. If fewer than `max_chunks` fit, the available chunks
    are averaged (shorter-than-standard evaluation length).
    """
    chunk_size = int(round(sampling_length_um / dx))
    if chunk_size < 2:
        raise ValueError("sampling_length_um too small relative to dx")

    n_chunks = len(roughness) // chunk_size
    if n_chunks == 0:
        chunk_size = len(roughness)
        n_chunks = 1
    n_chunks = min(n_chunks, max_chunks)

    trimmed = roughness[: n_chunks * chunk_size]
    chunks = np.array_split(trimmed, n_chunks)

    pv_heights = [chunk.max() - chunk.min() for chunk in chunks]
    return np.mean(pv_heights)


def diagnostic_spectrum(y, dx):
    # gives full frequency spectrum, applies hanning window
    y_d = detrend_profile(y)
    windowed = y_d * np.hanning(len(y_d))

    Y = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=dx)

    return freqs, np.abs(Y)



def compute_height(x_values, y_values, tolerance=5):
    """
    Crude two-level step height: cluster the y-values and take the distance
    between the two most-populated cluster centers.

    Works on a SORTED COPY of y_values — the caller's array is left in its
    original spatial order (an in-place sort here used to silently destroy
    the profile for every computation downstream of the call).

    Returns (height, y1, y2), or (None, None, None) if fewer than two
    clusters exist (e.g. a flat profile), so callers can unpack safely.

    NOTE: superseded by feature_analysis.analyze_features for the batch
    pipeline — kept for the legacy script()/tuning workflow.
    """
    y_sorted = np.sort(np.asarray(y_values, dtype=float))

    # Merge nearby y-values into clusters
    clusters = []

    for y in y_sorted:
        if not clusters:
            clusters.append({
                "center": y,
                "count": 1,
                "members": [y]
            })
            continue

        if abs(y - clusters[-1]["center"]) <= tolerance:
            clusters[-1]["members"].append(y)
            clusters[-1]["count"] += 1
            clusters[-1]["center"] = (
                sum(clusters[-1]["members"]) /
                len(clusters[-1]["members"])
            )
        else:
            clusters.append({
                "center": y,
                "count": 1,
                "members": [y]
            })

    # Sort by occurrence count
    clusters.sort(key=lambda c: c["count"], reverse=True)

    if len(clusters) < 2:
        return None, None, None

    peak1 = clusters[0]
    peak2 = clusters[1]

    y1 = peak1["center"]
    y2 = peak2["center"]

    height = abs(y1 - y2)

    return height, y1, y2




if __name__ == "__main__":
    # calls everythin on run

    if script_mode==True:
        script()
    
    else:        
        main()
