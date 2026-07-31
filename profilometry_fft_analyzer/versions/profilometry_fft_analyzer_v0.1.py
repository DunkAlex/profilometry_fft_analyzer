import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage, signal
import json


base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)

datafile_dir = os.path.join(base_dir, r'data_files')
default_corners_path = os.path.join(base_dir, 'default_corners.csv')
default_colors_path = os.path.join(base_dir, 'default_colors.csv')

IMAGE_NAME = 'example_profilometry_data.ppm'
USE_TUNING_RESULTS = True
UM_2_PX = 10
PLOT_DIAGNOSTICS = False

def main():
    params = None
    if USE_TUNING_RESULTS == False:
            params = import_tuning_param()

    if params is not None:
        buffer=params['color_buffer']
        max_gap=params['max_gap_px']
        lambda_c=params['lambda_c_um']
        dx=params['dx_um_per_px']

    else:
        dx=0.1
        lambda_c = 2  # starting guess in um - this is your GUI slider value
        buffer=30
        max_gap=2

    cropped = import_and_crop(IMAGE_NAME, load_default_area_flag=False)
    rgb = select_plot_color(cropped, load_default_color_flag=False)
    x_vals, y_vals = extract_trend(cropped, rgb, buffer=buffer, max_gap=max_gap, interp_gaps=True)
    h, w = cropped.shape[:2]
    plot_scatter(x_vals, y_vals, (h, w))

    y_um = y_vals/UM_2_PX
    y_detrended = detrend_profile(y_um)
    waviness = compute_waviness(y_detrended, lambda_c, dx)

    results = compute_roughness_params(y_detrended, waviness)
    Ra = results["Ra"]
    Rq = results["Rq"]
    roughness = results["roughness_profile"]

    Rz = compute_rz(roughness, dx, sampling_length_um=lambda_c * 5)  # ISO default: 5x lambda_c

    print(f"Ra = {Ra:.4f} um, Rq = {Rq:.4f} um, Rz = {Rz:.4f} um")

    if PLOT_DIAGNOSTICS == True:
        # For picking a good lambda_c range interactively:
        freqs, mag = diagnostic_spectrum(y_detrended, dx)
        plot_waviness_overlay(y_detrended, waviness, dx)
        plot_roughness_residual(roughness, dx, Ra=Ra, Rq=Rq)
        plot_frequency_scatter(freqs, mag, lambda_c=lambda_c)


def import_tuning_param(base_dir=base_dir):
    tuning_param = os.path.join(base_dir, 'tuned_params.json')
    try:
        with open(tuning_param, 'r') as file:
            params = json.load(file)
        return params
    
    except FileNotFoundError:
        return None


def get_tuning():
    params = import_tuning_param()
    if params is not None:
        buffer=params['color_buffer']
        max_gap=params['max_gap_px']
        lambda_c=params['lambda_c_um']
        dx=params['dx_um_per_px']

    else:
        buffer=params['color_buffer']
        max_gap=params['max_gap_px']
        lambda_c=params['lambda_c_um']
        dx=params['dx_um_per_px']

def plot_scatter(x, y, hxw):
    fig, ax = plt.subplots()
    if hxw is not None:
        h, w = hxw
        ax.set_ylim(0,h)
        ax.set_xlim(0,w)
    ax.set_title('Image plot reconstruction')
    ax.scatter(x,y, s=0.1)
    ax.plot(x, y, color='blue', linestyle='-', linewidth=1, label='Connecting Line', zorder=2)

    ax.set_ylabel("y-Pixel Value")
    ax.set_xlabel("x-Pixel Value")

    plt.show()

def plot_frequency_scatter(f, a, lambda_c=None):
    fig, ax = plt.subplots()
    ax.set_title('Frequency Plot')
    ax.plot(f, a, color='blue', linewidth=1)
    if lambda_c is not None and lambda_c > 0:
        cutoff_freq = 1 / lambda_c
        ax.axvline(cutoff_freq, color='red', linestyle='--', linewidth=1,
                   label=f'λc cutoff = {cutoff_freq:.3f} 1/um (λc={lambda_c} um)')
        ax.legend()
    ax.set_yscale('log')
    ax.set_ylabel("Magnitude")
    ax.set_xlabel("Frequency (1/um)")
    plt.show()


def get_image(image_name, display_plot=False):
    image_loc = os.path.join(datafile_dir, image_name)
    image = cv2.imread(image_loc)

    if image is None:
        raise FileNotFoundError(f'Could not read image at {image_loc} - check path/format.')

    # Convert from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if display_plot == True:
        # Display the image using Matplotlib
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.show()

    return image_rgb


def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def set_figure_bounds(image_rgb, save_path=default_corners_path):

    h, w = image_rgb.shape[:2]

    def get_two_clicks():
        """Displays the image and collects exactly two corner clicks.
        Left-click selects corners. Right-click-drag pans. Scroll wheel zooms.
        """
        fig, ax = plt.subplots()
        ax.imshow(image_rgb)
        ax.axis('off')
        ax.set_title(
            'Select two opposing corners of plot area'
        )

        border = Rectangle(
            (0, 0), w, h,
            linewidth=1,
            edgecolor='black',
            facecolor='none'
        )
        ax.add_patch(border)

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # inverted, since imshow has row 0 at the top

        corners = []
        pan_state = {"active": False, "x0": None, "y0": None,
                     "xlim0": None, "ylim0": None}

        def on_button_press(event):
            if event.button == 1:
                # Left click - select a corner
                print('Corner click detected')

                if event.xdata is None or event.ydata is None:
                    return

                x, y = event.xdata, event.ydata

                if not (0 <= x <= w and 0 <= y <= h):
                    print(f"Click ({x:.1f}, {y:.1f}) outside valid image area - ignored.")
                    return

                corners.append((x, y))

                ax.plot(x, y, "ro")
                ax.text(x, y, str(len(corners)), color="blue")
                fig.canvas.draw()

                if len(corners) == 2:
                    plt.close(fig)

        def on_scroll(event):
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

        fig.canvas.mpl_connect("button_press_event", on_button_press)
        fig.canvas.mpl_connect("scroll_event", on_scroll)

        plt.show()

        if len(corners) != 2:
            raise ValueError(
                f"Expected 2 corner clicks, got {len(corners)}. "
                "Make sure to click exactly two opposing corners before closing the window."
            )

        return corners

    def confirm_bounds(corners):
        """Displays the selected crop box for review and asks the user to confirm."""
        (x1, y1), (x2, y2) = corners
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])

        fig, ax = plt.subplots()
        ax.imshow(image_rgb)
        ax.axis('off')
        ax.set_title('Confirm selected bounds (check terminal)')

        selected_box = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor='lime',
            facecolor='none'
        )
        ax.add_patch(selected_box)

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)

        plt.show(block=False)
        plt.pause(0.1)

        while True:
            response = input("Confirm these bounds? (y/n): ").strip().lower()
            if response in ("y", "n"):
                break
            print("Please type 'y' or 'n'.")

        plt.close(fig)
        return response == "y"

    # --- Recursive selection loop ---
    corners = get_two_clicks()

    if confirm_bounds(corners):
        corner_array = np.array(corners)
        np.savetxt(save_path, corner_array, delimiter=",")
        print(f'Default corners updated to {save_path}')
        return corners
    else:
        print("Bounds rejected - please reselect corners.")
        return set_figure_bounds(image_rgb, save_path=save_path)


def crop_corners(image_rgb, corners):

    (x1, y1), (x2, y2) = corners

    x_min, x_max = sorted([x1, x2])
    y_min, y_max = sorted([y1, y2])

    x_min, x_max = int(round(x_min)), int(round(x_max))
    y_min, y_max = int(round(y_min)), int(round(y_max))

    # clip to image bounds just in case a click landed slightly outside
    h, w = image_rgb.shape[:2]
    x_min, x_max = max(0, x_min), min(w, x_max)
    y_min, y_max = max(0, y_min), min(h, y_max)

    cropped = image_rgb[y_min:y_max, x_min:x_max]

    return cropped


def load_default_corners(filepath=default_corners_path):

    if not os.path.isfile(filepath):
        return None

    try:
        corner_array = np.loadtxt(filepath, delimiter=",")
    except (OSError, ValueError):
        # file exists but is empty, malformed, or unreadable
        return None

    # loadtxt can return a 1D array if there's only one row - normalize shape
    corner_array = np.atleast_2d(corner_array)

    if corner_array.shape != (2, 2):
        # doesn't have exactly two (x, y) points
        return None

    corners = [tuple(row) for row in corner_array]
    return corners


def import_and_crop(image_name, load_default_area_flag=True):
    image_rgb = get_image(image_name)

    if load_default_area_flag:
        corners = load_default_corners()
        if corners is not None:
            return crop_corners(image_rgb, corners)

    corners = set_figure_bounds(image_rgb)
    return crop_corners(image_rgb, corners)


default_color_path = os.path.join(base_dir, "default_color.csv")


def load_default_color(filepath=default_color_path):
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

    if load_default_color_flag:
        rgb = load_default_color(save_path)
        if rgb is not None:
            print(f"Using saved default color: {rgb}")
            return rgb

    h, w = cropped.shape[:2]

    fig, ax = plt.subplots()
    ax.imshow(cropped)
    ax.axis('off')
    ax.set_title('Select plot color (left-click)')

    border = Rectangle(
        (0, 0), w, h,
        linewidth=1,
        edgecolor='black',
        facecolor='none'
    )
    ax.add_patch(border)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    result = {"rgb": None}

    def show_swatch_and_confirm(rgb):
        swatch_fig, swatch_ax = plt.subplots(figsize=(2, 2))
        swatch_ax.imshow(np.full((10, 10, 3), rgb, dtype=np.uint8))
        swatch_ax.axis('off')
        swatch_ax.set_title(f"RGB: {tuple(rgb)}")

        plt.show(block=False)
        plt.pause(0.1)

        while True:
            response = input("Is this the correct pixel/color? (y/n): ").strip().lower()
            if response in ("y", "n"):
                break
            print("Please type 'y' or 'n'.")

        plt.close(swatch_fig)
        return response == "y"

    def on_button_press(event):
        if event.button != 1:
            return

        print('Color click detected')

        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata

        if not (0 <= x <= w and 0 <= y <= h):
            print(f"Click ({x:.1f}, {y:.1f}) outside valid image area - ignored.")
            return

        px = min(max(int(round(x)), 0), w - 1)
        py = min(max(int(round(y)), 0), h - 1)

        rgb = cropped[py, px]

        rgb_val = tuple(rgb)
        int_rgb = [int(val) for val in rgb_val]
        print(f"Pixel ({px}, {py}) -> RGB: {int_rgb}")

        if show_swatch_and_confirm(rgb):
            result["rgb"] = tuple(rgb)
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_button_press)

    plt.show()

    if result["rgb"] is not None:
        # Save as the new default now that it's been confirmed
        np.savetxt(save_path, np.array(result["rgb"]), delimiter=",")
        print(f'Default color updated to {save_path}')

        return result["rgb"]
    else:
        print("No color confirmed - please reselect.")
        return select_plot_color(cropped, load_default_color_flag=False, save_path=save_path)

def largest_contiguous_run(matched_y, matched_dist, max_gap):
    """Given matched row indices (sorted ascending) and their distances,
    return only the values belonging to the largest contiguous run.
    A run continues as long as consecutive matched_y values are within
    `max_gap` of each other (allows tiny 1-2px internal gaps from
    antialiasing without breaking the run)."""
 
    if len(matched_y) == 0:
        return matched_y, matched_dist
 
    # matched_y comes from np.arange indexing with a boolean mask, so it's
    # already sorted ascending - safe to diff directly.
    breaks = np.where(np.diff(matched_y) > max_gap)[0]
    run_starts = np.concatenate(([0], breaks + 1))
    run_ends = np.concatenate((breaks, [len(matched_y) - 1]))
 
    run_lengths = run_ends - run_starts + 1
    best = np.argmax(run_lengths)
 
    s, e = run_starts[best], run_ends[best]
    return matched_y[s:e + 1], matched_dist[s:e + 1]

def extract_trend(image, plot_rgb, buffer=30, max_gap=2, interp_gaps=True):

    h, w = image.shape[:2]
    plot_rgb = np.array(plot_rgb, dtype=np.float64)

    # Per-pixel Euclidean distance in RGB space to the target color
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
            continue  # no matching pixels in this column - skip it

        matched_y = y_indices[mask]
        matched_dist = col_dist[mask]

        matched_y, matched_dist = largest_contiguous_run(matched_y, matched_dist, max_gap=max_gap)

        # Intensity weighting: closer color matches contribute more.
        # weight ranges from ~0 (at the edge of the buffer) up to `buffer`
        # (a perfect color match), so it behaves like a soft, distance-based mask.
        weights = buffer - matched_dist
        weights = np.clip(weights, a_min=1e-6, a_max=None)  # avoid divide-by-zero

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

    return x_vals, (h - 1) - y_vals


def lambda_c_to_sigma(lambda_c, dx):
    """
    lambda_c : cutoff wavelength in physical units (um)
    dx       : sample spacing in the SAME physical units (um/pixel after calibration)
    Returns sigma in samples, for use with gaussian_filter1d.
    """
    alpha = 0.4697  # ISO 16610-21 constant for the Gaussian weighting function
    sigma_physical = alpha * lambda_c        # sigma in um
    sigma_samples = sigma_physical / dx      # convert to samples for scipy
    return sigma_samples

def detrend_profile(y):
    return signal.detrend(y, type='linear')

def compute_waviness(y, lambda_c, dx, mode='reflect'):
    sigma = lambda_c_to_sigma(lambda_c, dx)
    waviness = ndimage.gaussian_filter1d(y, sigma=sigma, mode=mode)
    return waviness


def plot_waviness_overlay(y_detrended, waviness, dx, x_label="Position (um)"):
    x = np.arange(len(y_detrended)) * dx
    fig, ax = plt.subplots()
    ax.plot(x, y_detrended, color='gray', linewidth=0.7, alpha=0.7, label='Raw (detrended) profile')
    ax.plot(x, waviness, color='red', linewidth=1.5, label='Waviness (form)')
    ax.set_title('Waviness fit over raw profile')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Height (um)')
    ax.legend()
    plt.show()

def compute_roughness_params(y, waviness):
    roughness = y - waviness

    Ra = np.mean(np.abs(roughness))
    Rq = np.sqrt(np.mean(roughness**2))

    return {
        "roughness_profile": roughness,
        "Ra": Ra,
        "Rq": Rq,
    }

def plot_roughness_residual(roughness, dx, Ra=None, Rq=None):
    x = np.arange(len(roughness)) * dx
    fig, ax = plt.subplots()
    ax.plot(x, roughness, color='steelblue', linewidth=0.6)
    ax.axhline(0, color='black', linewidth=0.5)
    if Ra is not None:
        ax.axhline(Ra, color='green', linestyle='--', linewidth=0.8, label=f'Ra = {Ra:.3f} um')
        ax.axhline(-Ra, color='green', linestyle='--', linewidth=0.8)
    if Rq is not None:
        ax.axhline(Rq, color='orange', linestyle=':', linewidth=0.8, label=f'Rq = {Rq:.3f} um')
        ax.axhline(-Rq, color='orange', linestyle=':', linewidth=0.8)
    ax.set_title('Roughness residual')
    ax.set_xlabel('Position (um)')
    ax.set_ylabel('Deviation (um)')
    ax.legend()
    plt.show()


def compute_rz(roughness, dx, sampling_length_um):
    """
    Rz: average peak-to-valley height across chunks of a given
    physical sampling length (per ISO convention, typically 5 chunks averaged).

    If the profile is too short to fit even one chunk at the requested
    sampling length, falls back to a single chunk over the full profile
    (with a warning) instead of raising - keeps batch runs from dying on
    short/tightly-cropped images.
    """
    chunk_size = int(round(sampling_length_um / dx))
    if chunk_size < 2:
        raise ValueError("sampling_length_um too small relative to dx")

    profile_len_um = len(roughness) * dx

    n_chunks = len(roughness) // chunk_size
    if n_chunks == 0:
        # print(f"Warning: requested sampling_length_um={sampling_length_um:.2f} "
        #       f"exceeds profile length ({profile_len_um:.2f} um). "
        #       f"Falling back to a single chunk over the full profile - "
        #       f"Rz will be less statistically robust than the ISO 5-chunk average.")
        chunk_size = len(roughness)
        n_chunks = 1

    trimmed = roughness[: n_chunks * chunk_size]
    chunks = np.array_split(trimmed, n_chunks)

    pv_heights = [chunk.max() - chunk.min() for chunk in chunks]
    return np.mean(pv_heights)


def diagnostic_spectrum(y, dx):
    y_d = detrend_profile(y)
    windowed = y_d * np.hanning(len(y_d))

    Y = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=dx)  # cycles per um

    return freqs, np.abs(Y)



if __name__ == "__main__":
    main()