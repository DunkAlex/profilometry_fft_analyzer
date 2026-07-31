import os
from plot_calibration import calibrate_image

IMAGE_NAME = 'example_profile1.jpeg'

# set base directory for file access
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f'Current directory: {base_dir}')

# define I/O directories
datafile_dir = os.path.join(base_dir, r'example_data')

from profilometry_fft_analyzer import get_image

img = get_image(IMAGE_NAME, dir=datafile_dir)

result = calibrate_image(
    img,
    image_filename=IMAGE_NAME,
    profiles_path=os.path.join(base_dir, "calibration_profiles.json"),
    on_no_match="gui",   # or "return_none" for batch, "raise" to hard-fail
)

if result is None:
    print("Calibration failed — see logs.")
else:
    print(f"X: {result.x_px_per_unit:.3f} px/unit  (confidence {result.x_confidence:.2f})")
    print(f"Y: {result.y_px_per_unit:.3f} px/unit  (confidence {result.y_confidence:.2f})")
    print(f"Profile: {result.profile_used}")
    # world value at a pixel: value = slope * pixel + intercept
    