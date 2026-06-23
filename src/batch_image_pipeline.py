import os
import sys

# Get the path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Add it to the Python search path if it isn't there already
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.cvp_batch_utils import run_batch_cvp

if __name__ == "__main__":
    # folder paths
    image_folder = r".\data\license_plate_detection\test\images"

    # labels
    labels = [
        "car",
        "truck",
        "bus",
        "vehicle",
        "motorcycle",
        "no vehicle",
        "only license plate",
    ]

    # function to run batches
    run_batch_cvp(labels, image_folder)
