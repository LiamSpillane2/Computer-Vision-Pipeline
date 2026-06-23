import os
import sys

# Get the path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# Add it to the Python search path if it isn't there already
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.cvp_batch_utils import run_batch_cvp

if __name__ == "__main__":
    
    # Adjusting function variables for batch wise weld integration test
    image_folder = r".\data\weld_data\test\images"
    yolo_model_weights = r'.\models\yolo_weld\detect\train-4\weights\best.pt'
    output_path = r'.\data\yolo_weld_pipeline.csv'
    yolo_conf = 0.25
    labels = ['Pore/Porosity', 
              'Inclusion', 
              'Undercut', 
              'Burn-through', 
              'Crack', 
              'Overlap', 
              'Reference Standard 1', 
              'Reference Standard 2', 
              'Reference Standard 3', 
              'Hidden Porosity', 
              'Shrinkage Depression', 
              'Lack of Fusion', 
              'Incomplete Root Penetration']

    # function to test run batches of the test images
    run_batch_cvp(labels = labels, image_folder = image_folder, do_zs = False, do_ocr = False, yolo_model_weights = yolo_model_weights, output_path = output_path, yolo_conf = yolo_conf)

