import os
import cv2
import sys
import pandas as pd
from fast_alpr import ALPR
from ultralytics import YOLO
from PIL import Image
import tqdm

# Get the path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Add it to the Python search path if it isn't there already
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.ocr_utils import alpr_single_image
from utils.zero_shot_utils import download_clip


def run_model_batch(paths, model, conf=0.25):
    """Run batches of file through YOLO model

    Args:
        -paths: list of file paths to images
        -model: intitialized YOLO model
        -conf: confidence threshold

    Returns:
        -nested list of xy bounding box coordinates, class ID and confidence estimate

    """
    # batch results from yolo model
    results = model.predict(source=paths, save=False, conf=conf, verbose=False)

    out = []

    # loop through image results
    for result in results:
        # return None if no result detected
        if len(result.boxes) == 0:
            out.append((None, None, None))
            continue

        # otherwise return xy coords, class ID and confidence estimate
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
        out.append((xyxy, cls_id, confidence))
    return out


def clip_batch(pil_images, label_list, clip_model, batch_size=16):
    """Runs batch of images through CLIP model.

    Args:
        -pil images: directory of YOLO annotations
        -label_list: list of labels to be analyzed in clip model
        -clip_model: initialize clip model
        -batch_size: batch size of images

    Returns:
        -dataframe with prediction values for each image in folder

    """

    # get batch results from clip_model
    clip_results = clip_model(
        pil_images, candidate_labels=label_list, batch_size=batch_size
    )

    dfs = []
    # loop through clip results and format each result
    for preds in clip_results:
        df = pd.DataFrame(preds).T
        df.columns = df.iloc[-1]
        dfs.append(df[:-1])
    return dfs


def run_batch_cvp(file_list, labels, output_path, image_folder, batch_size=16):
    """Runs files through CVP. Performs bounding box
    prediction, zero shot classification and ocr text extraction

    Args:
        -file_list: list of files to be analyzed
        -labels: list of labels to be analyzed in clip model
        -output_path: file path to save outputs
        -image_folder: folder where images are stored
        -batch_size: batch size of images to process

    Returns:
        -csv file with license plate bounding boxes, extracted text and zero shot probabilities
    """
    # initialize models
    yolo_model = YOLO(r"..\models\Yolo26Model\weights\best.pt")
    alpr_model = ALPR()
    clip_model = download_clip()

    master_list = []

    # define batches based on batch size
    batches = [
        file_list[i : i + batch_size] for i in range(0, len(file_list), batch_size)
    ]

    # loop through batches
    for idx, batch in enumerate(tqdm.tqdm(batches, desc="running batches...")):

        # define path and file names
        paths = [os.path.join(image_folder, f) for f in batch]
        names = [os.path.splitext(f)[0] for f in batch]

        # list of cv2 image objects (assumed to never be None)
        bgr_images = [cv2.imread(path) for path in paths]

        # list of pil image objects
        pil_images = [
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in bgr_images
        ]

        # yolo predictions for batch of images
        yolo_out = run_model_batch(paths, model=yolo_model)

        # clip predictions for batch of images
        clip_dfs = clip_batch(pil_images, labels, clip_model, batch_size)

        # loop through batch of images and append ocr results to list
        alpr_results = []
        for i, img, name in zip(range(0, batch_size), bgr_images, names):

            predict_dict = alpr_single_image(img, name, alpr_model)
            xyxy, cls_id, conf = yolo_out[i]
            clip_df = clip_dfs[i]

            if clip_df.empty:
                df_row_dict = {}
            else:
                df_row_dict = clip_df.to_dict()

            master_list.append(
                {
                    "file_name": name,
                    "yolo_model_confidence": conf,
                    "yolo_class_id": cls_id,
                    "yolo_xy_coords": xyxy,
                    **predict_dict[0],
                    **df_row_dict,
                }
            )

    pd.DataFrame(master_list).to_csv(output_path, index=False)
    print(f"Saved {len(master_list)} rows to {output_path}")


if __name__ == "__main__":
    # folder paths
    image_folder = r".\data\license_plate_detection\train\images"
    output_path = r".\data\cvp_model_results.csv"

    # file list
    file_list = [f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]

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
    run_batch_cvp(file_list, labels, output_path, image_folder)
