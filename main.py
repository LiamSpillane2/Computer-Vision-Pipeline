import os
import csv
import shutil
import tempfile
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, File, Query, UploadFile
from ultralytics import YOLO

from src.single_image_pipeline import single_image_pipeline
from src.models.predict import run_model
from utils.zero_shot_utils import load_clip_pipeline

import warnings
from torch.jit import TracerWarning
warnings.filterwarnings("ignore", category=TracerWarning)


# App setup

app = FastAPI(title="Computer Vision Pipeline")

# Models are loaded once, not on every request
yolo_model = YOLO("models/Yolo26Model/weights/best.pt")
clip_pipeline = load_clip_pipeline()

def save_upload_to_temp(file: UploadFile) -> str:
    """Save an uploaded file to a temp path and return that path."""
    suffix = Path(file.filename).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    shutil.copyfileobj(file.file, tmp)
    tmp.close()
    return tmp.name


# def parse_pipeline_output(captured: str) -> dict:
#     """
#     single_image_pipeline() prints its results instead of returning them.
#     This function captures that printed output and parses it into a dict.
#     """
#     result = {}

#     for line in captured.splitlines():
#         line = line.strip()
#         if line.startswith("Bounding boxes:"):
#             result["bounding_boxes"] = line.removeprefix("Bounding boxes:").strip()
#         elif line.startswith("Class IDs:"):
#             result["class_ids"] = line.removeprefix("Class IDs:").strip()
#         elif line.startswith("Confidences:"):
#             result["confidences"] = line.removeprefix("Confidences:").strip()

#     return result


# Routes

@app.get("/health")
def health():
    return {"status": "solid"}


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(default=0.5, description="Confidence threshold (0-1)"),
):
    """
    Run YOLO object detection only on an uploaded image.
    Returns bounding boxes, class IDs, and confidence scores.
    """
    tmp_path = save_upload_to_temp(file)

    try:
        boxes, class_ids, confidences = run_model(tmp_path, model=yolo_model, conf=conf)
    finally:
        os.unlink(tmp_path)

    return {
        "bounding_boxes": boxes,
        "class_ids": class_ids,
        "confidences": confidences,
    }


@app.post("/pipeline")
async def pipeline(
    file: UploadFile = File(...),
    conf: float = Query(default=0.5, description="Confidence threshold (0-1)"),
    labels: str = Query(default="white, red, License Plate", description="Comma-separated labels for zero-shot"),
    zs_label: str = Query(default="License Plate", description="Primary label to filter zero-shot results by"),
    zs_prob_thres: float = Query(default=0.75, description="Zero-shot probability threshold"),
    do_zs: bool = Query(default=True),
    do_ocr: bool = Query(default=True)
):
    """
    Run the full single_image_pipeline on an uploaded image.
    """
    tmp_path = save_upload_to_temp(file)
    label_list = [l.strip() for l in labels.split(",")]
    zs_results_path = tmp_path + "_zs_results.json"

    try:
        result = single_image_pipeline(# Capture printed output from single_image_pipeline
        # buffer = io.StringIO()
        # with redirect_stdout(buffer):
        #     single_image_pipeline(
                image_path=tmp_path,
                model=yolo_model,
                zs_results_path=zs_results_path,
                zs_label_list=label_list,
                zs_label=zs_label,
                conf=conf,
                zs_prob_thres=zs_prob_thres,
                do_zs=do_zs,
                do_ocr=do_ocr
            )
    finally:
        os.unlink(tmp_path)
        if os.path.exists(zs_results_path):
            os.unlink(zs_results_path)

    csv_path = Path('results1.csv')

    # Prepare the data dictionary (combining image and result data)
    row_data = {'image': file.filename, **result}
    df = pd.DataFrame([row_data])
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode='a', index=False, header=write_header)
    # return result
# parse_pipeline_output(buffer.getvalue())

