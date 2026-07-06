import numpy as np
import pandas as pd

from src.models.predict import run_model
from utils.ocr_utils import alpr_single_image
from utils.pipeline_utils import do_zero_shot, crop_image


def _convert_numpy(obj):
    """Recursively convert numpy/pandas types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, list):
        return [_convert_numpy(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    return obj


def single_image_pipeline(
    image_path: str,
    model,
    zs_results_path: str,
    zs_label_list: list[str],
    do_zs: bool = False,
    do_ocr: bool = False,
    save: bool = False,
    conf: float = 0.5,
    alpr_model=None,
):
    # Run image through object detection model
    bounding_boxes, class_ids, confidences = run_model(
        path=image_path,
        model=model,
        save=save,
        conf=conf,
    )

    # Extract bounded objects from full image
    cropped_images = crop_image(
        image_path=image_path,
        bounding_boxes=bounding_boxes,
    )

    # Zero-Shot Classification
    results = None
    if do_zs:
        results = do_zero_shot(
            image_folder=image_path,
            results_path=zs_results_path,
            label_list=zs_label_list,
        )

    # OCR Text Extraction
    ocr_predictions = None
    if do_ocr:
        ocr_predictions = []
        for cropped_image in cropped_images:
            ocr_predictions.append(
                alpr_single_image(
                    image=cropped_image,
                    file_name=image_path,
                    alpr=alpr_model,
                )
            )

    result = _convert_numpy(
        {
            "bounding_boxes": bounding_boxes,
            "class_ids": class_ids,
            "confidences": confidences,
            "zs_results": results,
            "ocr_predictions": ocr_predictions,
        }
    )

    return result


if __name__ == "__main__":
    image_path = r"data\license_plate_detection\test\images\lp_test_002.jpg"
    zs_results_path = r"src\demo\zero_shot.json"
    zs_label_list = [
        "white",
        "red",
        "American",
    ]

    single_image_pipeline(
        image_path=image_path,
        model=None,  # will default to models/Yolo26Model/weights/best.pt
        zs_results_path=zs_results_path,
        zs_label_list=zs_label_list,
        # kwargs
        do_zs=True,
        do_ocr=True,
        save=False,
        conf=0.5,
        alpr_model=None,  # will default to fast_alpr.ALPR()
    )
