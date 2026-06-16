import cv2
import pandas as pd
from src.models.predict import run_model
from utils.ocr_utils import alpr_single_image
from utils.zero_shot_utils import load_clip_pipeline, run_classification, prob_results


def do_zero_shot(
    image_folder: str,
    results_path: str,
    label_list: list[str],
    label: str,
    prob_thres: float = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    classifier = load_clip_pipeline()
    results = run_classification(
        classifier=classifier,
        image_folder=image_folder,
        label_list=label_list,
    )
    p_results = prob_results(
        results,
        results_path,
        label=label,
        prob_thres=prob_thres,
    )

    return results, p_results


def crop_image(image_path: str, bounding_boxes: list[str]) -> list:
    image = cv2.imread(image_path)
    cropped_images = []
    for bounding_box in bounding_boxes:
        x1, y1, x2, y2 = map(int, bounding_box)
        cropped_image = image[y1:y2, x1:x2]
        cropped_images.append(cropped_image)

    return cropped_images


def single_image_pipeline(
    image_path: str,
    model,
    zs_results_path: str,
    zs_label_list: list,
    save: bool = False,
    conf: float = 0.5,
    zs_label: str = None,
    zs_prob_thres: float = 0.75,
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
    results, p_results = do_zero_shot(
        image_folder=image_path,
        results_path=zs_results_path,
        label_list=zs_label_list,
        label=zs_label,
        prob_thres=zs_prob_thres,
    )

    # OCR Text Extraction
    ocr_predictions = []
    for cropped_image in cropped_images:
        ocr_predictions.append(
            alpr_single_image(
                image=cropped_image,
                file_name=image_path,
                alpr=alpr_model,
            )
        )

    print(
        "\n\n====================================== RESULTS ======================================"
    )
    print(
        "\n---------------------------------- OBJECT DETECTION ---------------------------------\n"
    )
    print(f"Bounding boxes: {bounding_boxes}")
    print(f"Class IDs: {class_ids}")
    print(f"Confidences: {confidences}")
    print(
        "\n-------------------------------------- ZERO-SHOT ------------------------------------\n"
    )
    print(f"Zero-shot Results: {results}")
    print(f"Zero-shot P Results: {p_results}")
    print(
        "\n----------------------------------------- OCR ---------------------------------------\n"
    )
    print(f"OCR Predictions: {ocr_predictions}")


if __name__ == "__main__":
    image_path = r"data\license_plate_detection\test\images\lp_test_002.jpg"
    zs_results_path = r"src\demo\zero_shot.txt"
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
        zs_label="white",
        # kwargs
        save=False,
        conf=0.5,
        alpr_model=None,  # will default to fast_alpr.ALPR()
    )
