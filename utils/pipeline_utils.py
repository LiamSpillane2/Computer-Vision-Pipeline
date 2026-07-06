import cv2
import pandas as pd
from .zero_shot_utils import load_clip_pipeline, run_classification, prob_results


def do_zero_shot(
    image_folder: str,
    results_path: str,
    label_list: list[str],
) -> pd.DataFrame:
    classifier = load_clip_pipeline()
    results = run_classification(
        classifier=classifier,
        image_folder=image_folder,
        label_list=label_list,
    )

    return results


def crop_image(image_path: str, bounding_boxes: list[str]) -> list:
    image = cv2.imread(image_path)
    cropped_images = []
    for bounding_box in bounding_boxes:
        x1, y1, x2, y2 = map(int, bounding_box)
        cropped_image = image[y1:y2, x1:x2]
        cropped_images.append(cropped_image)

    return cropped_images
