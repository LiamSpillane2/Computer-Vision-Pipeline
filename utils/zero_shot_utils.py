"""---Feature Extraction---"""

import os
import math
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor, pipeline
from optimum.onnxruntime import ORTModelForZeroShotImageClassification


def load_images(
    directory, num_img, use_rand=True, seed=42, gs=True, normalize=True, img_obj=False
):
    """loads image file(s) from a directory, returns numpy array(s)

    Args:
        -directory: directory with image files

        -num_img: number of images to load (default = 1)

        -use_rand: (True/False)
            -True: random file will be selected
            -False: first file will be selected

        -seed: random seed number to use (if applicable)

        -gs: (True/False)
            -True: returned array is grayscale
            -False: returned array is RGB

        -normalize: (True/False)
            -True: returned array is normalized (0-1)
            -False: returned array is raw GS or RGB values (0-255)

        -img_obj: (default = False)
            - False: returns numpy array of image
            - True: returns Pillow image object

    Returns: array(s) or pillow object(s) representing image(s)

    Raises:
        -WindowsError: if directory is not valid

    """
    try:

        # valid extensions
        extensions = (".jpg", ".png")

        if directory[-4:] in extensions:
            directory = Path(directory)
            valid_files = [directory.name]
            directory = directory.parent
        else:
            # list of files with valid extensions
            valid_files = [f for f in os.listdir(directory) if f.endswith(extensions)]

        # initialize image array list
        img_arrays = []

        for i in range(num_img):

            # define file paths based on random parameter
            if use_rand:
                test_file = os.path.join(directory, random.sample(valid_files, k=1)[0])
            else:
                test_file = os.path.join(directory, valid_files[i])

            # open file
            img = Image.open(test_file)

            if img_obj == True:
                img_arr = img

            else:
                # define array based on gs parameter
                if gs:
                    img_arr = np.asarray(img.convert("L"))
                else:
                    img_arr = np.asarray(img.convert("RGB"))

                # normalize array based on normalize parameter
                if normalize:
                    img_arr = img_arr / 255

            # add array to array list
            img_arrays.append([img_arr, test_file])

        # return single array or array list based on num_img
        # if num_img == 1:
        #     return img_arrays[0]
        # else:
        #     return img_arrays
        return img_arrays
    except WindowsError as e:
        return "Directory Not Valid"
    except Exception as e:
        print(e)


def download_clip(force_download=False):
    """downloads model to device (defaul to openai CLIP)

    Args: None

    Returns: Pretrained moodel

    Raises:
    """
    if torch.cuda.is_available():
        # use larger model if GPU is available
        device = 0
        model = r"openai/clip-vit-large-patch14"
        print("using GPU")
    else:
        # otherwise use CPU model
        device = -1
        model = r"openai/clip-vit-base-patch32"
        print("using CPU")

    # define model save path as downloads folder
    local_directory = rf"C:/Users/{os.getlogin()}/Downloads/{model}"

    if os.path.exists(local_directory):
        print(f"Loading existing model from: {local_directory}")
        classifier = pipeline(
            task="zero-shot-image-classification",
            model=str(local_directory),
            device=device,
        )
        return classifier

    # get user input if model already saved locally
    else:
        classifier = pipeline(task="zero-shot-image-classification", model=model)
        classifier.save_pretrained(local_directory)
        print(f"Model successfully saved to: {local_directory}")

    return classifier


def clip_predictor(image, label_list, model):
    """assigns probablility of an image matching a label based on a classifier

    Args:
    - image: image to be analyzed (PIL object)
    - label_list: labels to be tested for probability
    - model: CLIP image classifier

    Returns:
    - dataframe with prediction values for each list item

    Raises:
    """

    # create predictions using classiier
    predictions = model(image, candidate_labels=label_list)

    # take list of dictionaries, convert to df and transpose
    df = pd.DataFrame(predictions).T

    # column names are last row values
    df.columns = df.iloc[-1]

    # return all values except last row
    return df[:-1]


# fuction to combine zero-shot and ocr data into a dataframe.
def add2metadata(zsc_data, ocr_output):

    zsc_df = pd.DataFrame.from_dict(zsc_data)
    ocr_df = pd.DataFrame.from_dict(ocr_output)
    cols_to_add = ["text", "confidence", "bounding_box", "file_name"]
    # access ocr elements from the end of the dataframe
    model_df = zsc_df.join(ocr_df[cols_to_add])

    return model_df


# loading the OpenAI clip model (def model_id unless otherwise stated)
def load_clip_pipeline(model_id="openai/clip-vit-base-patch32"):

    model = ORTModelForZeroShotImageClassification.from_pretrained(
        model_id, export=True
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # initialize the clip model
    classifier = pipeline(
        "zero-shot-image-classification",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.image_processor,
        provider="CPUExecutionProvider",
    )
    return classifier


# Run with load_clip_pipeline (run_classification is to seperate loading the model from the classification)
def run_classification(
    classifier,
    image_folder="../data/license_plate_detection/train/images",
    label_list=[],
    num_img=100,
    batch_size=8,
):
    # load images from image directory

    image_set = load_images(
        directory=image_folder, num_img=num_img, use_rand=True, img_obj=True
    )

    # get pillow image objects and file names from loaded images
    pil_images = [img[0] for img in image_set]
    file_names = [img[1] for img in image_set]

    prob_list = []
    results = classifier(pil_images, candidate_labels=label_list, batch_size=batch_size)

    for idx, (predictions, file_name) in enumerate(zip(results, file_names)):
        df = pd.DataFrame(predictions).T
        df.columns = df.iloc[-1]
        df = df[:-1]
        df["fn"] = file_name
        prob_list.append(df)

    return pd.concat(prob_list, ignore_index=True)


def prob_results(
    result: pd.DataFrame,
    json_path: str,
    label: str = "license plate",
    prob_thres: float = 0.5,
    plot: bool = False,
    max_cols: int = 3,
):
    result.to_json(json_path)
    df = pd.read_json(json_path)
    print(df)

    if label not in df.columns:
        raise KeyError(f"label: '{label}' not in label_list")

    df_license_plate_only = df[df[label] > prob_thres].reset_index()

    if plot:
        n = len(df_license_plate_only)
        cols = min(n, max_cols)
        rows = math.ceil(n / cols)
        squeeze = False
        _, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))

        if rows > 1:
            axes_flat = axes.flatten()

            for idx, row in df_license_plate_only.iterrows():
                axes_flat[idx].imshow(Image.open(row["fn"]))
                axes_flat[idx].set_title(f"Probability {row[label]:.2f}")
            plt.show()
        else:
            axes.imshow(Image.open(df_license_plate_only.iloc[0]["fn"]))
            axes.set_title(f"Probability {df_license_plate_only.iloc[0][label]:.2f}")
            plt.show()

    return df_license_plate_only
