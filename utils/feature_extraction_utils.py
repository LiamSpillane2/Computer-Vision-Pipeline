import os
import random
from PIL import Image
import numpy as np
import pandas as pd
import torch
from transformers import AutoImageProcessor,  AutoModel, pipeline
import warnings

def load_images(directory, num_img = 1, use_rand = True, seed = 42, gs = True, normalize = True, img_obj = False):
    '''loads image file(s) from a directory, returns numpy array(s)
    
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

    Returns: numpy array (or list of arrays) representing image(s)
    
    Raises: 
        -WindowsError: if directory is not valid
    
    '''
    try:

        # valid extensions
        extensions = ('.jpg', '.png')

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
                    img_arr = img_arr/255
            
            # add array to array list
            img_arrays.append([img_arr, test_file])

        # return single array or array list based on num_img
        if num_img == 1:
            return img_arrays[0]
        else:
            return img_arrays


    except WindowsError as e:
        return "Directory Not Valid"
    except Exception as e:
        print(e)

def get_image_embeddings(img_arr):

    os.environ["HF_HUB_VERBOSITY"] = "error"
    warnings.filterwarnings("ignore", category=UserWarning)
    # 2. Suppress the Windows symlink warning
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # VERIFIED REPO STR: Native Google-backed checkpoint on HF
    processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
    model = AutoModel.from_pretrained("google/mobilenet_v2_1.0_224").to(device)

    inputs = processor(img_arr, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # MobileNetV2 outputs pooler_output with extra dimensions: [batch, channels, 1, 1]
    # We use squeeze() to turn it into a flat 2D vector: [batch, channels]
    embedding = outputs.pooler_output.squeeze(-1).squeeze(-1)

    return embedding

def download_model(model = "openai/clip-vit-large-patch14"):
    '''downloads model to device (defaul to openai CLIP)
    
    Args: model
       
    Returns: classifier
    
    Raises: 
    '''
    if torch.cuda.is_available():
        model = r"openai/clip-vit-large-patch14"
    else: 
        model = r"openai/clip-vit-base-patch32"
 
    # define model save path as downloads folder
    local_directory = rf"C:/Users/{os.getlogin()}/Downloads/{model}"

    # check if model is already saved
    if not os.path.exists(local_directory):
        
        # save model to directory
        classifier = pipeline(task="zero-shot-image-classification", model= model)
        classifier.save_pretrained(local_directory)
        print(f"Model successfully saved to {local_directory}")
    
    # get user input if model already saved locally
    else:
        response = input("Clip model already downloaded, would you like to redownload? (Y/N) ")

        # redownload based on user input
        if response.upper() == "Y":
            classifier.save_pretrained(local_directory)
            classifier = pipeline(task="zero-shot-image-classification", model= model)
            print(f"Model successfully saved to: {local_directory}")
        
        # otherwise return
        else:
            classifier = pipeline(task="zero-shot-image-classification",model=local_directory)
            print(f"Model already saved to: {local_directory}")
        
    return classifier

def image_label_pred(image, label_list, classifier):
    '''assigns probablility of an image matching a label based on a classifier
    
    Args: 
    - image: image to be analyzed (PIL object)
    - label_list: labels to be tested for probability
    - classifier: CLIP image classifier
       
    Returns:
    - dataframe with 
    
    Raises: 
    '''

    predictions = classifier(image, candidate_labels=label_list)
    
    # take list of dictionaries, convert to df and transpose
    df = pd.DataFrame(predictions).T
    
    # column names are last row values
    df.columns = df.iloc[-1]

    # return all values except last row
    return df[:-1]

if __name__ == "__main__":

    print('main')



