import os
import random
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor,  AutoModel

def load_images(directory, num_img = 1, use_rand = True, seed = 42, gs = True, normalize = True):
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
            
            # define array based on gs parameter
            if gs:
                img_arr = np.asarray(img.convert("L"))
            else:
                img_arr = np.asarray(img.convert("RGB"))

            # normalize array based on normalize parameter
            if normalize:
                img_arr = img_arr/255
            
            # add array to array list
            img_arrays.append(img_arr)

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

if __name__ == "__main__":
    test_dir =  r"./data/formatted/license_plate_detection/train/images"
    print("success!")

    test_arr = load_images(test_dir, num_img= 3, gs = False, normalize= True)

    embeddings = get_image_embeddings(test_arr)

    print(embeddings)

