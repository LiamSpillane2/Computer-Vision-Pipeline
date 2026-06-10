from tqdm import tqdm
import os
import cv2
from fast_alpr import ALPR
import pandas as pd
from itertools import chain
import ast
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def crop_img_yolo(img_path, annotation_line, pad=0):
    '''Crops image based on yolo annotation format
    
    Args:
        -img_path: path to image file (.jpg or .png)

        -annotation_line: annotation string in YOLO format

        -pad: additional pading (percentage) around the cropped region to include (default 0)

    Returns: 
        - cv2 image object of cropped image    
   
    '''
    
    try:
        # Split the string first to inspect raw text parts
        parts = annotation_line.strip().split()
        if len(parts) < 5:
            return None

        # Safe check: Try parsing the values; skip line if it contains header words
        try:
            parts = [float(x) for x in parts]
        except ValueError:
            return None  # Quietly ignores header rows like "class x_center..."

        img = cv2.imread(img_path)
        if img is None:
            return None
        H, W, _ = img.shape

        _, xc, yc, w, h = parts[:5]

        x1 = int(((xc - w/2) * W) - (W * pad))
        y1 = int(((yc - h/2) * H) - (H * pad))
        x2 = int(((xc + w/2) * W) + (W * pad))
        y2 = int(((yc + h/2) * H) + (H * pad))

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        cropped_img = img[y1:y2, x1:x2]
        if cropped_img.size == 0:
            return None
            
        return cropped_img
    except Exception as e:
        print(f"Error cropping {img_path}: {e}")
        return None

def alpr_single_image(alpr, image, file_name):
    '''OCR predictions for single image through ALPR model
    
    Args:
        -alpr:ALPR model (typically use "alpr = ALPR()")

        -image: cv2 image object

        -file_name: name of file being processed

    Returns: 
        - dictionary with prediction values    
   
    '''
    
    results = alpr.predict(image)
    output_list = []
    if len(results) == 0:
        result_dict = {
            "bbox" : "NA",
            "bbox_norm": "NA",
            "confidence" : "NA",
            "label" : "NA",
            "text" : "NA",
            "confidence" : "NA",
            "region" : "NA" ,
            "region_confidence" : "NA",
            "file_name": file_name
            }
        return [result_dict]
    else:
        img_h, img_w, _ = image.shape
        for result in results:
            
            bbox = result.detection.bounding_box
            x1_bounded = max(0, int(bbox.x1))
            y1_bounded = max(0, int(bbox.y1))
            x2_bounded = min(img_w, int(bbox.x2))
            y2_bounded = min(img_h, int(bbox.y2))

            bbox_list = [x1_bounded,x2_bounded, y1_bounded, y2_bounded] 

            bbox_norm = [x1_bounded/img_w, x2_bounded/img_w, y1_bounded/img_h, y2_bounded/img_h]
            
            result_dict = {
            "bbox" : bbox_list,
            "bbox_norm": bbox_norm,
            "confidence" : result.detection.confidence,
            "label" : result.detection.label,
            "text" : result.ocr.text,
            "confidence" : result.ocr.confidence,
            "region" : result.ocr.region ,
            "region_confidence" : result.ocr.region_confidence,
            "file_name": file_name
            }
            output_list.append(result_dict)
        return output_list
    
def do_alpr(anno_dir, image_dir):
    '''Processes many files through ALPR prediction model.
    Assumes annotations and images share same base file name.
    
    Args:
        -anno_dir: directory of YOLO annotations

        -image_dir: directory of images

    Returns: 
        -dataframe with prediction values for each image in folder   
   
    '''

    #initialize APLR model
    alpr = ALPR()
    print("\n")
    
    image_list = os.listdir(image_dir)
    file_list = [file.replace(".jpg","") for file in image_list]

    output_list = []

    for file in tqdm(file_list, desc = "Processing Files..."):
        img_file, anno_file = file + ".jpg", file + ".txt"
        ann_fp = os.path.join(anno_dir, anno_file)
        img_fp = os.path.join(image_dir, img_file)
        
        # open annotation and read lines
        try:
            with open(ann_fp, "r") as f:
                lines = f.readlines()
        except:
            continue

        # run each line of the file through the OCR models
        for idx, line in enumerate(lines):
            if idx > 0:
                # 
                if not line.strip():
                    continue

                # process image through cropping function    
                cropped_img = crop_img_yolo(img_fp, line)
                output_list.append(alpr_single_image(alpr, cropped_img, file))
        
    # flatten output list (for files with mutiple annotations)
    flat_output_list = list(chain.from_iterable(output_list))
    
    # return dataframe of output list
    return pd.DataFrame(flat_output_list)

def annotate_images_ocr(img_dir, anno_dir, save_dir, csv_fp = None, df = None):
    '''annotates images with OCR bounding boxes and text
    
    Args:
        -img_dir: image directory where you want to 

        -anno_dir: cv2 image object

        -save_dir: name of file being processed
         
        -csv_fp: file path to .csv file with OCR prediction data

        -df: dataframe with OCR prediction data

    Returns: 
        - dictionary with prediction values    
   
    '''

    # determine whether to use file path or dataframe based on parameters
    if csv_fp is None and df is None:
        return
    elif csv_fp is not None and df is None:
        eval_df = pd.read_csv(csv_fp)
    elif csv_fp is not None and df is not None:
        eval_df = pd.read_csv(csv_fp)
    else:
        eval_df = df
    

    # drop na columns which are blank
    eval_df = eval_df.dropna(subset=['text'])

    # loop through rows in dataframe
    for idx, row in tqdm(eval_df.iterrows(), "processing images..."):
        
        # define file paths
        file  = row["file_path"]
        file_path = os.path.join(img_dir, file+".jpg")
        anno_fp = os.path.join(anno_dir, file+".txt")

        # open annotation and read lines
        with open(anno_fp, "r") as f:
            lines = f.readlines()
        
        # run each line of the file through the OCR models
        for idx, line in enumerate(lines):
            if idx > 0:
                
                # crop image based on annotation
                cropped_img = crop_img_yolo(file_path, line)

                # define figure and show copped image
                fig, axes = plt.subplots()
                axes.imshow(cropped_img)

                # determine bounding box coords and width/height
                coords = ast.literal_eval(row["bbox"])
                ymax, xmax = float(coords[3]),float(coords[1])
                ymin, xmin = float(coords[2]),float(coords[0])

                width = xmax-xmin
                height = ymax-ymin

                # create rectangle obejct
                lw = 2
                rect = patches.Rectangle(
                    (xmin+lw, ymin+lw), width-lw, height-lw, 
                    linewidth=2, edgecolor='red', facecolor='none'
                )

                # add rectangle to plot
                axes.add_patch(rect)

                # add text to plot
                text_y_pos = ymin - 10 if ymin - 10 > 0 else ymin + 20
                axes.text(
                    xmin, text_y_pos, row["text"], 
                    color='white', fontsize=12, fontweight='bold',
                    bbox=dict(facecolor='red', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.3'))
                
                # define save path and save figure
                save_path = file_path = os.path.join(save_dir, f"{file}_w_bbox.png")
                plt.savefig(save_path)
 
if __name__ == "__main__":

    alpr = ALPR()
    
    img_path = r".\data\license_plate_detection\train\images\lp_train_972.jpg"
    image = cv2.imread(img_path)
    alpr_dict = alpr_single_image(alpr,image,"lp_train_972")
    
    print(f"{alpr_dict} \n")
    
    train_img_dir = r"./data/license_plate_detection/train/images"
    train_anno_dir = r"./data/license_plate_detection/train/labels"

    
    ocr_df = do_alpr(train_anno_dir,train_img_dir)

    print(ocr_df.head(5))