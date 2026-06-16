import os
import cv2
import pandas as pd
from fast_alpr import ALPR
from ultralytics import YOLO
from PIL import Image
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from ocr_utils import alpr_single_image
from zero_shot_utils import download_clip

def run_model_batch(paths, model, conf=.25):
    '''Run batches of file through YOLO model
    
    Args:
        -paths: list of file paths to images

        -model: intitialized YOLO model

        -conf: confidence threshold

    Returns: 
        -nested list of xy bounding box coordinates, class ID and confidence estimate 
   
    '''
    results = model.predict(source=paths, save=False, conf=conf, verbose=False)

    out = []
    for result in results:
        if len(result.boxes) == 0:
            out.append((None, None, None))
            continue
        # Match original behaviour: keep the last box per image
        xyxy, cls_id, confidence = None, None, None
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist() or []
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
        out.append((xyxy, cls_id, confidence))
    return out

def clip_batch(pil_images, label_list, clip_model, batch_size = 16):
    '''Runs batch of images through CLIP model.
    
    Args:
        -pil images: directory of YOLO annotations

        -label_list: list of labels to be analyzed in clip model

        -clip_model: initialize clip model

        -batch_size: batch size of images

    Returns: 
        -dataframe with prediction values for each image in folder   
   
    '''
    raw = clip_model(pil_images, candidate_labels=label_list, batch_size=batch_size)

    dfs = []
    for preds in raw:
        df = pd.DataFrame(preds).T
        df.columns = df.iloc[-1]
        dfs.append(df[:-1])
    return dfs

def _alpr_worker(args):
    '''helper function to run image through CLIP model.
    
    Args:
        -args: input arguments of alpr model instance, image and file name 

    Returns: 
        -file name and prediction results from OCR   
   
    '''
    alpr_instance, image, name = args
    
    return name, alpr_single_image(alpr_instance, image, name)

def run_batch_cvp(file_list,labels,output_path, batch_size =16, num_workers =4):
        '''Runs files through CVP. Performs bounding box 
        prediction, zero shot classification and ocr text extraction
    
    Args:
        -file_list: list of files to be analyzed

        -labels: list of labels to be analyzed in clip model

        -output_path: file path to save outputs

        -batch_size: batch size of images to process

        -num_workers: number of workers for cpu

    Returns: 
        -csv file with license plate bounding boxes, extracted text and zero shot probabilities   
   
    '''
        # intialize models
        yolo_model = YOLO(r"..\models\Yolo26Model\weights\best.pt")
        alpr_model = ALPR()
        clip_model = download_clip()

        master_list = []

        # define batches based on batch size
        batches = [file_list[i : i + batch_size] for i in range(0, len(file_list), batch_size)]

        for batch in tqdm.tqdm(batches, desc="running batches..."):

            # define path and file names
            paths = [os.path.join(image_folder, f) for f in batch]
            names = [os.path.splitext(f)[0]       for f in batch]

            # list of cv2 image objects
            bgr_images = [cv2.imread(path) for path in paths]
            
            # list of pil image objects
            pil_images = []
            for img in bgr_images:
                if img is not None:
                    pil_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                else:
                    pil_images.append(None)

            # get predictions from batch of images
            yolo_out = run_model_batch(paths, model=yolo_model)

            # get pairs of images and index where images exist
            valid_pairs = [(i, img) for i, img in enumerate(pil_images) if img is not None]
            clip_lookup: dict = {}
            
            
            if valid_pairs:
                valid_idx, valid_pil = zip(*valid_pairs)
                clip_dfs  = clip_batch(list(valid_pil), labels, clip_model, batch_size)
                clip_lookup = dict(zip(valid_idx, clip_dfs))

            alpr_lookup: dict = {}
            alpr_args = [
                (alpr_model, bgr_images[i], names[i])
                for i in range(len(batch))
                if bgr_images[i] is not None
            ]
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {pool.submit(_alpr_worker, arg): arg[2] for arg in alpr_args}
                for future in as_completed(futures):
                    name_key, result = future.result()
                    alpr_lookup[name_key] = result

            for i, name in enumerate(names):
                xyxy, cls_id, conf = yolo_out[i]
                predict_dict = alpr_lookup.get(name, [{}])
                clip_df      = clip_lookup.get(i, pd.DataFrame())
                df_row_dict  = clip_df.to_dict() if not clip_df.empty else {}

                master_list.append({
                    "file_name":        name,
                    "model_confidence": conf,
                    "class_id":         cls_id,
                    "xy_coords":        xyxy,
                    **predict_dict[0],
                    **df_row_dict,
                })

        pd.DataFrame(master_list).to_csv(output_path, index=False)
        print(f"Saved {len(master_list)} to {output_path}")

if __name__ == "__main__":
    # folder paths
    image_folder   = r".\data\license_plate_detection\train\images"
    output_path     = r".\data\cvp_model_results.csv"

    # file list
    file_list = [f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]

    # labels
    labels = ["car", "truck", "bus", "vehicle", "motorcycle","no vehicle", "only license plate"]

    # function to run batches 
    run_batch_cvp(file_list, labels, output_path=output_path)