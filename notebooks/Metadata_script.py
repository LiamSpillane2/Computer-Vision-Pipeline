import pandas as pd
from pathlib import Path
import re
import os
from PIL import Image
import time

def scan_dir(directory, records):
    file_paths  = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            scan_dir(entry.path, records)
        elif entry.is_file() and ".ods" not in entry.name and ".yaml" not in entry.name:
            path_splits = re.split(r'\\|\.',entry.path)

            name_splits = re.split(r'_|\.', entry.name)
            if "weld_data" in path_splits:
                weld_num = name_splits[0] + "-" + name_splits[1]
            else:
                weld_num = ""

            record = {
                "file path": entry.path,
                "file name": path_splits[-2], 
                "file number": name_splits[-2], 
                "weld number": weld_num, 
                "file type": path_splits[-1], 
                "image/label": path_splits[-3],
                "dataset": path_splits[-4],
                "size (kb)": (os.path.getsize(entry.path))/1000
                }
            
            file_paths.append(entry.path)
            records.append(record)
   
    df = pd.DataFrame(records)
    return df, file_paths

def get_info(file_path):
    stats = Path(file_path).stat()
    c_date = time.ctime(stats.st_birthtime)
    a_date = time.ctime(stats.st_atime)
    return c_date, a_date
            
def get_image_info(file_path, file_type):
    if file_type == "jpg":
        with Image.open(file_path) as img:
            rows_count = 0
            width, height = img.size
    elif file_type == "txt":
        with Path.open(file_path) as file:
            rows_count = sum(1 for line in file)
        width, height = 0,0
    else:
        width, height, rows_count = 0,0,0

    return width, height, rows_count

#Main
records = []
directory = Path(r"C:\Users\Installer\Documents\Data Science Class\Computer-Vision-Pipeline\data")

print("Getting Initial Information")
metadata_df, paths = scan_dir(directory, records)
print("Getting Creation and Accessed Dates")
metadata_df[["Creation Date", "Accessed Date"]] = metadata_df.apply(lambda row: get_info(row["file path"]), axis=1, result_type='expand')
print("Getting Additional File Information")
metadata_df[["Image Width","Image Height", "Number of Boxes"]] = metadata_df.apply(lambda row: get_image_info(row["file path"], row["file type"]), axis=1, result_type='expand')
#Send the data to a Json file
print("Outputting to JSON")
metadata_df.to_json('metadata.json',orient='records',indent=4)
print("Complete")