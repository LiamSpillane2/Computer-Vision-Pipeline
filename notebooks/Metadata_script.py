import pandas as pd
from pathlib import Path
import re
import os
from PIL import Image
import time

#Functions used to get metadata
def get_info(file_path):
    #This function takes in the file path to each file, and returns the creation and accessed date
    stats = Path(file_path).stat()
    c_date = time.ctime(stats.st_birthtime)
    a_date = time.ctime(stats.st_atime)
    return c_date, a_date
            
def get_image_info(file_path, file_type):
    #This function takes in the file path to each file and the file type determined from the get_paths function
    if file_type == "jpg":
        #If the file path is to a jpg file, then this function will open the image and get the width and height
        img = Image.open(file_path)
        width, height = img.size
        rows_count = 0
        img.close()
    elif file_type == "txt":
        #If the file type is txt, then this function counts the number of boxes in the annotation
        file = Path.open(file_path)
        rows_count = len(file.readlines())
        file.close()
        width, height = 0,0
    else:
        #If it is neither txt or jpg, then it returns everything equal to zero
        width, height, rows_count = 0,0,0

    return width, height, rows_count

def get_paths(directory):
    # Finds allfiles in the directory and subdirectories
    file_paths = []
    file_type = []
    data_type = []
    file_size = []
    file_num = []
    file_imageset = []
    weld_num = []
    file_dataset = []

    for root, dirs, files in os.walk(directory):
        #Gets the root directory dataset
        for filename in files:
            #This gets the full path for the file
            filepath = os.path.join(root, filename)
            file_paths.append(filepath) 
            #This gets the file name and type
            name_splits = re.split(r'\\', root)
            name_splits = [item for item in name_splits if item] #Clean whitespace out
            padded_name_splits = [item.zfill(3) if item.isdigit() else item for item in name_splits]

            #This is for getting the file number and file type for each file
            filename_splits = re.split(r'_|\.|-', filename)

            if len(padded_name_splits) == 10:
                file_imageset.append(padded_name_splits[len(padded_name_splits) - 3])
                data_type.append(padded_name_splits[len(padded_name_splits) - 1])
                file_num.append(filename_splits[len(filename_splits) - 2])
                file_type.append(filename_splits[len(filename_splits) - 1])
                file_dataset.append(padded_name_splits[len(padded_name_splits) - 2])
                file_size.append(round(os.path.getsize(filepath)/1000, ndigits=2))
                #If the file is in the weld directory, it will collect additional information to seperate each row of data more
                if padded_name_splits[len(padded_name_splits) - 3] == "weld_data":
                    weld_num.append(filename_splits[0] + "-" + filename_splits[1])
                else:
                    weld_num.append("")
            else:
                file_imageset.append("Other")
                data_type.append("Other")
                file_num.append(000)
                file_type.append(filename_splits[len(filename_splits) - 1])
                file_size.append(round(os.path.getsize(filepath)/1000, ndigits=2))
    
    #Covert to series so they can be stored in the dataframe and be sorted by the main part of the code
    file_paths = pd.Series(file_paths)
    file_num = pd.Series(file_num)
    file_type = pd.Series(file_type)
    file_size = pd.Series(file_size)
    data_type = pd.Series(data_type)
    file_imageset = pd.Series(file_imageset)
    weld_num = pd.Series(weld_num)
    file_dataset = pd.Series(file_dataset)
    return file_paths, file_num, weld_num, file_type, file_size, file_imageset, file_dataset, data_type

#Main
#Set the directory for where the images and labels are to get metadata with
directory = Path(r"C:\Users\Installer\Documents\Data Science Class\Computer-Vision-Pipeline\data")

#Gets the initial round of metadata, information about the files
metadata_df = pd.DataFrame()
metadata_df["File Paths"], metadata_df["File Number"], metadata_df["Weld Number"], metadata_df["File Type"],  metadata_df["Size (Kb)"], metadata_df["Image Set"], metadata_df["Dataset Type"], metadata_df["Type"] = get_paths(directory)
metadata_df[["Creation Date", "Access Date"]] = metadata_df.apply(lambda row: get_info(row["File Paths"]), axis=1, result_type = 'expand')
print("Initial Data Collected")

#This sorts the data, placing the images and their corresponding labels next to each other
metadata_df["File Number"] = metadata_df["File Number"].astype(int)
metadata_df = metadata_df.sort_values(by=['Image Set', 'File Number', 'Weld Number'])
print("Data Sorted")

#This gets the width and height of all of the images, and gets the number of boxes in an annotation file
print("Opening files for information")
metadata_df[["Image Width","Image Height", "Number of Boxes"]] = metadata_df.apply(lambda row: get_image_info(row["File Paths"], row["File Type"]), axis=1, result_type='expand')

#Send the data to a Json file
print("Outputting to JSON")
metadata_df.to_json('metadata.json',orient='records',indent=4)
print("Complete")