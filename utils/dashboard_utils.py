import pandas as pd
from pathlib import Path
from ast import literal_eval

# fuction to combine zero-shot and ocr data into a dataframe.
def add2metadata(zsc_data, ocr_output):

    zsc_df = pd.DataFrame.from_dict(zsc_data)
    ocr_df = pd.DataFrame.from_dict(ocr_output)
    cols_to_add = ["text", "confidence", "bounding_box", "file_name"]
    # access ocr elements from the end of the dataframe
    model_df = zsc_df.join(ocr_df[cols_to_add])

    return model_df


# GUI search function, returns all columns for matching plate text
def search4id(metadata, search_text):

    filtered_df = metadata[
        metadata["ocr_text"].str.contains(
            search_text,
            case=False,
            na=False
        )
    ]

    if filtered_df.empty:
        return [], None, None, None

    filtered_df = filtered_df.sort_values(
        by="avg_confidence",
        ascending=False
    )

    return (
        filtered_df.values.tolist(),
        filtered_df["file_name"].tolist(),
        filtered_df["ocr_text"].tolist(),
        filtered_df["avg_confidence"].tolist()
    )

# Used in the metadata function
def safe_literal_eval(x):
    try:
        result = literal_eval(x)

        if isinstance(result, list):
            return result
        elif isinstance(result(int,float)):
            return [result]
        else:
            return []
    except:
        return []
        # print(f"Error calling literal_eval on {x} of type {type(x)}")

# Reads output of OCR data and returns dictionary for GUI
def get_metadata(route = 'csv'):
    # Read in OCR results file
    if route == 'csv':
        print("Reading CSV File")
        df = pd.read_csv(Path(f"{Path.cwd().parent}/data/cvp_model_results.csv"))
    else:
        print("Reading database")
        query = '''
        SELECT *
        FROM cvp_results
        '''
        conn = f'sqlite:///../data/cvp_database.db'
        df = pd.read_sql(query, conn)
    
    # Cleans up the dataframe columns for processing
    df["ocr_confidence"] = df["ocr_confidence"].apply(safe_literal_eval)
    df["ocr_text_list"] = df["ocr_text"].apply(lambda x: list(x) if isinstance(x, str) else [])

    # This uses the literal_eval function to convert the columns from strings to lists or dictionaries
    columns_to_convert = ["vehicle", "no vehicle", "car", "bus", "truck", "motorcycle", "only license plate"]
    for col in columns_to_convert:
        df[col] = df[col].apply(literal_eval)

    #This will loop through the rows 
    records = []
    for index, row in df.iterrows():
        #Cleaning up scores for each type of vehicle
        confidence = row["ocr_confidence"]
        text = list(row["ocr_text_list"])
        if confidence and text:
            trimmed_confidence = confidence[:len(text)] # TODO
            min_char, min_confidence = min(zip(text,trimmed_confidence), key=lambda x: x[1])
        else:
            min_char, min_confidence = None,0

        letters = pd.Series(row["ocr_text_list"])
        confs = pd.Series(trimmed_confidence)
        letter_confidence = pd.concat([letters, confs], axis=1)

        if row["ocr_region_confidence"] is None:
            row["ocr_region_confidence"] = 0.0

        # This creates a record to process some of the data, later to be transformed into a dataframe
        record = {
            "file_name" : row["index"],
            "ocr_region_confidence": row["ocr_region_confidence"] if not row["ocr_region_confidence"] == "NA" else 0,
            "avg_confidence" : sum(row["ocr_confidence"])/len(row["ocr_confidence"]) if row["ocr_confidence"] else 0,
            "min confidence" : min_confidence,
            "min confidence char" : min_char,
            "vehicle score" : row["vehicle"]["score"],
            "no vehicle score" : row["no vehicle"]["score"],
            "car score" : row["car"]["score"],
            "bus score" : row["bus"]["score"],
            "truck score" : row["truck"]["score"],
            "motorcycle score" : row["motorcycle"]["score"],
            "license plate only score" : row["only license plate"]["score"],
            "letter confidence": letter_confidence
        }
        records.append(record)
    
    # This creates the final dataframe to process
    columns_to_drop = ["vehicle", "no vehicle", "car", "bus", "truck", "motorcycle", "only license plate", "yolo_class_id", "ocr_bbox", "ocr_text_list", "ocr_region_confidence"]
    df2 = pd.DataFrame(records)
    df = df.drop(columns=columns_to_drop)
    df_final = pd.merge(df, df2, on='index')

    #This line calculates the average confidence for each region
    region_avg_confidence = df_final.groupby('ocr_region')['ocr_region_confidence'].mean()

    #This line counts the number of times each character appears in the dataset
    region_counts = df_final["ocr_region"].value_counts()

    #This block of code creates a list of all of the characters that appear and their confidences, then creates an avg for each char
    all_letters = pd.concat(df_final["letter confidence"].tolist(), ignore_index=True)
    all_letters.columns = ["char", "confidence"]
    all_letters = all_letters.dropna(subset=["char"])
    avg_char_conf = (
        all_letters
        .groupby("char")["confidence"]
        .mean()
        .sort_index()
    )

    #This block of code creates the dictionaries to return to the GUI
    overall_metadata = {
        "region stats": pd.concat([region_counts, region_avg_confidence], axis = 1),
        "char counts" : pd.concat([avg_char_conf, df_final["min confidence char"].value_counts()], axis=1)
    }
    meta_dict = {
        "Record Metadata": df_final,
        "Overall Metadata": overall_metadata,
        "Raw Data": df
    }

    print("Complete")
    return meta_dict

    #Jacks Code 
    # search_output_list = []
    # img_path = None
    # plate_text = None
    # ocr_confidence = None
    # cols = metadata.columns.to_list()
    # filtered_df = metadata[metadata["text"] == search_text]
    # search_output_list = filtered_df.values.tolist()
    # img_path = metadata["file_path"].tolist()
    # plate_text = metadata["text"].tolist()
    # ocr_confidence = metadata["avg_confidence"].tolist()
#Miguels orginial code 
    # for x in range(len(metadata)):
    #     if search_text == metadata["text"].iloc[x]:
    #         print("Match found!")
    #         for y in range(len(cols)):
    #             search_element = print(f"{cols[y]}: {metadata[cols[y]].iloc[x]}")
    #             search_output_list.append(search_element)

    #         img_path = metadata["file_path"].iloc[x]
    #         plate_text = metadata["text"].iloc[x]
    #         ocr_confidence = metadata["avg_confidence"].iloc[x]

   # return search_output_list, img_path, plate_text, ocr_confidence
