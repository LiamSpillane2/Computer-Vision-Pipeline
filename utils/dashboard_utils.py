import pandas as pd


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
        metadata["text"].str.contains(
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
        filtered_df["file_path"].tolist(),
        filtered_df["text"].tolist(),
        filtered_df["avg_confidence"].tolist()
    )



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
