# GUI search function, returns all columns for matching plate text
def search4id(metadata, search_text):
    search_output_list = []
    cols = metadata.columns.to_list()
    for x in range(len(metadata)):
        if search_text == metadata["text"].iloc[x]:
            print("Match found!")
            for y in range(len(cols)):
                search_element = print(f"{cols[y]}: {metadata[cols[y]].iloc[x]}")
                search_output_list.append(search_element)

            img_path = metadata["fn"].iloc[x]
            plate_text = metadata["text"].iloc[x]
            ocr_confidence = metadata["confidence"].iloc[x]

    return search_output_list, img_path, plate_text, ocr_confidence
