import shutil
import os


def rename_and_copy_all(base_input_dir, base_output_dir, dataset):

    ##Loops over test/train/valid subfolders and renames all images in each.

    folders = ["test", "train", "valid"]
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for folder_name in folders:
        source_folder = os.path.join(base_input_dir, folder_name, "images")

        if not os.path.exists(source_folder):
            print(f"Skipping '{folder_name}' — folder not found.")
            continue

        image_files = sorted(
            [
                f
                for f in os.listdir(source_folder)
                if os.path.splitext(f)[1].lower() in image_extensions
            ]
        )

        output_folder = os.path.join(base_output_dir, folder_name, "images")
        os.makedirs(output_folder, exist_ok=True)

        for i, filename in enumerate(image_files, start=1):
            ext = os.path.splitext(filename)[1].lower()
            new_name = f"{dataset}_{folder_name}_{str(i).zfill(3)}{ext}"

            src = os.path.join(source_folder, filename)
            dst = os.path.join(output_folder, new_name)

            shutil.copy2(src, dst)
            print(f"[{folder_name}] {filename}  →  {new_name}")


def rename_copy_and_add_headers_all(base_input_dir, base_output_dir, headers, dataset):

    ##Loops over test/train/valid subfolders, reads from their 'labels' subfolder,
    ##copies and renames .txt files, and adds column headers to each file.

    folders = ["test", "train", "valid"]

    for folder_name in folders:
        source_folder = os.path.join(base_input_dir, folder_name, "labels")

        if not os.path.exists(source_folder):
            print(f"Skipping '{folder_name}/labels' — folder not found.")
            continue

        txt_files = sorted(
            [
                f
                for f in os.listdir(source_folder)
                if os.path.splitext(f)[1].lower() == ".txt"
            ]
        )

        output_folder = os.path.join(base_output_dir, folder_name, "labels")
        os.makedirs(output_folder, exist_ok=True)

        for i, filename in enumerate(txt_files, start=1):
            new_name = f"{dataset}_{folder_name}_{str(i).zfill(3)}.txt"

            src = os.path.join(source_folder, filename)
            dst = os.path.join(output_folder, new_name)

            # Read original content
            with open(src, "r") as f:
                original_content = f.read()

            # Write headers + original content to destination
            with open(dst, "w") as f:
                f.write(headers + "\n" + original_content)

            print(f"[{folder_name}] {filename}  →  {new_name}")


def corners_to_yolo(row_values):

    # Converts 4-corner bounding box coordinates to YOLO format.

    # Expects row_values to be a list of strings:
    # [class, x1, y1, x2, y2, x3, y3, x4, y4]

    # YOLO format requires normalized values (0-1), so img_width
    # and img_height of the corresponding image are needed.

    cls = row_values[0]

    # Extract all x and y corner coordinates
    x_coords = [float(row_values[i]) for i in [1, 3, 5, 7]]
    y_coords = [float(row_values[i]) for i in [2, 4, 6, 8]]

    # Bounding box edges
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    # YOLO center + dimensions, normalized to image size
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_labels_all(base_input_dir, base_output_dir, dataset):
    ##Loops over test/train/valid subfolders, reads from their 'labels' subfolder,
    ##converts corner-format bounding boxes to YOLO format, adds headers,
    ##and saves renamed files to the output directory.

    folders = ["test", "train", "valid"]
    header = "class x_center y_center width height"

    for folder_name in folders:
        source_folder = os.path.join(base_input_dir, folder_name, "labels")

        if not os.path.exists(source_folder):
            print(f"Skipping '{folder_name}/labels' — folder not found.")
            continue

        txt_files = sorted(
            [
                f
                for f in os.listdir(source_folder)
                if os.path.splitext(f)[1].lower() == ".txt"
            ]
        )

        output_folder = os.path.join(base_output_dir, folder_name, "labels")
        os.makedirs(output_folder, exist_ok=True)

        for i, filename in enumerate(txt_files, start=1):
            new_name = f"{dataset}_{folder_name}_{str(i).zfill(3)}.txt"

            src = os.path.join(source_folder, filename)
            dst = os.path.join(output_folder, new_name)

            converted_rows = [header]

            with open(src, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue  # skip blank lines

                    row_values = line.split()  # splits on any whitespace

                    if len(row_values) != 9:
                        print(
                            f"  WARNING: unexpected column count in {filename}, skipping row: {line}"
                        )
                        continue

                    yolo_row = corners_to_yolo(row_values)
                    converted_rows.append(yolo_row)

            with open(dst, "w") as f:
                f.write("\n".join(converted_rows))

            print(f"[{folder_name}] {filename}  →  {new_name}")
