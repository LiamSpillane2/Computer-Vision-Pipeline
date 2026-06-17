import os
import json
from pathlib import Path
from PIL import Image
from collections import defaultdict


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

ROOT = Path(__file__).resolve().parents[1]
LICENSE_DATASET_ROOT = f'{ROOT}\data\license_plate_detection'
WELD_DATASET_ROOT = f'{ROOT}\data\weld_data'

SPLIT = ['train', 'val', 'test']

# =========================
# SAFETY UTILITIES
# =========================

def safe_load_image_size(img_path):
    try:
        with Image.open(img_path) as img:
            return img.size  # (w, h)
    except Exception:
        return None


def clamp(val, min_v, max_v):
    return max(min_v, min(val, max_v))


# =========================
# CLASS DISCOVERY
# =========================

def discover_classes(root, splits):
    class_ids = set()

    for split in splits:
        label_dir = root / split / "labels"
        if not label_dir.exists():
            continue

        for label_file in label_dir.glob("*.txt"):
            try:
                with open(label_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_ids.add(int(parts[0]))
            except Exception:
                print(f"[WARN] Skipping corrupted label: {label_file}")

    return sorted(class_ids)


# =========================
# YOLO TO COCO CONVERTER
# =========================

def convert_split(root, split, class_ids, output_path):
    images_dir = root / split / "images"
    labels_dir = root / split / "labels"

    if not images_dir.exists():
        print(f"[SKIP] Missing images dir: {images_dir}")
        return

    coco = {
        "info": {
            "description": "YOLO → COCO converted dataset",
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cid, "name": f"class_{cid}", "supercategory": "none"}
            for cid in class_ids
        ]
    }

    image_id = 1
    ann_id = 1

    image_files = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS
    ]

    print(f"\n[INFO] Processing {split}: {len(image_files)} images")

    for img_path in sorted(image_files):
        label_path = labels_dir / f"{img_path.stem}.txt"

        # ---- HARD VALIDATION: image must exist ----
        if not img_path.exists():
            print(f"[SKIP] Missing image: {img_path}")
            continue

        # ---- HARD VALIDATION: label must exist ----
        if not label_path.exists():
            print(f"[SKIP] Missing label: {label_path.name}")
            continue

        size = safe_load_image_size(img_path)
        if size is None:
            print(f"[SKIP] Corrupt image: {img_path}")
            continue

        img_w, img_h = size

        # ---- Add image entry ----
        coco["images"].append({
            "id": image_id,
            "file_name": str(img_path.name),  # relative name only
            "width": img_w,
            "height": img_h
        })

        # ---- Parse labels ----
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:5])
            except ValueError:
                continue

            # ---- YOLO → COCO conversion ----
            abs_w = w * img_w
            abs_h = h * img_h

            abs_x = (x - w / 2) * img_w
            abs_y = (y - h / 2) * img_h

            # ---- CLAMP (prevents invalid boxes) ----
            abs_x = clamp(abs_x, 0, img_w - 1)
            abs_y = clamp(abs_y, 0, img_h - 1)
            abs_w = clamp(abs_w, 1, img_w - abs_x)
            abs_h = clamp(abs_h, 1, img_h - abs_y)

            area = abs_w * abs_h

            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cls,
                "bbox": [
                    round(abs_x, 2),
                    round(abs_y, 2),
                    round(abs_w, 2),
                    round(abs_h, 2)
                ],
                "area": round(area, 2),
                "iscrowd": 0
            })

            ann_id += 1

        image_id += 1

    # ---- SAVE ----
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    # ---- REPORT ----
    print(f"\n[DONE] {split}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Saved to: {output_path}")

# CONVERTING PIPELINE

def convert_dataset(root_dir):
    root = Path(root_dir)

    splits = ["train", "val", "test"]

    print("\n[STEP 1] Discovering classes...")
    class_ids = discover_classes(root, splits)
    print(f"Classes found: {class_ids}")

    for split in splits:
        print(f"\n[STEP 2] Converting split: {split}")

        output_file = root / f"instances_{split}.json"
        convert_split(root, split, class_ids, output_file)

    print("\n[COMPLETE] COCO dataset generated safely.")


    for i in SPLIT:
        IMG_DIR = os.path.join(root_dir, i, "images")
        LBL_DIR = os.path.join(root_dir, i, "labels")
        COCO_JSON = os.path.join(root_dir, f'instances_{i}.json')


        # =========================
        # LOAD FILES
        # =========================
        images = {p.stem for p in Path(IMG_DIR).glob("*.*")}
        labels = {p.stem for p in Path(LBL_DIR).glob("*.txt")}

        with open(COCO_JSON, "r") as f:
            coco = json.load(f)

        coco_images = {Path(img["file_name"]).stem for img in coco["images"]}


        # =========================
        # BASIC COUNTS
        # =========================
        print("\n===== COUNTS =====")
        print(f"Images on disk:   {len(images)}")
        print(f"Labels on disk:   {len(labels)}")
        print(f"COCO images:      {len(coco_images)}")


        # =========================
        # EXACT MATCH CHECKS
        # =========================
        missing_labels = images - labels
        missing_images = labels - images
        missing_coco = images - coco_images
        extra_coco = coco_images - images


        # =========================
        # REPORT
        # =========================
        print("\n===== MISMATCH REPORT =====")

        print(f"Images without labels: {len(missing_labels)}")
        print(f"Labels without images: {len(missing_images)}")
        print(f"Images missing in COCO: {len(missing_coco)}")
        print(f"COCO images not on disk: {len(extra_coco)}")


        # =========================
        # SHOW SAMPLES
        # =========================
        print("\n===== SAMPLE ISSUES =====")

        print("Missing labels example:", list(missing_labels)[:10])
        print("Missing images example:", list(missing_images)[:10])
        print("Missing in COCO example:", list(missing_coco)[:10])
        print("Extra in COCO example:", list(extra_coco)[:10])


        # =========================
        # FINAL STATUS
        # =========================
        ok = (
            len(missing_labels) == 0 and
            len(missing_images) == 0 and
            len(missing_coco) == 0 and
            len(extra_coco) == 0
        )

        print("\n===== FINAL RESULT =====")
        print(f"{i.upper()} DATASET OK" if ok else f"{i.upper()} DATASET BROKEN")


convert_dataset(WELD_DATASET_ROOT) #Converting Weld Dataset YOLO Labels -> COCO Labels
convert_dataset(LICENSE_DATASET_ROOT) #Converting License Plate Dataset YOLO Labels -> COCO Labels

