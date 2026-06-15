from ultralytics import YOLO


def run_model(path, model=None, save=False, conf=0.25):

    if not model:
        model = YOLO(r"..\..\models\Yolo26Model\weights\best.pt")

    results = model.predict(source=path, save=save, conf=conf)

    # 3. Process the results programmatically
    for result in results:
        boxes = result.boxes  # Bounding boxes object
        for box in boxes:
            # Get coordinates, class IDs, and confidence scores
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
    return xyxy, cls_id, confidence


if __name__ == "__main__":
    run_model(
        r"data\license_plate_detection\test\images\lp_test_001.jpg"
    )  # place image path into run model
