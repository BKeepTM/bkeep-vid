from ultralytics import YOLO
import supervision as sv
import sys
import numpy as np

if __name__ == "__main__":
    # naložen natreniran model
    model = YOLO("runs/detect/train/weights/best.pt")

    results = model.predict(
        source=sys.argv[1],
        conf=0.25,
        show=False,
        save=True,
        name="predict",          
        exist_ok=True  
    )

    result = results[0]
    detections = sv.Detections.from_ultralytics(result)

    class_names = model.names  # dict {id: name}

    for class_id, class_name in class_names.items():
        count = np.sum(detections.class_id == class_id)
        print(f"{class_name}: {count}")
