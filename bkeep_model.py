from ultralytics import YOLO


if __name__ == "__main__":
    # Load a model
    #model = YOLO("yolo26n.yaml")  # build a new model from YAML
    model = YOLO("runs/detect/train/weights/last.pt")

    # Train the model
    #results = model.train(data="data.yaml", epochs=100, resume=True, imgsz=640, device=0)

    #resume training
    results = model.train(resume=True, device=0, workers=0)