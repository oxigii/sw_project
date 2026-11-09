from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640
)