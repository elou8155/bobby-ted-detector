from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data=r"C:\Users\jimmm\Desktop\machine learning project\data\stuffed_animals\config.yaml", epochs=10, imgsz=640)