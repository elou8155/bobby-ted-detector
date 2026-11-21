from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data="C:/Users/navjot.singh/Documents/Emily/stuffed_animals/config.yaml", epochs=50, imgsz=640)
results = model.train(data="C:Users/jimmm/Desktop/machine learning project/data/backup_stuffed_animals/config.yaml", epochs=50, imgsz=640)