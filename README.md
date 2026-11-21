# Bobby-Ted Detector

A YOLO11 object-detection project for identifying the stuffed animals Bobby and Ted.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip ultralytics
```

## Run the trained model

```bash
python3 - <<'PY'
from ultralytics import YOLO

model = YOLO("data/Emily/runs/detect/train5/weights/best.pt")
results = model.predict(
    source="data/Emily/IMG_0942.jpg",
    save=True,
)

print("Results:", results)
PY
```

## Dataset

The dataset YAML is [`data/Emily/stuffed_animals/config.yaml`](data/Emily/stuffed_animals/config.yaml).
It contains two classes:

- `Bobby`
- `Ted`
