# Bobby-Ted Detector

A YOLO11 object-detection project for identifying my stuffed animals, Bobby and Ted.

## Project process

I collected photographs of both stuffed animals and used them to build a custom image dataset.

1. **Captured the images:** I took roughly 100 pictures of Bobby and Ted in a variety of positions and surroundings.
2. **Labelled the objects:** I used [LabelImg](https://github.com/HumanSignal/labelImg) to draw a rectangular bounding box around each stuffed animal.
3. **Named the classes:** I assigned every box to either `Bobby` or `Ted`.
4. **Exported YOLO labels:** LabelImg saved each image's annotation as a `.txt` file in YOLO format:

   ```text
   class_id center_x center_y width height
   ```

   The coordinates are normalized so they work at different image sizes.
5. **Split the dataset:** I divided the labelled images into training and validation sets.
6. **Fine-tuned YOLO11:** I started with pretrained `yolo11n.pt` weights and fine-tuned the model on the two classes.

The repository's active dataset contains 60 training images and 8 validation images. The labels contain 51 Bobby instances and 49 Ted instances.

## Label format

An annotation such as:

```text
0 0.254960 0.435268 0.210979 0.200893
1 0.664187 0.447917 0.407077 0.316964
```

means that the image contains:

- One `Bobby` box (`class 0`)
- One `Ted` box (`class 1`)

The dataset configuration is [`data/Emily/stuffed_animals/config.yaml`](data/Emily/stuffed_animals/config.yaml).

```yaml
names:
  0: Bobby
  1: Ted
```

## Training

The model was trained with Ultralytics YOLO:

- Base model: YOLO11 nano (`yolo11n`)
- Image size: 640 x 640
- Epochs: 50
- Batch size: 16
- Task: object detection

The training configuration for the final recorded run is available at [`data/Emily/runs/detect/train5/args.yaml`](data/Emily/runs/detect/train5/args.yaml).

The trained weights are stored in [`data/Emily/runs/detect/train5/weights/`](data/Emily/runs/detect/train5/weights/):

- `best.pt` — checkpoint with the strongest recorded validation performance
- `last.pt` — checkpoint from the final epoch

The final run reached about 0.995 mAP50 and 0.918 mAP50-95 on the validation set. Because the validation set is small, these values are not as statistically reliable as results from a larger and more diverse dataset.

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
