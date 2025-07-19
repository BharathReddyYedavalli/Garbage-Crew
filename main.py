import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

print(torch.__version__)

parser = argparse.ArgumentParser(description="Garbage Classifier")
parser.add_argument(
    "-q", "--quantized", action="store_true", help="Use quantized model"
)
parser.add_argument("-y", "--yolo", action="store_true", help="Enable YOLO detection")
parser.add_argument(
    "-s", "--snapshot", action="store_true", help="Enable snapshot mode"
)
parser.add_argument(
    "-p", "--pretrained", action="store_true", help="Use pretrained model"
)
args = parser.parse_args()

use_quantized = args.quantized
use_yolo = args.yolo
snapshot_mode = args.snapshot
use_pretrained = args.pretrained

# Constants
IMG_SIZE = 224
CLASSES = [
    "battery",
    "glass",
    "metal",
    "organic_waste",
    "paper_cardboard",
    "plastic",
    "textiles",
    "trash",
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if use_quantized:
    # torch.backends.quantized.engine = "qnnpack"
    torch.set_num_threads(2)

    if use_pretrained:
        model = models.quantization.mobilenet_v3_large(quantize=True, weights="DEFAULT")
        model.classifier[3] = nn.quantized.Linear(1280, len(CLASSES))
        model = torch.jit.script(model)
    else:
        model = torch.jit.load("./models/quantized_mobilejit.pt", map_location="cpu")
else:
    if use_pretrained:
        model = models.mobilenet_v3_large(weights="DEFAULT")
        model.classifier[3] = nn.Linear(1280, len(CLASSES))
    else:
        model = torch.load(
            "./models/mobilenetv3_garbage_classifier_full.pt",
            map_location=DEVICE,
            weights_only=False,
        )
    model = torch.jit.script(model)
    model.to(DEVICE)
model.eval()

# Optional YOLO import and model load
if use_yolo:
    from ultralytics import YOLO

    # yolo_model = YOLO("models/yolov8n.pt")  # Replace with your own model if needed
    yolo_model = YOLO(
        "./models/yolo11n.pt"
    )  # latest YOLOv11 model, pretty much same latency as yolov8 but more accurate

# Preprocessing
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_SIZE)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_SIZE)
cap.set(cv2.CAP_PROP_FPS, 36)

if not cap.isOpened():
    raise IOError("Webcam not accessible")

print("\nPress 'q' to quit.")
if snapshot_mode:
    print("Press 's' to take snapshot, 'r' to reset live view.\n")

with torch.no_grad():
    frozen = False
    freeze_frame = None

    frame_count = 0
    last_fps_time = time.time()
    fps = 0.0

    boxes = np.empty((0, 4))
    results = None

    while True:
        # Always read a new frame if not frozen
        if not frozen:
            ret, frame = cap.read()
            if not ret:
                break
            display_frame = frame.copy()
        else:
            display_frame = freeze_frame.copy()

        # Only run detection/classification if frozen (snapshot taken), or if not in snapshot mode
        if not snapshot_mode or frozen:
            if use_yolo:
                if frame_count % 3 == 0:
                    results = yolo_model(display_frame, imgsz=480)[0]
                    boxes = results.boxes.xyxy.cpu().numpy()

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    obj_crop = display_frame[y1:y2, x1:x2]

                    if obj_crop.size == 0:
                        continue

                    input_tensor = preprocess(obj_crop).unsqueeze(0).to(DEVICE)
                    outputs = model(input_tensor)
                    pred = torch.argmax(outputs, 1).item()
                    label = CLASSES[pred]

                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        display_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )
            else:
                h, w, _ = display_frame.shape
                min_dim = min(h, w)
                start_x, start_y = (w - min_dim) // 2, (h - min_dim) // 2
                cropped = display_frame[
                    start_y : start_y + min_dim, start_x : start_x + min_dim
                ]
                input_tensor = preprocess(cropped).unsqueeze(0).to(DEVICE)
                outputs = model(input_tensor)
                pred = torch.argmax(outputs, 1).item()
                label = CLASSES[pred]

                cv2.putText(
                    display_frame,
                    label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

        # FPS calculation
        frame_count += 1
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            fps = frame_count / (current_time - last_fps_time)
            last_fps_time = current_time
            frame_count = 0

        # Draw FPS on frame
        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (10, display_frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Garbage Classifier", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if snapshot_mode:
            if key == ord("s") and not frozen:
                freeze_frame = frame.copy()
                frozen = True
            elif key == ord("r") and frozen:
                frozen = False
                freeze_frame = None

        if key == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()
