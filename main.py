import cv2
import numpy as np
import timm
import torch
from torchvision import transforms

# Ask user preferences
use_yolo = input("Enable YOLO object detection? (Y/n): ").lower() != "n"
snapshot_mode = (
    input("Enable snapshot mode instead of live feed? (Y/n): ").lower() != "n"
)


# Optional YOLO import and model load
if use_yolo:
    from ultralytics import YOLO

    yolo_model = YOLO("models/yolov8n.pt")  # Replace with your own model if needed

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

# Load classifier
model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=8)
model.load_state_dict(
    torch.load("models/mobilenetv3_garbage_classifier.pth", map_location=DEVICE)
)
model.to(DEVICE).eval()

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
if not cap.isOpened():
    raise IOError("Webcam not accessible")

print("\nPress 'q' to quit.")
if snapshot_mode:
    print("Press 's' to take snapshot, 'r' to reset live view.\n")

with torch.no_grad():
    frozen = False
    freeze_frame = None

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
                results = yolo_model(display_frame)[0]
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

        # Always show the current frame
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
