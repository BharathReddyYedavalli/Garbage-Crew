import cv2
import numpy as np
import timm
import torch
from torchvision import transforms
from ultralytics import YOLO  # works for YOLOv5 and YOLOv8

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
    torch.load("models\mobilenetv3_garbage_classifier.pth", map_location=DEVICE)
)
model.to(DEVICE).eval()

# Load YOLO object detector (e.g., yolov8n.pt or yolov5n.pt for small, fast models)
yolo_model = YOLO("yolov8n.pt")  # Or your trained detection model

# Preprocessing for classifier
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Webcam not accessible")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = yolo_model(frame)[0]  # First result
        boxes = results.boxes.xyxy.cpu().numpy()  # xyxy format

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            obj_crop = frame[y1:y2, x1:x2]

            if obj_crop.size == 0:
                continue

            input_tensor = preprocess(obj_crop).unsqueeze(0).to(DEVICE)
            outputs = model(input_tensor)
            pred = torch.argmax(outputs, 1).item()
            label = CLASSES[pred]

            # Draw detection + classification label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Garbage Detection & Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
