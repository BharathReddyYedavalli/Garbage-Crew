import cv2
import numpy as np
import timm
import torch
from torchvision import transforms

# Define constants
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

# Load model
model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=8)
model.load_state_dict(
    torch.load("mobilenetv3_garbage_classifier.pth", map_location=DEVICE)
)
model.to(DEVICE).eval()

# Define preprocessing
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Webcam not accessible")

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Center crop and preprocess
        h, w, _ = frame.shape
        min_dim = min(h, w)
        start_x, start_y = (w - min_dim) // 2, (h - min_dim) // 2
        cropped = frame[start_y : start_y + min_dim, start_x : start_x + min_dim]
        input_tensor = preprocess(cropped).unsqueeze(0).to(DEVICE)

        # Inference
        outputs = model(input_tensor)
        pred = torch.argmax(outputs, 1).item()
        label = CLASSES[pred]

        # Display
        cv2.putText(
            frame, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.imshow("Garbage Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
