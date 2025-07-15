from model import get_transform, classes
import cv2
import torch
from torchvision.models import resnet18
import torch.nn as nn

import ev3dev2.motor
from ev3dev2.motor import LargeMotor, OUTPUT_A, OUTPUT_B, OUTPUT_C, SpeedPercent
from ev3dev2.sound import Sound

# Initialize motors (adjust OUTPUT_X to your wiring)
motor_recycle = LargeMotor(OUTPUT_A)
motor_compost = LargeMotor(OUTPUT_B)
motor_trash = LargeMotor(OUTPUT_C)
sound = Sound()

# Load model
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("waste_classifier_resnet18.pth", map_location=torch.device('cpu')))
model.eval()

# Get transform
transform = get_transform()

# Start webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = transform(frame_rgb).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = classes[predicted.item()]

    cv2.putText(frame, f'{label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Waste Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()