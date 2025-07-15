import os
import timm
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns
import rainbow_tqdm
from rainbow_tqdm import tqdm
import kagglehub

data_dir = kagglehub.dataset_download("siddhantmaji/unified-waste-classification-dataset")

print("Path to dataset files:", data_dir)

data_dir = f"{data_dir}/content/unified_dataset"
BATCH_SIZE = 32
IMG_SIZE = 224

# Preprocessing transforms only, no data augmentation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load full dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Stratified train/val split
targets = [sample[1] for sample in dataset.samples]
train_idx, val_idx = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    stratify=targets,
    random_state=42
)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MobileNetV3-Large from timm, pretrained on ImageNet
model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=8)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f} | Accuracy = {acc:.2f}%")

model.eval()
val_loss, correct, total = 0.0, 0, 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        val_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

val_acc = 100 * correct / total
print(f"Validation Accuracy: {val_acc:.2f}% | Loss: {val_loss/len(val_loader):.4f}")

# save model to a path
torch.save(model.state_dict(), "mobilenetv3_garbage_classifier.pth")