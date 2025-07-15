import os
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns
from tqdm import tqdm
import kagglehub

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device) # for using gpu

dataset_root = kagglehub.dataset_download("siddhantmaji/unified-waste-classification-dataset")
print("Dataset downloaded to:", dataset_root)

# Need to fix this later: 
base_dir = os.path.join(dataset_root, "dataset")
print("Base dataset path:", base_dir)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
# in main.py # parameters should include classes
def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

classes = ["Battery", "Glass", "Metal", "Organic Waste", "Paper", "Plastic", "Textiles", "Trash"]
full_dataset = datasets.ImageFolder(
    root=base_dir,
    transform=transform
)

print("Classes:", full_dataset.classes)

# training size is 85%, validation size is 15%
train_size = int(0.85 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train size: {train_size}, Validation size: {val_size}")

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)



# Replace final layer to match your 8 classes:
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8)  # 8 categories: Compost, Recycle, Other, Trashes

# Move to GPU or CPU
model = model.to(device)

print(model)



# loss function
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if trained model already exists
save_path = "waste_classifier_resnet18.pth"

if os.path.exists(save_path):
    print(f"Found existing trained model at {save_path}")
    print("Loading existing model instead of training...")
    
    # Load the existing trained model
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
else:
    print(f"No existing model found at {save_path}")
    print("Starting training process...")
    
    # tunable iterations for training
    epochs = 10  # Start with 5 to check everything works

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # tqdm for live progress bar
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            # Zero gradients, forward, backward, step
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Running stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}, Accuracy = {100 * correct / total:.2f}%")

    # Save the newly trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model training completed and saved to: {save_path}")


# evaluation mode
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# check accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Validation Accuracy: {acc:.2%}")

# confusion matrix, based upon probabilities
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=full_dataset.classes,
            yticklabels=full_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Precision, Recall, F1
report = classification_report(y_true, y_pred, target_names=full_dataset.classes)
print("Classification Report:\n")
print(report)

print("Model ready for inference.")