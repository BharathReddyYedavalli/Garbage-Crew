# -------------------------
# 📌 1️⃣ IMPORTS & PATHS
# -------------------------
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import kagglehub

# ✅ USE GPU IF AVAILABLE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ✅ DOWNLOAD DATASET USING KAGGLEHUB
dataset_root = kagglehub.dataset_download("amankamath/garbage-v999")
print("Dataset downloaded to:", dataset_root)

# ✅ SET BASE DIR TO ACTUAL SUBFOLDER
base_dir = os.path.join(dataset_root, "Garbage", "garbage_classification")
print("Base dataset path:", base_dir)


# -------------------------
# 📌 2️⃣ TRANSFORMS & DATALOADERS
# -------------------------

# Good practice: Resize to 224 for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load all data
full_dataset = datasets.ImageFolder(
    root=base_dir,
    transform=transform
)

print("Classes:", full_dataset.classes)

# 80% train, 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Train size: {train_size}, Validation size: {val_size}")


# -------------------------
# 📌 3️⃣ LOAD PRE-TRAINED RESNET
# -------------------------

from torchvision import models

# Load ResNet18 with pretrained weights
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model = model.to(device)


# Replace final layer to match your 4 classes:
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 categories: Compost, Recycle, Other, Trashes

# Move to GPU or CPU
model = model.to(device)

print(model)


# -------------------------
# 📌 4️⃣ LOSS, OPTIMIZER, TRAINING LOOP
# -------------------------

# ✅ CrossEntropyLoss for multi-class classification
criterion = nn.CrossEntropyLoss()

# ✅ Adam optimizer, works well for most image tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Number of epochs (adjust as needed)
epochs = 5  # Start with 5 to check everything works

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


# -------------------------
# 📌 5️⃣ VALIDATION & METRICS
# -------------------------

# Switch to eval mode
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

# ✅ Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Validation Accuracy: {acc:.2%}")

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=full_dataset.classes,
            yticklabels=full_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# ✅ Precision, Recall, F1
report = classification_report(y_true, y_pred, target_names=full_dataset.classes)
print("Classification Report:\n")
print(report)



# -------------------------
# 📌 6️⃣ SAVE & LOAD MODEL
# -------------------------

# ✅ SAVE MODEL WEIGHTS
save_path = "waste_classifier_resnet18.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to: {save_path}")

# ✅ LOAD MODEL LATER
# (Example — run this in a new script or notebook when needed)
model_loaded = models.resnet18(pretrained=True)
model_loaded.fc = nn.Linear(model_loaded.fc.in_features, 4)  # Must match your trained head!
model_loaded.load_state_dict(torch.load(save_path))
model_loaded = model_loaded.to(device)
model_loaded.eval()

print("Model loaded and ready for inference.")
