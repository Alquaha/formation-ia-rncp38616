import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights
)
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

# -----------------------
# PARAMETRES
# -----------------------
batch_size = 64
epochs = 5
lr = 0.0003
train_path = "dataset/train"
val_path = "dataset/val"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device :", device)

# -----------------------
# TRANSFORMATIONS
# -----------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------
# DATASETS
# -----------------------
train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_path,   transform=val_transform)
num_classes   = len(train_dataset.classes)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

# -----------------------
# FONCTION D'ENTRAINEMENT
# -----------------------
def train_model(model, model_name):
    print(f"\n{'='*40}")
    print(f"  Entraînement : {model_name}")
    print(f"{'='*40}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_accuracy": []}
    start_time = time.time()

    for epoch in range(epochs):
        # --- TRAIN ---
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # --- VALIDATION ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        history["train_loss"].append(total_loss)
        history["val_accuracy"].append(accuracy)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.2f} | Val Accuracy: {accuracy:.2f}%")

    duration = time.time() - start_time
    print(f"⏱️ Temps total : {duration:.1f}s")
    return history, duration

# -----------------------
# CREATION DES MODELES
# -----------------------
def get_resnet50():
    m = resnet50(weights=ResNet50_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def get_efficientnet():
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

def get_mobilenet():
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m

# -----------------------
# LANCEMENT
# -----------------------
models_to_compare = {
    "ResNet50":       get_resnet50(),
    "EfficientNet-B0": get_efficientnet(),
    "MobileNetV3":    get_mobilenet(),
}

results = {}
for name, model in models_to_compare.items():
    history, duration = train_model(model, name)
    results[name] = {"history": history, "duration": duration}

# -----------------------
# GRAPHIQUES
# -----------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, data in results.items():
    axes[0].plot(data["history"]["val_accuracy"], marker='o', label=name)
    axes[1].plot(data["history"]["train_loss"],   marker='o', label=name)

axes[0].set_title("Validation Accuracy par epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()
axes[0].grid(True)

axes[1].set_title("Train Loss par epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("comparaison_modeles.png")
plt.show()

# -----------------------
# TABLEAU RECAPITULATIF
# -----------------------
print("\n📊 RÉSULTATS FINAUX")
print(f"{'Modèle':<20} {'Accuracy finale':>16} {'Temps (s)':>10}")
print("-" * 50)
for name, data in results.items():
    acc = data["history"]["val_accuracy"][-1]
    dur = data["duration"]
    print(f"{name:<20} {acc:>15.2f}% {dur:>10.1f}s")
