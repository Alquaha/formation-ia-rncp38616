import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader

# -----------------------
# PARAMETRES
# -----------------------

batch_size = 64
epochs = 20
lr = 0.0003

train_path = "dataset/train"
val_path = "dataset/val"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device :", device)

# -----------------------
# TRANSFORMATIONS
# -----------------------

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -----------------------
# DATASET
# -----------------------

train_dataset = datasets.ImageFolder(
    train_path,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_path,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=True
)

num_classes = len(train_dataset.classes)

print("Classes :", train_dataset.classes)

# -----------------------
# MODELE
# -----------------------

model = resnet50(weights=ResNet50_Weights.DEFAULT)

model.fc = nn.Linear(model.fc.in_features, num_classes)

# Fine tuning complet
for param in model.parameters():
    param.requires_grad = True

model = model.to(device)

# -----------------------
# LOSS + OPTIMIZER
# -----------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=lr
)

best_accuracy = 0

# -----------------------
# TRAINING
# -----------------------

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss:.4f}")

    # -----------------------
    # VALIDATION
    # -----------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print("Validation Accuracy:", accuracy)

    # sauvegarde meilleur modèle
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(model.state_dict(), "sports_model.pth")

        print("✅ Best model saved")

print("Training finished")