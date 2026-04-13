import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet50
from PIL import Image
import os

# -------------------------
# DEVICE
# -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# -------------------------
# LOAD CLASSES AUTOMATICALLY
# -------------------------

dataset = datasets.ImageFolder("dataset/train")
classes = dataset.classes

print("Classes:", classes)

# -------------------------
# LOAD MODEL
# -------------------------

model = resnet50()

model.fc = nn.Linear(model.fc.in_features, len(classes))

model.load_state_dict(
    torch.load("sports_model.pth", map_location=device)
)

model = model.to(device)

model.eval()

# -------------------------
# IMAGE TRANSFORM
# -------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# -------------------------
# PREDICT ALL JPG FILES
# -------------------------

for file in os.listdir():

    if file.endswith(".jpg") or file.endswith(".png"):

        img = Image.open(file).convert("RGB")

        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():

            outputs = model(img)

            probs = torch.softmax(outputs, dim=1)

            confidence, predicted = torch.max(probs, 1)

        player = classes[predicted.item()]

        conf = confidence.item() * 100

        print(f"{file} → {player} ({conf:.2f}%)")