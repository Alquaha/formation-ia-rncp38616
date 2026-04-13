import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io

# -----------------------
# CONFIG
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    'Courtois', 'Dybala', 'Kane Williamson', 'Kobe Bryant', 'Kross',
    'Lionel Messi', 'Maria Sharapova', 'Mohamed Salah', 'Neymar', 'Pogba',
    'Roger Federer', 'Ronaldo', 'alcaraz', 'curry', 'djokovic', 'durant',
    'lebron', 'leclerc', 'mbappe', 'nadal', 'sabalenka', 'sinner', 'woods'
]

# -----------------------
# CHARGEMENT MODELE
# -----------------------
model = resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(torch.load("sports_model.pth", map_location=device))
model.eval()
model = model.to(device)

# -----------------------
# TRANSFORMATION
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------
# API
# -----------------------
app = FastAPI(title="Athlete Classifier API 🏆")

@app.get("/")
def root():
    return {
        "message": "API opérationnelle ✅",
        "classes": CLASSES,
        "nb_classes": len(CLASSES)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 0)

    return JSONResponse({
        "prediction": CLASSES[predicted.item()],
        "confidence": round(confidence.item() * 100, 2),
        "all_probabilities": {
            cls: round(probabilities[i].item() * 100, 2)
            for i, cls in enumerate(CLASSES)
        }
    })