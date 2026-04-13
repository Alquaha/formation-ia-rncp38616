import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from collections import Counter

# -----------------------
# CHEMINS
# -----------------------
train_path = "dataset/train"
val_path = "dataset/val"

train_dataset = datasets.ImageFolder(train_path, transform=transforms.ToTensor())
val_dataset = datasets.ImageFolder(val_path, transform=transforms.ToTensor())

classes = train_dataset.classes
num_classes = len(classes)

print(f"Nombre de classes : {num_classes}")
print(f"Classes : {classes}")
print(f"Total images train : {len(train_dataset)}")
print(f"Total images val   : {len(val_dataset)}")

# -----------------------
# 1. DISTRIBUTION DES CLASSES
# -----------------------
# Utiliser les classes propres à chaque dataset
train_counts = Counter([train_dataset.classes[label] for _, label in train_dataset.samples])
val_counts   = Counter([val_dataset.classes[label] for _, label in val_dataset.samples])


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(train_counts.keys(), train_counts.values(), color="steelblue")
axes[0].set_title("Distribution - Train")
axes[0].set_xlabel("Classe")
axes[0].set_ylabel("Nombre d'images")
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(val_counts.keys(), val_counts.values(), color="coral")
axes[1].set_title("Distribution - Validation")
axes[1].set_xlabel("Classe")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("distribution_classes.png")
plt.show()
print("✅ Graphique sauvegardé : distribution_classes.png")

# -----------------------
# 2. EXEMPLES D'IMAGES PAR CLASSE
# -----------------------
fig, axes = plt.subplots(num_classes, 5, figsize=(15, num_classes * 3))

for class_idx, class_name in enumerate(classes):
    class_folder = os.path.join(train_path, class_name)
    images = os.listdir(class_folder)[:5]

    for img_idx, img_name in enumerate(images):
        img_path = os.path.join(class_folder, img_name)
        img = mpimg.imread(img_path)
        axes[class_idx][img_idx].imshow(img)
        axes[class_idx][img_idx].axis("off")
        if img_idx == 0:
            axes[class_idx][img_idx].set_title(class_name, fontsize=12, fontweight="bold")

plt.suptitle("Exemples d'images par classe", fontsize=16)
plt.tight_layout()
plt.savefig("exemples_par_classe.png")
plt.show()
print("✅ Graphique sauvegardé : exemples_par_classe.png")

# -----------------------
# 3. TAILLE MOYENNE DES IMAGES
# -----------------------
from PIL import Image

sizes = []
for img_path, _ in train_dataset.samples[:100]:  # échantillon de 100
    with Image.open(img_path) as img:
        sizes.append(img.size)

widths  = [s[0] for s in sizes]
heights = [s[1] for s in sizes]

print(f"\n📐 Taille moyenne des images (avant resize) :")
print(f"   Largeur  : {np.mean(widths):.0f}px (min {min(widths)}, max {max(widths)})")
print(f"   Hauteur  : {np.mean(heights):.0f}px (min {min(heights)}, max {max(heights)})")

# -----------------------
# 4. RATIO TRAIN / VAL
# -----------------------
total = len(train_dataset) + len(val_dataset)
print(f"\n📊 Ratio Train/Val :")
print(f"   Train : {len(train_dataset)} images ({100*len(train_dataset)/total:.1f}%)")
print(f"   Val   : {len(val_dataset)} images ({100*len(val_dataset)/total:.1f}%)")
