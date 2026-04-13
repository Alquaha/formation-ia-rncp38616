### Dataset
Le dataset est disponible sur Kaggle :
🔗 [Athlete Image Classification Dataset](https://www.kaggle.com/datasets/alexsponsoos/athlete-image-classification-datase)

### Installation & Setup
```bash
cd bloc5-athlete-classifier
pip install -r requirements.txt
```

Télécharger le dataset via Kaggle API :
```bash
pip install kaggle
kaggle datasets download -d alexsponsoos/athlete-image-classification-datase
unzip athlete-image-classification-datase.zip -d dataset/
```

Puis entraîner le modèle :
```bash
python train.py        # génère sports_model.pth
python main.py         # lance l'API
```
