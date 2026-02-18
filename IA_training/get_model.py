import torch
from PIL import Image
from transformers import AutoFeatureExtractor, ResNetForImageClassification

# Charger le modèle et le processeur sauvegardés

# Objectif : Charger un modèle de régression entraîné pour prédire les 
# coordonnées (X, Y) à partir d'une image.

model_path = "./mon_modele_regression"
loaded_processor = AutoFeatureExtractor.from_pretrained(model_path)
loaded_model = ResNetForImageClassification.from_pretrained(model_path)

loaded_model.eval() # Mode évaluation

def predict_coords(image_path):
    img = Image.open(image_path).convert("RGB")
    
    # Prétraitement
    inputs = loaded_processor(img, return_tensors="pt")
    
    with torch.no_grad():
        outputs = loaded_model(**inputs).logits
    
    # On récupère les valeurs normalisées (0-1)
    norm_x, norm_y = outputs[0][0].item(), outputs[0][1].item()
    
    real_x = norm_x * 1920
    real_y = norm_y * 1080
    
    return real_x, real_y