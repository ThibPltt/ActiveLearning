import torch
from PIL import Image
from transformers import AutoFeatureExtractor, ResNetForImageClassification

# Charger le modèle et le processeur sauvegardés
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
    
    # Optionnel : Remettre à l'échelle réelle (si vous aviez divisé par 1280 et 720)
    real_x = norm_x * 1280
    real_y = norm_y * 720
    
    return real_x, real_y

# Test
# x, y = predict_coords("chemin/vers/votre/image_test.jpg")
# print(f"Coordonnées prédites : X={x:.2f}, Y={y:.2f}")