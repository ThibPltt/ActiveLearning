import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, ResNetForImageClassification
import random
import sys

# ==========================================
# 0. CONFIGURATION
# ==========================================

# Objectif : montrer les performances relatives de chaque stratégie sur une même image, 
# en utilisant les modèles "Ratio 50%" (meilleurs modèles actuellement disponibles).

STRATEGIES = [
    ("Random", "best_models_random", "gray", "o"),            # Rond Gris
    ("Dropout 0.25", "best_models_dropout_0.25", "red", "v"),   # Triangle bas Rouge
    ("Data Aug", "best_models_data_aug", "blue", "s"),        # Carré Bleu
    ("Clustering", "best_models_clustering", "orange", "D"),    # Losange Orange
    ("Clustering L2", "best_models_clustering_L2", "green", "^"), # Triangle haut Vert
    ("Learning Loss", "best_models_learning_loss", "purple", "X"), # Croix Violette
    ("Mixte", "best_models_mixed", "cyan", "P")             # Pentagone Cyan
]

# On vise les meilleurs loss
TARGET_SUBFOLDER = "best_model_ratio_50"

MAX_X = 1920.0
MAX_Y = 1080.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. RECHERCHE ROBUSTE
# ==========================================

sys.stdout.reconfigure(encoding='utf-8')
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent 

def find_path_recursive(root_path, target_name):
    """Cherche un dossier n'importe où dans le projet"""
    for root, dirs, files in os.walk(root_path):
        if target_name in dirs:
            return Path(root) / target_name
    return None

# Recherche du Dataset Split, avec les données de train, test et AL
dataset_dir = find_path_recursive(project_root, "FINAL_DATASET")
if not dataset_dir:
    print("Erreur: FINAL_DATASET introuvable.")
    sys.exit()

dir_test_B = dataset_dir / "Domain_B_Test"
all_files = list(dir_test_B.glob("*.jpg")) + list(dir_test_B.glob("*.JPG"))

if not all_files:
    print("Erreur: Pas d'images dans Test B.")
    sys.exit()

# Choix aléatoire d'une image
target_image_path = random.choice(all_files)
print(f"--- GRAPHIQUE DISPERSION FINAL ---")
print(f"Image : {target_image_path.name}")

# ==========================================
# 2. VÉRITÉ TERRAIN
# ==========================================

def get_coords(filename):
    try:
        name = filename.replace('.jpg', '').replace('.JPG', '')
        parts = name.split('_')
        x = float(parts[-2]); y = float(parts[-1])
        return x, y
    except: return MAX_X/2, MAX_Y/2

true_x, true_y = get_coords(target_image_path.name)
pil_image = Image.open(target_image_path).convert("RGB")

# ==========================================
# 3. CALCULS ET PLOT
# ==========================================

plt.figure(figsize=(12, 10))
ax = plt.gca()
ax.invert_yaxis()

# CIBLE (VRAI)
plt.scatter(true_x, true_y, c='lime', s=400, marker='+', linewidth=4, label='VRAI (Cible)', zorder=10)
plt.text(true_x + 15, true_y + 15, "CIBLE", color='green', fontweight='bold', fontsize=12)

# Variables pour le zoom auto
min_x, max_x = true_x, true_x
min_y, max_y = true_y, true_y

print("\n--- PRÉDICTIONS ---")

for strat_name, folder_name, color, marker in STRATEGIES:
    
    # 1. Trouver le dossier
    main_folder = find_path_recursive(project_root, folder_name)
    
    if main_folder:
        model_path = main_folder / TARGET_SUBFOLDER
        
        if model_path.exists():
            try:
                # 2. Charger modèle
                processor = AutoImageProcessor.from_pretrained(model_path)
                model = ResNetForImageClassification.from_pretrained(model_path).to(DEVICE)
                model.eval()
                
                # 3. Prédire
                inputs = processor(pil_image, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    logits = model(**inputs).logits
                    preds = torch.sigmoid(logits).cpu().squeeze().numpy()
                
                pred_x = preds[0] * MAX_X
                pred_y = preds[1] * MAX_Y
                dist = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
                
                print(f" > {strat_name.ljust(15)} : Erreur {dist:.1f} px")
                
                # 4. Plot
                plt.scatter(pred_x, pred_y, c=color, s=200, marker=marker, label=f"{strat_name}", zorder=5)
                
                plt.plot([true_x, pred_x], [true_y, pred_y], color=color, linestyle='--', alpha=0.5)
                
                # Zoom boundaries
                min_x = min(min_x, pred_x); max_x = max(max_x, pred_x)
                min_y = min(min_y, pred_y); max_y = max(max_y, pred_y)
                
            except Exception as e:
                print(f" > {strat_name.ljust(15)} : Erreur ({e})")
        else:
            print(f" > {strat_name.ljust(15)} : Modèle 'Ratio 50' pas encore entraîné.")
    else:
        print(f" > {strat_name.ljust(15)} : Dossier principal introuvable.")

# ==========================================
# 4. FINALISATION
# ==========================================

plt.title(f"Précision des Stratégies (Modèles Ratio 50%)\nImage: {target_image_path.name}", fontsize=14, fontweight='bold')
plt.xlabel("Position X (pixels)")
plt.ylabel("Position Y (pixels)")
plt.legend(loc='best', shadow=True, fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

# Zoom
margin = 150
plt.xlim(min_x - margin, max_x + margin)
plt.ylim(max_y + margin, min_y - margin) # Y inversé

filename = "graphique_dispersion.png"
plt.savefig(filename, dpi=300)
print(f"\n[OK] Graphique sauvegardé : {filename}")
plt.show()