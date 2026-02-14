import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, ResNetForImageClassification
import random
import sys
from tqdm import tqdm

# ==========================================
# 0. CONFIGURATION
# ==========================================
# Nombre d'images à tester pour la moyenne
N_SAMPLES = 5000  

# Stratégies à comparer
STRATEGIES = [
    ("Random", "best_models_random"),
    ("Dropout 0.25", "best_models_dropout_0.25"),
    ("Data Aug", "best_models_data_aug"),
    ("Clustering", "best_models_clustering"),
    ("Clustering L2", "best_models_clustering_L2"),
    ("Learning Loss", "best_models_learning_loss"),
    ("Mixte", "best_models_mixed")
]

TARGET_SUBFOLDER = "best_model_ratio_50"
MAX_X = 1920.0
MAX_Y = 1080.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. PRÉPARATION DES DONNÉES
# ==========================================
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent

# Recherche Dataset
def find_path_recursive(root_path, target_name):
    for root, dirs, files in os.walk(root_path):
        if target_name in dirs:
            return Path(root) / target_name
    return None

dataset_dir = find_path_recursive(project_root, "FINAL_DATASET")
if not dataset_dir:
    print("Erreur: FINAL_DATASET introuvable.")
    sys.exit()

dir_test_B = dataset_dir / "Domain_B_Test"
all_files = list(dir_test_B.glob("*.jpg")) + list(dir_test_B.glob("*.JPG"))

if not all_files:
    print("Erreur: Pas d'images dans Test B.")
    sys.exit()

# On prend N échantillons ou tout le dossier si < N
real_n = min(N_SAMPLES, len(all_files))
test_files = random.sample(all_files, real_n)

print(f"--- BENCHMARK MASSIF ({real_n} IMAGES) ---")
print(f"Cible : Modèles Ratio 50%")

# ==========================================
# 2. FONCTIONS UTILES
# ==========================================
def get_coords(filename):
    try:
        name = filename.replace('.jpg', '').replace('.JPG', '')
        parts = name.split('_')
        x = float(parts[-2])
        y = float(parts[-1])
        return x, y
    except:
        return MAX_X/2, MAX_Y/2

# Dictionnaire pour stocker les résultats : {"Random": [err1, err2...], ...}
results_data = {name: [] for name, _ in STRATEGIES}
mean_scores = {}

# ==========================================
# 3. BOUCLE PAR STRATÉGIE (Pour économiser le GPU)
# ==========================================
for strat_name, folder_name in STRATEGIES:
    print(f"\n>>> Test de la stratégie : {strat_name}")
    
    # Recherche du modèle
    main_folder = find_path_recursive(project_root, folder_name)
    if not main_folder:
        print(f"   [SKIP] Dossier {folder_name} introuvable.")
        continue
        
    model_path = main_folder / TARGET_SUBFOLDER
    if not model_path.exists():
        print(f"   [SKIP] Modèle 50% introuvable dans {folder_name}.")
        continue

    # Chargement Modèle
    try:
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = ResNetForImageClassification.from_pretrained(model_path).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"   [ERREUR] Chargement échoué ({e})")
        continue

    # Inférence sur les 5000 images
    errors = []
    
    # On utilise tqdm pour la barre de progression
    for img_path in tqdm(test_files, desc=f"   Inférence {strat_name}", ascii=True):
        try:
            # Vérité
            true_x, true_y = get_coords(img_path.name)
            
            # Prédiction
            image = Image.open(img_path).convert("RGB")
            inputs = processor(image, return_tensors="pt").to(DEVICE)
            
            with torch.no_grad():
                logits = model(**inputs).logits
                preds = torch.sigmoid(logits).cpu().squeeze().numpy()
            
            pred_x = preds[0] * MAX_X
            pred_y = preds[1] * MAX_Y
            
            # Erreur Euclidienne
            dist = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
            errors.append(dist)
            
        except Exception as e:
            continue # Si une image est corrompue, on passe

    # Stockage et Nettoyage
    results_data[strat_name] = errors
    mean_val = np.mean(errors) if errors else 0
    std_val = np.std(errors) if errors else 0
    mean_scores[strat_name] = mean_val
    
    print(f"   => Erreur Moyenne : {mean_val:.2f} px (±{std_val:.2f})")
    
    # Libération VRAM GPU
    del model
    del processor
    torch.cuda.empty_cache()

# ==========================================
# 4. RÉSULTATS & GRAPHIQUE (BOX PLOT)
# ==========================================
print("\n" + "="*40)
print("CLASSEMENT FINAL (Basé sur 1000 images)")
print("="*40)

# Tri par performance (le plus petit score est le meilleur)
sorted_strats = sorted(mean_scores.items(), key=lambda item: item[1])

for rank, (name, score) in enumerate(sorted_strats, 1):
    print(f"{rank}. {name.ljust(15)} : {score:.2f} px d'erreur moyenne")

# --- GRAPHIQUE ---
plt.figure(figsize=(10, 6))

# On filtre les stratégies vides (celles qui n'ont pas été trouvées)
plot_data = []
plot_labels = []
colors = []

# Code couleur cohérent avec les graphes précédents
color_map = {
    "Random": "gray",
    "Dropout 0.25": "red",
    "Data Aug": "blue",
    "Clustering": "orange",
    "Clustering L2": "green",
    "Learning Loss": "purple",
    "Mixte": "cyan"
}

for name in [k for k, v in sorted_strats]: # On affiche dans l'ordre du classement
    data = results_data[name]
    if data:
        plot_data.append(data)
        plot_labels.append(f"{name}\n(µ={np.mean(data):.0f})")
        colors.append(color_map.get(name, "black"))

# Création du Box Plot
bplot = plt.boxplot(plot_data, labels=plot_labels, patch_artist=True, showfliers=False) 
# showfliers=False cache les points aberrants extrêmes pour garder le graphe lisible

# Coloriage
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

plt.title(f"Distribution des Erreurs sur {real_n} Images Test (Ratio 50%)", fontsize=14, fontweight='bold')
plt.ylabel("Erreur en Pixels (Plus bas est meilleur)")
plt.grid(True, axis='y', linestyle='--', alpha=0.5)

save_name = "benchmark_massif_boxplot3.png"
plt.savefig(save_name, dpi=300)
print(f"\n[OK] Graphique statistique généré : {save_name}")
plt.show()