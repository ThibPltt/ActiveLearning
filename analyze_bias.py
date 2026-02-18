import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import sys

# Import conditionnel
try:
    from transformers import AutoImageProcessor, ResNetForImageClassification
except ImportError:
    print("ERREUR: 'transformers' manquant. Faites: pip install transformers")
    sys.exit()

# ==========================================
# 0. CONFIGURATION
# ==========================================

# Objectif : Analyser les biais des modèles entraînés en IA_Training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> GPU ACTIVÉ : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

MAX_X = 1920.0
MAX_Y = 1080.0
N_SAMPLES = 1000 

# Noms des dossiers modèles (qui sont DANS IA_Training)
STRATEGIES = {
    "Random": "best_models_random",
    "Dropout": "best_models_dropout_0.25",
    "DataAug": "best_models_data_aug",
    "ClusterL2": "best_models_clustering_L2",
    "Mixte": "best_models_mixed"
}

TARGET_SUBFOLDER = "best_model_ratio_50"

# ==========================================
# 1. LOCALISATION INTELLIGENTE (DATASET & IA_TRAINING)
# ==========================================

def find_roots():
    current_path = Path(".").resolve()
    
    # 1. Trouver FINAL_DATASET
    dataset_path = None
    # On cherche ici, puis dans le parent, puis grand-parent
    search_paths = [current_path, current_path.parent, current_path.parent.parent]
    
    for p in search_paths:
        if (p / "FINAL_DATASET").exists():
            dataset_path = p / "FINAL_DATASET"
            break
            
    # 2. Trouver IA_Training (Les cerveaux)
    ia_training_path = None
    for p in search_paths:
        if (p / "IA_Training").exists():
            ia_training_path = p / "IA_Training"
            break
            
    # Cas où le script est DEJA dans IA_Training
    if not ia_training_path and (current_path / "best_models_random").exists():
        ia_training_path = current_path

    return dataset_path, ia_training_path

# Exécution de la recherche
dataset_dir, ia_dir = find_roots()

print("-" * 40)
if dataset_dir and dataset_dir.exists():
    print(f"[OK] DATASET trouvé : {dataset_dir}")
    test_dir = dataset_dir / "Domain_B_Test"
else:
    print("[ERREUR] FINAL_DATASET introuvable. Vérifiez l'emplacement.")
    sys.exit()

if ia_dir and ia_dir.exists():
    print(f"[OK] IA_TRAINING trouvé : {ia_dir}")
else:
    print("[ERREUR] Dossier 'IA_Training' introuvable.")
    print("Assurez-vous que le script est à côté de 'IA_Training' ou dedans.")
    sys.exit()
print("-" * 40)

# Chargement fichiers
all_files = sorted(list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.JPG")))
import random
random.seed(42)
test_files = random.sample(all_files, min(N_SAMPLES, len(all_files)))

def get_coords_and_id(filename):
    name = filename.replace('.jpg', '').replace('.JPG', '')
    parts = name.split('_')
    try:
        x = float(parts[-2])
        y = float(parts[-1])
        group_id = parts[0] 
        return x, y, group_id
    except:
        return MAX_X/2, MAX_Y/2, "Inconnu"

# ==========================================
# 2. INFERENCE & COLLECTE
# ==========================================

results = []
print(f"--- ANALYSE EN COURS ---")

for strat_name, folder in STRATEGIES.items():
    # Construction du chemin : IA_Training -> best_models_xxx -> best_model_ratio_50
    model_path = ia_dir / folder / TARGET_SUBFOLDER
    
    if not model_path.exists():
        print(f"[SKIP] {strat_name} introuvable dans {model_path}")
        continue
        
    print(f">>> Chargement {strat_name}...")
    try:
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = ResNetForImageClassification.from_pretrained(model_path).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"[ERR] Échec chargement {strat_name}: {e}")
        continue

    # Inférence
    for img_path in tqdm(test_files, leave=False, desc=f"  -> {strat_name}"):
        true_x, true_y, group_id = get_coords_and_id(img_path.name)
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = model(**inputs).logits
                preds = torch.sigmoid(logits).cpu().numpy()[0]
            
            pred_x = preds[0] * MAX_X
            pred_y = preds[1] * MAX_Y
            
            results.append({
                "Strategy": strat_name,
                "True_X": true_x, "True_Y": true_y,
                "Pred_X": pred_x, "Pred_Y": pred_y,
                "Error": np.sqrt((pred_x-true_x)**2 + (pred_y-true_y)**2),
                "Group": group_id
            })
        except: continue
    
    del model
    torch.cuda.empty_cache()

if not results:
    print("\n[STOP] Aucune donnée. Vérifiez vos dossiers modèles.")
    sys.exit()

df = pd.DataFrame(results)

# ==========================================
# 3. VISUALISATION
# ==========================================

print("\n[>] Génération des graphiques...")

# 1. Biais Spatial
try:
    g = sns.FacetGrid(df, col="Strategy", col_wrap=3, height=4)
    g.map(plt.scatter, "Pred_X", "Pred_Y", alpha=0.3, s=10, color="blue")
    for ax in g.axes.flat:
        ax.set_xlim(0, MAX_X); ax.set_ylim(MAX_Y, 0) 
        ax.plot([MAX_X/2], [MAX_Y/2], 'r+', markersize=15, label="Centre")
        ax.grid(True, alpha=0.3)
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle("1. Biais Spatial : Où tombent les prédictions ?")
    plt.savefig("BIAS_1_SPATIAL.png")
    print("[OK] BIAS_1_SPATIAL.png")
except: pass

# 2. Performance par Groupe
try:
    if df['Group'].nunique() > 1:
        plt.figure(figsize=(14, 6))
        sns.barplot(data=df, x="Group", y="Error", hue="Strategy", errorbar=None)
        plt.title("2. Performance par Identifiant (Qui est négligé ?)")
        plt.ylabel("Erreur Moyenne (px)")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig("BIAS_2_GROUPS.png")
        print("[OK] BIAS_2_GROUPS.png")
except: pass

# 3. Densité Latérale
try:
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df, x="Pred_X", hue="Strategy", fill=True, alpha=0.1, common_norm=False)
    plt.axvline(x=MAX_X/2, color='red', linestyle='--')
    plt.title("3. Biais Latéral : Gauche ou Droite ?")
    plt.xlim(0, MAX_X)
    plt.savefig("BIAS_3_LATERAL.png")
    print("[OK] BIAS_3_LATERAL.png")
except: pass

print("\n--- TERMINE ---")