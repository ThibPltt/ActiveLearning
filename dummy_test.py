import os
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 0. CONFIGURATION
# ==========================================
MAX_X = 1920.0
MAX_Y = 1080.0
N_REPEATS = 5  # On répète 5 fois chaque image pour lisser la chance

# ==========================================
# 1. CHARGEMENT
# ==========================================
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent

def find_path_recursive(root_path, target_name):
    for root, dirs, files in os.walk(root_path):
        if target_name in dirs:
            return Path(root) / target_name
    return None

dataset_dir = find_path_recursive(project_root, "FINAL_DATASET")
if not dataset_dir: raise FileNotFoundError("FINAL_DATASET introuvable.")

dir_test_B = dataset_dir / "Domain_B_Test"

# Correction des doublons (Windows .jpg + .JPG)
raw_files = list(dir_test_B.glob("*.jpg")) + list(dir_test_B.glob("*.JPG"))
test_files = sorted(list(set(raw_files))) # set() élimine les doublons

print(f"--- BASELINE RANDOM (HASARD TOTAL) ---")
print(f"Images uniques : {len(test_files)}")

def get_true_coords(filename):
    try:
        name = filename.replace('.jpg', '').replace('.JPG', '')
        parts = name.split('_')
        x = float(parts[-2]); y = float(parts[-1])
        return x, y
    except: return MAX_X/2, MAX_Y/2

# ==========================================
# 2. SIMULATION
# ==========================================
all_errors = []

for _ in range(N_REPEATS):
    for img_path in tqdm(test_files, desc="Simulation Hasard", leave=False):
        true_x, true_y = get_true_coords(img_path.name)
        
        # PRÉDICTION ALÉATOIRE (0 à 1920, 0 à 1080)
        pred_x = random.uniform(0, MAX_X)
        pred_y = random.uniform(0, MAX_Y)
        
        # Distance en Pixels
        dist = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
        all_errors.append(dist)

mean_error = np.mean(all_errors)
std_error = np.std(all_errors)

print(f"\n>>> RÉSULTATS HASARD")
print(f"    Moyenne : {mean_error:.2f} pixels")
print(f"    Écart-type : {std_error:.2f} pixels")
print(f"    Min : {np.min(all_errors):.2f} px | Max : {np.max(all_errors):.2f} px")

# ==========================================
# 3. GRAPHIQUE (HISTOGRAMME)
# ==========================================
plt.figure(figsize=(10, 6))

# Histogramme des erreurs
counts, bins, patches = plt.hist(all_errors, bins=50, color='gray', alpha=0.7, edgecolor='black', label='Distribution des Erreurs')

# Ligne verticale pour la moyenne
plt.axvline(mean_error, color='red', linestyle='dashed', linewidth=2, label=f'Moyenne : {mean_error:.0f} px')

# Décoration
plt.title(f"Baseline : Distribution des Erreurs (Stratégie Hasard Total)", fontsize=14)
plt.xlabel("Erreur (Distance en Pixels)")
plt.ylabel("Nombre d'Occurrences")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Annotation Textuelle
plt.text(mean_error + 50, max(counts)*0.9, f"Moyenne\n{mean_error:.0f} px", color='red', fontweight='bold')

plt.savefig("baseline_random_dist.png", dpi=300)
print(f"\n[OK] Graphique généré : baseline_random_dist.png")
plt.show()