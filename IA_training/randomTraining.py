import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification, logging
import random
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 0. CONFIGURATION WINDOWS SAFE
# ==========================================
logging.set_verbosity_error()
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 1. PARAMÈTRES
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/resnet-18"

# Ratios imposés
RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]

# Répétitions pour la confiance
N_REPEATS = 3  

EPOCHS = 20      
BATCH_SIZE = 10  
LEARNING_RATE = 1e-4 

# ==========================================
# 2. DATASET CLASS
# ==========================================
class ImageRegressionDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
        self.max_x = 1280.0 
        self.max_y = 720.0

    def __len__(self):
        return len(self.image_paths)

    def extract_coords(self, filename):
        try:
            name_no_ext = filename.replace('.jpg', '')
            parts = name_no_ext.split('_')
            x = float(parts[-2])
            y = float(parts[-1])
            return x / self.max_x, y / self.max_y
        except:
            return 0.5, 0.5

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        norm_x, norm_y = self.extract_coords(os.path.basename(path))
        target = torch.tensor([norm_x, norm_y], dtype=torch.float32)
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        return pixel_values, target

try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
except:
    from transformers import ConvNextImageProcessor
    processor = ConvNextImageProcessor.from_pretrained(MODEL_NAME)

# ==========================================
# 3. PRÉPARATION DES DONNÉES
# ==========================================
current_script_dir = Path(__file__).resolve().parent
processed_dir = current_script_dir.parent / "processed_frames"

# --- DOMAINE A (ON GARDE TOUT) ---
path_A = processed_dir / "Domain_A"
if not path_A.exists(): raise FileNotFoundError(f"Manque: {path_A}")
files_A = [str(f) for f in path_A.rglob("*.jpg")]
# Split Domaine A
random.shuffle(files_A)
split_A = int(len(files_A) * 0.8)
train_files_A = files_A[:split_A] # On garde ces 80% pour TOUS les entraînements
test_files_A = files_A[split_A:]

# --- DOMAINE B (ON PIOCHE DEDANS) ---
path_B = processed_dir / "Domain_B"
if not path_B.exists(): raise FileNotFoundError(f"Manque: {path_B}")
files_B = [str(f) for f in path_B.rglob("*.jpg")]
# Split Domaine B
random.shuffle(files_B)
split_B = int(len(files_B) * 0.8)
pool_files_B = files_B[:split_B] # Le pool d'Active Learning
test_files_B = files_B[split_B:] # Le Test Set B (20%)

print(f"--- PROTOCOLE JOINT (A + x% B) ---")
print(f"Base Fixe (Domaine A Train) : {len(train_files_A)} images")
print(f"Pool Variable (Domaine B)   : {len(pool_files_B)} images")
print(f"Test Set B (Evaluation)     : {len(test_files_B)} images")
print("----------------------------------\n")

# Loader pour le Test B (On évalue si l'ajout de B améliore la perf sur B)
ds_test_B = ImageRegressionDataset(test_files_B, processor)
loader_test_B = DataLoader(ds_test_B, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==========================================
# 4. FONCTIONS
# ==========================================
def train_cycle_joint(train_files_combined, desc_bar):
    """Entraîne sur le mélange A + B"""
    model = ResNetForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    ).to(DEVICE)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Dataset combiné
    train_ds = ImageRegressionDataset(train_files_combined, processor)
    # Shuffle=True est CRUCIAL ici pour bien mélanger A et B dans les batchs
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    for _ in tqdm(range(EPOCHS), desc=desc_bar, leave=False, ascii=True):
        for pv, targets in train_ld:
            pv, targets = pv.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            out = model(pv).logits
            preds = torch.sigmoid(out)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
    return model

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    count = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for pv, targets in loader:
            pv, targets = pv.to(DEVICE), targets.to(DEVICE)
            out = model(pv).logits
            preds = torch.sigmoid(out)
            loss = criterion(preds, targets)
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else 0

# ==========================================
# 5. EXECUTION
# ==========================================
print("=== ETAPE 1 : REFERENCE (ZERO-SHOT) ===")
# Performance si on entraîne QUE sur A (équivalent à Ratio B = 0%)
print("[>] Entrainement Baseline (A uniquement)...")
model_baseline = train_cycle_joint(train_files_A, "Training Baseline A")
loss_zero_shot = evaluate(model_baseline, loader_test_B)
print(f"    [OK] Zero-Shot (Train A -> Test B) : Loss = {loss_zero_shot:.5f}")


print("\n=== ETAPE 2 : BENCHMARK JOINT (TRAIN A + RANDOM B) ===")
results = {}

for idx_r, ratio in enumerate(RATIOS):
    # Combien d'images de B on ajoute ?
    n_samples_B = int(len(pool_files_B) * ratio)
    if n_samples_B < 1: n_samples_B = 1
    
    print(f"\n[>] RATIO {ratio*100}% (A + {n_samples_B} img de B) -- Etape {idx_r+1}/{len(RATIOS)}")
    
    ratio_losses = []
    
    for i in range(N_REPEATS):
        # 1. Sélection aléatoire de B
        subset_B = random.sample(pool_files_B, n_samples_B)
        
        # 2. FUSION : On combine la liste fixe A avec le subset aléatoire B
        #  "Keep Domain A... in addition to new subset of B"
        combined_train_files = train_files_A + subset_B
        
        # 3. Entraînement sur le mélange
        desc = f"Run {i+1}/{N_REPEATS} (Size: {len(combined_train_files)})"
        model_joint = train_cycle_joint(combined_train_files, desc)
        
        # 4. Évaluation sur Test B
        loss = evaluate(model_joint, loader_test_B)
        ratio_losses.append(loss)
        
        print(f"    Run {i+1}: Loss = {loss:.5f}", flush=True)
    
    mean_l = np.mean(ratio_losses)
    std_l = np.std(ratio_losses)
    results[ratio] = (mean_l, std_l)
    print(f"    [OK] Moyenne: {mean_l:.5f} (+/- {std_l:.5f})")

# ==========================================
# 6. RAPPORT FINAL
# ==========================================
print("\n" + "="*60)
print(" RESULTATS PROTOCOLE JOINT (A + RANDOM B)")
print("="*60)
print(f"{'Configuration':<30} | {'Loss (MSE)':<15} | {'Std Dev':<15}")
print("-" * 65)

# Ligne Zero-Shot
print(f"{'Baseline (A only)':<30} | {loss_zero_shot:.5f}         | {'-':<15}")

# Lignes Ratios
for r in RATIOS:
    m, s = results[r]
    label = f"A + {r*100}% B"
    print(f"{label:<30} | {m:.5f}         | {s:.5f}")

print("-" * 65)

# ==========================================
# 7. GENERATION DU GRAPHIQUE (STYLE ACTIVE LEARNING)
# ==========================================
print("\n[>] Génération du graphique 'benchmark_random.png'...")

# 1. Préparation des données
# On convertit les ratios en pourcentages pour l'axe X (ex: 0.01 -> 1.0)
x_ratios = [r * 100 for r in RATIOS] 
y_means = [results[r][0] for r in RATIOS]
y_stds = [results[r][1] for r in RATIOS]

# Création de la figure
plt.figure(figsize=(10, 6))

# 2. Tracer la Baseline (Ligne rouge pointillée horizontale = 0% de B)
# C'est votre point de départ, la performance sans aucune donnée B
plt.axhline(y=loss_zero_shot, color='red', linestyle='--', linewidth=1.5, label=f'Baseline (0% B): {loss_zero_shot:.4f}')

# 3. Tracer les lignes verticales (Grille pour les ratios)
# Ce sont les pointillés verticaux de votre image de référence
for x in x_ratios:
    plt.axvline(x=x, color='gray', linestyle=':', alpha=0.5)

# 4. Tracer la courbe "Random" (Ligne grise avec barres d'erreur)
# fmt='-o' fait une ligne avec des points, color='gray' pour imiter votre image
plt.errorbar(x_ratios, y_means, yerr=y_stds, 
             fmt='-o', color='gray', ecolor='black', capsize=5, linewidth=2, 
             label='Random Sampling')

# 5. Mise en forme du graphique
plt.title("Evolution de la Loss selon la quantité de données ajoutées (Domaine B)")
plt.xlabel("Pourcentage de données du Domaine B ajoutées (%)")
plt.ylabel("Loss (MSE) - Plus bas est meilleur")
plt.legend()
plt.grid(True, which='major', linestyle='-', alpha=0.2)

# Astuce : On inverse pas l'axe Y car c'est une Loss, 
# mais visuellement la courbe va descendre "en toboggan".

# Sauvegarde
output_plot = "benchmark_random.png"
plt.savefig(output_plot, dpi=300)
print(f"[OK] Graphique sauvegardé : {output_plot}")
plt.close()