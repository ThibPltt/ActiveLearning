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
import shutil
import csv
from scipy.interpolate import make_interp_spline 

# ==========================================
# 0. CONFIGURATION
# ==========================================
logging.set_verbosity_error()
sys.stdout.reconfigure(encoding='utf-8')

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f">>> GPU ACTIVÉ : {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True 
else:
    DEVICE = torch.device("cpu")
    print(">>> ATTENTION : CPU UNIQUEMENT (LENT)")

MODEL_NAME = "microsoft/resnet-18"

# Ratios d'Active Learning
TARGET_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
N_REPEATS = 3 

# CONFIG DATA AUGMENTATION (TTA)
MC_PASSES = 10           # Nombre de variations par image pour tester l'incertitude
BATCH_SIZE_UNCERTAINTY = 64

EPOCHS = 20      
BATCH_SIZE = 10
LEARNING_RATE = 1e-4 
NUM_WORKERS = 0

SAVE_DIR = "best_models_data_aug"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# ==========================================
# 1. CHEMINS & DATASET
# ==========================================
current_script_dir = Path(__file__).resolve().parent
dataset_dir = current_script_dir.parent / "FINAL_DATASET"

print(f"--- PROTOCOLE DATA AUGMENTATION (TTA) | 1080p | MAE ---")
dir_train_A = dataset_dir / "Domain_A_Train"
dir_pool_B  = dataset_dir / "Domain_B_Pool"
dir_test_B  = dataset_dir / "Domain_B_Test"

if not dir_train_A.exists():
    raise FileNotFoundError("Impossible de localiser FINAL_DATASET.")

def load_images(folder):
    exts = ["*.jpg", "*.JPG", "*.png", "*.jpeg"]
    files = []
    for ext in exts: files.extend([str(f) for f in folder.glob(ext)])
    return sorted(list(set(files)))

base_train_files_A = load_images(dir_train_A)
base_pool_files_B  = load_images(dir_pool_B)
test_files_B       = load_images(dir_test_B)

print(f"Train A : {len(base_train_files_A)} images")
print(f"Pool B  : {len(base_pool_files_B)} images")
print(f"Test B  : {len(test_files_B)} images")
print("-" * 60)

class ImageRegressionDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
        self.max_x = 1920.0 
        self.max_y = 1080.0

    def __len__(self):
        return len(self.image_paths)

    def extract_coords(self, filename):
        try:
            name = filename.replace('.jpg', '').replace('.JPG', '')
            parts = name.split('_')
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

ds_test_B = ImageRegressionDataset(test_files_B, processor)
loader_test_B = DataLoader(ds_test_B, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ==========================================
# 3. FONCTIONS
# ==========================================
def train_model(train_files, desc_bar):
    model = ResNetForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    ).to(DEVICE)
    
    # Pas de Dropout spécifique ici, on utilise l'architecture standard
    # car l'incertitude vient des données, pas du modèle.
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss() # MAE
    
    train_ds = ImageRegressionDataset(train_files, processor)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
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
    total_loss = 0; count = 0
    criterion = nn.L1Loss()
    with torch.no_grad():
        for pv, targets in loader:
            pv, targets = pv.to(DEVICE), targets.to(DEVICE)
            out = model(pv).logits
            preds = torch.sigmoid(out)
            loss = criterion(preds, targets)
            total_loss += loss.item(); count += 1
    return total_loss / count if count > 0 else 0

def compute_uncertainty_TTA(model, candidate_files):
    """
    Test-Time Augmentation (TTA) :
    On perturbe l'image (Bruit, Contraste, Luminosité) N fois.
    Si la prédiction change beaucoup = Incertitude.
    """
    model.eval()
    uncertainty_scores = []
    
    ds_pool = ImageRegressionDataset(candidate_files, processor)
    ld_pool = DataLoader(ds_pool, batch_size=BATCH_SIZE_UNCERTAINTY, shuffle=False, num_workers=NUM_WORKERS)
    
    start_idx = 0
    
    with torch.no_grad():
        for pv, _ in tqdm(ld_pool, desc="Scoring Uncertainty (TTA)", leave=False):
            pv = pv.to(DEVICE)
            batch_len = pv.size(0)
            
            # Stockage des prédictions pour ce batch
            batch_preds_x = np.zeros((batch_len, MC_PASSES))
            batch_preds_y = np.zeros((batch_len, MC_PASSES))
            
            for k in range(MC_PASSES):
                # --- DATA AUGMENTATION SUR LE TENSEUR ---
                # On ne touche pas aux coordonnées (pas de flip/crop), 
                # juste photométrique pour voir la robustesse.
                
                augmented_pv = pv.clone()
                
                # 1. Bruit Gaussien léger
                noise = torch.randn_like(augmented_pv) * 0.05
                augmented_pv += noise
                
                # 2. Variation Luminosité (x 0.8 à 1.2)
                brightness_factor = 0.8 + (torch.rand(1, device=DEVICE) * 0.4)
                augmented_pv *= brightness_factor
                
                # Passage dans le modèle
                out = model(augmented_pv).logits
                preds = torch.sigmoid(out).cpu().numpy() # [Batch, 2]
                
                batch_preds_x[:, k] = preds[:, 0]
                batch_preds_y[:, k] = preds[:, 1]
            
            # Calcul de la variance (Incertitude)
            # Variance moyenne sur X et Y
            var_x = np.var(batch_preds_x, axis=1)
            var_y = np.var(batch_preds_y, axis=1)
            total_variance = var_x + var_y
            
            for i in range(batch_len):
                uncertainty_scores.append((start_idx + i, total_variance[i]))
            start_idx += batch_len
            
    # Tri décroissant (Plus grande variance = Plus incertain)
    uncertainty_scores.sort(key=lambda x: x[1], reverse=True)
    return uncertainty_scores

# ==========================================
# 4. EXECUTION
# ==========================================
aggregated_results = {0.0: []}
for r in TARGET_RATIOS: aggregated_results[r] = []
best_losses = {r: float('inf') for r in [0.0] + TARGET_RATIOS}

for run_idx in range(N_REPEATS):
    print(f"\n████ RUN {run_idx+1} / {N_REPEATS} ████")
    
    current_train_set = list(base_train_files_A)
    current_pool = list(base_pool_files_B)
    
    # Baseline
    print("[Step 0] Training Baseline...")
    model_current = train_model(current_train_set, "Training Baseline")
    loss_base = evaluate(model_current, loader_test_B)
    aggregated_results[0.0].append(loss_base)
    
    if loss_base < best_losses[0.0]:
        best_losses[0.0] = loss_base
        save_path = os.path.join(SAVE_DIR, "best_baseline")
        if os.path.exists(save_path): shutil.rmtree(save_path)
        model_current.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"    [NEW RECORD] Baseline sauvegardée (MAE: {loss_base:.5f})")
    else:
        print(f"    [INFO] Baseline MAE: {loss_base:.5f}")

    accumulated_count = 0
    total_pool_size = len(base_pool_files_B)
    
    # Active Learning Loop
    for target_ratio in TARGET_RATIOS:
        target_count = int(total_pool_size * target_ratio)
        n_to_add = target_count - accumulated_count
        
        if n_to_add > 0:
            # ICI : Appel à la nouvelle fonction TTA
            ranked_indices = compute_uncertainty_TTA(model_current, current_pool)
            
            top_indices = [idx for idx, score in ranked_indices[:n_to_add]]
            
            new_samples = []
            remaining_pool = []
            top_indices_set = set(top_indices)
            for idx, f in enumerate(current_pool):
                if idx in top_indices_set: new_samples.append(f)
                else: remaining_pool.append(f)
            current_pool = remaining_pool
            current_train_set.extend(new_samples)
            accumulated_count += len(new_samples)
        
        desc = f"Run {run_idx+1} - Data Aug {target_ratio*100}%"
        model_current = train_model(current_train_set, desc)
        loss = evaluate(model_current, loader_test_B)
        aggregated_results[target_ratio].append(loss)
        
        if loss < best_losses[target_ratio]:
            best_losses[target_ratio] = loss
            save_path = os.path.join(SAVE_DIR, f"best_model_ratio_{int(target_ratio*100)}")
            if os.path.exists(save_path): shutil.rmtree(save_path)
            model_current.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"    [NEW RECORD] {target_ratio*100}% sauvegardé (MAE: {loss:.5f})")
        else:
            print(f"    [INFO] {target_ratio*100}% MAE: {loss:.5f}")

# ==========================================
# 5. GRAPHIQUE FINAL COMPARATIF
# ==========================================
print("\n[>] Génération du graphique 'benchmark_data_aug.png'...")

final_ratios = [0.0] + TARGET_RATIOS

# Données Data Augmentation (TTA)
tta_means = [np.mean(aggregated_results[r]) for r in final_ratios]
tta_stds  = [np.std(aggregated_results[r]) for r in final_ratios]
tta_bests = [best_losses[r] for r in final_ratios]

# --- ANCIENNES VALEURS A COMPARER ---
# Random Best
random_best_values = [0.22662, 0.18325, 0.14990, 0.11003, 0.07957, 0.05597, 0.03629]

# Dropout 0.25
dropout_025_best_values = [0.23847, 0.19540, 0.15526, 0.10915, 0.08544, 0.07129, 0.04272] 

min_len = min(len(random_best_values), len(final_ratios))
x_vals = np.array([r * 100 for r in final_ratios[:min_len]])

plt.figure(figsize=(12, 7))

# 1. RANDOM (Gris)
plt.plot(x_vals, random_best_values[:min_len], color='gray', linestyle='--', label='Random (Best)')

# 3. DROPOUT 0.25 (Vert)
plt.plot(x_vals, dropout_025_best_values[:min_len], color='green', linestyle='--', label='Dropout 0.25 (Best)')

# 4. DATA AUGMENTATION (Bleu - Ligne Pleine - Notre Test)
try:
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
    spl_tta = make_interp_spline(x_vals, tta_means[:min_len], k=2)
    y_smooth_tta = np.clip(spl_tta(x_smooth), 0, None)
    plt.plot(x_smooth, y_smooth_tta, color='blue', linestyle='-', linewidth=3, label='Data Augmentation (Moyenne)')
except:
    plt.plot(x_vals, tta_means[:min_len], color='blue', linestyle='-', linewidth=3, label='Data Augmentation (Moyenne)')

# Barres d'erreur TTA
plt.errorbar(x_vals, tta_means[:min_len], yerr=tta_stds[:min_len], fmt='o', color='navy', capsize=5)
# Étoiles Records TTA
plt.scatter(x_vals, tta_bests[:min_len], color='cyan', marker='*', s=150, edgecolors='black', label='Record Absolu (TTA)', zorder=10)

plt.title(f"Benchmark : Data Augmentation vs Dropout 0.25 vs Random")
plt.xlabel("Pourcentage de données du Domaine B ajoutées (%)")
plt.ylabel("Loss MAE (Plus bas est meilleur)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.savefig("benchmark_data_aug.png", dpi=300)

# CSV
csv_filename = "resultats_data_aug.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Ratio", "Mean", "Std", "Best"] + [f"Run {i+1}" for i in range(N_REPEATS)])
    for r in final_ratios:
        vals = aggregated_results[r]
        writer.writerow([r, np.mean(vals), np.std(vals), best_losses[r]] + [f"{v:.5f}" for v in vals])

print(f"[OK] Terminé. Fichiers : {csv_filename} et benchmark_data_aug.png")