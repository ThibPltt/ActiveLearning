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

# ====================
# 0. CONFIGURATION
# ====================

# Objectif : Implémenter une stratégie d'Active Learning basée sur le Dropout pour 
# sélectionner les échantillons les plus incertains du pool.

logging.set_verbosity_error()
sys.stdout.reconfigure(encoding='utf-8')

# Vérification GPU
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

# Config Dropout
MC_PASSES = 10           
BATCH_SIZE_UNCERTAINTY = 64
DROPOUT_RATE = 0.25

EPOCHS = 20      
BATCH_SIZE = 10
LEARNING_RATE = 1e-4 

# Dossier de sauvegarde
SAVE_DIR = "best_models_dropout_0.25"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# ==========================================
# 1. DATASET & CHEMINS (FINAL_DATASET)
# ==========================================

current_script_dir = Path(__file__).resolve().parent
dataset_dir = current_script_dir.parent.parent / "FINAL_DATASET"

print(f"--- DIAGNOSTIC CHEMINS ---")
print(f"Script : {current_script_dir}")
print(f"Dataset: {dataset_dir}")

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

print(f"--- PROTOCOLE DROPOUT {DROPOUT_RATE} (1080p | MAE) ---")
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
loader_test_B = DataLoader(ds_test_B, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. FONCTIONS
# ==========================================

def train_model(train_files, desc_bar):
    model = ResNetForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    ).to(DEVICE)
    
    # Remplacement classifier avec Dropout
    hidden_size = model.config.hidden_sizes[-1]
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p=DROPOUT_RATE), 
        nn.Linear(hidden_size, 2)
    ).to(DEVICE)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss() # MAE
    
    train_ds = ImageRegressionDataset(train_files, processor)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
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
    criterion = nn.L1Loss() # MAE
    with torch.no_grad():
        for pv, targets in loader:
            pv, targets = pv.to(DEVICE), targets.to(DEVICE)
            out = model(pv).logits
            preds = torch.sigmoid(out)
            loss = criterion(preds, targets)
            total_loss += loss.item(); count += 1
    return total_loss / count if count > 0 else 0

def compute_uncertainty_dropout(model, candidate_files):
    model.train() # Force Dropout ON
    uncertainty_scores = []
    ds_pool = ImageRegressionDataset(candidate_files, processor)
    ld_pool = DataLoader(ds_pool, batch_size=BATCH_SIZE_UNCERTAINTY, shuffle=False)
    
    start_idx = 0
    with torch.no_grad():
        for pv, _ in tqdm(ld_pool, desc="Scoring Uncertainty", leave=False):
            pv = pv.to(DEVICE)
            batch_len = pv.size(0)
            batch_preds = []
            for _ in range(MC_PASSES):
                out = model(pv).logits
                preds = torch.sigmoid(out)
                batch_preds.append(preds.cpu().numpy())
            
            variances = np.var(np.stack(batch_preds), axis=0).sum(axis=1)
            for i in range(batch_len):
                uncertainty_scores.append((start_idx + i, variances[i]))
            start_idx += batch_len
            
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
        print(f"    [NEW BEST] Baseline sauvegardée (MAE: {loss_base:.5f})")
    else:
        print(f"    [INFO]     Baseline MAE: {loss_base:.5f} (Record: {best_losses[0.0]:.5f})")

    accumulated_count = 0
    total_pool_size = len(base_pool_files_B)
    
    # Active Learning
    for target_ratio in TARGET_RATIOS:
        target_count = int(total_pool_size * target_ratio)
        n_to_add = target_count - accumulated_count
        
        if n_to_add > 0:
            ranked_indices = compute_uncertainty_dropout(model_current, current_pool)
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
        
        desc = f"Run {run_idx+1} - Dropout {target_ratio*100}%"
        model_current = train_model(current_train_set, desc)
        loss = evaluate(model_current, loader_test_B)
        aggregated_results[target_ratio].append(loss)
        
        if loss < best_losses[target_ratio]:
            best_losses[target_ratio] = loss
            save_path = os.path.join(SAVE_DIR, f"best_model_ratio_{int(target_ratio*100)}")
            if os.path.exists(save_path): shutil.rmtree(save_path)
            model_current.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"    [NEW BEST] {target_ratio*100}% sauvegardé (MAE: {loss:.5f})")
        else:
            print(f"    [INFO]     {target_ratio*100}% MAE: {loss:.5f} (Record: {best_losses[target_ratio]:.5f})")

# ==========================================
# 5. GRAPHIQUE FINAL COMPARATIF
# ==========================================

print("\n[>] Génération du graphique 'benchmark_dropout_0.25.png'...")

final_ratios = [0.0] + TARGET_RATIOS
dropout_025_means = [np.mean(aggregated_results[r]) for r in final_ratios]
dropout_025_stds  = [np.std(aggregated_results[r]) for r in final_ratios]
dropout_bests = [best_losses[r] for r in final_ratios]

# --- REPERES COMPARATIFS ---
# Best Random MAE (Référence)
random_best_values = [0.22662, 0.18325, 0.14990, 0.11003, 0.07957, 0.05597, 0.03629]
# Best Dropout 0.5 MAE (Référence - pour voir si 0.25 fait mieux)
dropout_05_best_values = [0.23559, 0.21853, 0.17978, 0.12031, 0.09534, 0.07733, 0.04807]

min_len = min(len(random_best_values), len(final_ratios))
x_vals = np.array([r * 100 for r in final_ratios[:min_len]])

plt.figure(figsize=(12, 7))

# 1. RANDOM (Gris - Pointillés)
try:
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
    spl_rand = make_interp_spline(x_vals, random_best_values[:min_len], k=2)
    y_smooth_rand = np.clip(spl_rand(x_smooth), 0, None)
    plt.plot(x_smooth, y_smooth_rand, color='gray', linestyle='--', linewidth=2, label='Random (Best)', alpha=0.5)
except:
    plt.plot(x_vals, random_best_values[:min_len], color='gray', linestyle='--', label='Random (Best)')
plt.scatter(x_vals, random_best_values[:min_len], color='gray', marker='x', alpha=0.5)

# 2. DROPOUT 0.5 (Rouge - Pointillés - REFERENCE)
try:
    spl_05 = make_interp_spline(x_vals, dropout_05_best_values[:min_len], k=2)
    y_smooth_05 = np.clip(spl_05(x_smooth), 0, None)
    plt.plot(x_smooth, y_smooth_05, color='red', linestyle='--', linewidth=2, label='Dropout 0.5 (Best Ref)', alpha=0.5)
except:
    plt.plot(x_vals, dropout_05_best_values[:min_len], color='red', linestyle='--', label='Dropout 0.5 (Best Ref)')
plt.scatter(x_vals, dropout_05_best_values[:min_len], color='red', marker='x', alpha=0.5)

# 3. DROPOUT 0.25 (Vert - Ligne Pleine - ACTUEL)
try:
    spl_025 = make_interp_spline(x_vals, dropout_025_means[:min_len], k=2)
    y_smooth_025 = np.clip(spl_025(x_smooth), 0, None)
    plt.plot(x_smooth, y_smooth_025, color='green', linestyle='-', linewidth=3, label=f'Dropout {DROPOUT_RATE} (Moyenne)')
except:
    plt.plot(x_vals, dropout_025_means[:min_len], color='green', linestyle='-', label=f'Dropout {DROPOUT_RATE} (Moyenne)')

# Barres d'erreur
plt.errorbar(x_vals, dropout_025_means[:min_len], yerr=dropout_025_stds[:min_len], fmt='o', color='darkgreen', capsize=5)

# ÉTOILES (Records Actuels)
plt.scatter(x_vals, dropout_bests[:min_len], color='lime', marker='*', s=150, edgecolors='black', label=f'Record Absolu (Dropout {DROPOUT_RATE})', zorder=10)

plt.title(f"Benchmark Final : Random vs Dropout 0.5 vs Dropout 0.25 {DROPOUT_RATE}")
plt.xlabel("Pourcentage de données du Domaine B ajoutées (%)")
plt.ylabel("Loss MAE")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

# Sauvegardes
plt.savefig(f"benchmark_dropout_{DROPOUT_RATE}.png", dpi=300)

csv_filename = f"resultats_dropout_{DROPOUT_RATE}.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    # Correction Headers
    writer.writerow(["Ratio", "Mean", "Std", "Best"] + [f"Run {i+1}" for i in range(N_REPEATS)])
    for r in final_ratios:
        vals = aggregated_results[r]
        writer.writerow([r, np.mean(vals), np.std(vals), best_losses[r]] + [f"{v:.5f}" for v in vals])
print(f"[OK] Terminé. Fichiers : {csv_filename} et benchmark_dropout_{DROPOUT_RATE}.png")