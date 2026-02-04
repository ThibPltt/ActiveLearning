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
from scipy.interpolate import make_interp_spline # <--- L'AJOUT SCIPY

# ==========================================
# 0. CONFIGURATION
# ==========================================
logging.set_verbosity_error()
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/resnet-18"

# Ratios d'Active Learning
TARGET_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
N_REPEATS = 3 

EPOCHS = 20      
BATCH_SIZE = 10  
LEARNING_RATE = 1e-4 

SAVE_DIR = "best_models_random"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# ==========================================
# 1. DATASET & CHEMINS (FINAL_DATASET)
# ==========================================
# On pointe vers le dossier FINAL_DATASET créé par split_folders_final.py
current_script_dir = Path(__file__).resolve().parent
# Remonte d'un cran pour trouver FINAL_DATASET s'il est à côté de ActiveLearning
dataset_dir = current_script_dir.parent.parent / "FINAL_DATASET"

print(f"--- DIAGNOSTIC CHEMINS ---")
print(f"Script situé dans : {current_script_dir}")
print(f"Dossier Dataset visé : {dataset_dir}")

dir_train_A = dataset_dir / "Domain_A_Train"
dir_pool_B  = dataset_dir / "Domain_B_Pool"
dir_test_B  = dataset_dir / "Domain_B_Test"

if not dir_train_A.exists():
    raise FileNotFoundError(f"Impossible de trouver {dir_train_A}. Avez-vous lancé split_folders_final.py ?")

# Fonctions de chargement
def load_images(folder):
    exts = ["*.jpg", "*.JPG", "*.png", "*.jpeg"]
    files = []
    for ext in exts: files.extend([str(f) for f in folder.glob(ext)])
    return sorted(list(set(files)))

# Chargement des listes
base_train_files_A = load_images(dir_train_A)
base_pool_files_B  = load_images(dir_pool_B)
test_files_B       = load_images(dir_test_B)

print(f"--- PROTOCOLE RANDOM (FINAL DATASET) ---")
print(f"Train A : {len(base_train_files_A)} images")
print(f"Pool B  : {len(base_pool_files_B)} images")
print(f"Test B  : {len(test_files_B)} images")
print("-" * 60)

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
# 3. FONCTIONS D'ENTRAINEMENT
# ==========================================
def train_model(train_files, desc_bar):
    model = ResNetForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    ).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
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
    criterion = nn.MSELoss()
    with torch.no_grad():
        for pv, targets in loader:
            pv, targets = pv.to(DEVICE), targets.to(DEVICE)
            out = model(pv).logits
            preds = torch.sigmoid(out)
            loss = criterion(preds, targets)
            total_loss += loss.item(); count += 1
    return total_loss / count if count > 0 else 0

def select_random(pool, n_samples):
    random.shuffle(pool)
    return pool[:n_samples], pool[n_samples:]

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

    accumulated_count = 0
    total_pool_size = len(base_pool_files_B)
    
    # Active Learning Loop
    for target_ratio in TARGET_RATIOS:
        target_count = int(total_pool_size * target_ratio)
        n_to_add = target_count - accumulated_count
        
        if n_to_add > 0:
            new_samples, current_pool = select_random(current_pool, n_to_add)
            current_train_set.extend(new_samples)
            accumulated_count += len(new_samples)
        
        desc = f"Run {run_idx+1} - Random {target_ratio*100}%"
        model_current = train_model(current_train_set, desc)
        loss = evaluate(model_current, loader_test_B)
        aggregated_results[target_ratio].append(loss)
        
        if loss < best_losses[target_ratio]:
            best_losses[target_ratio] = loss
            save_path = os.path.join(SAVE_DIR, f"best_model_ratio_{int(target_ratio*100)}")
            if os.path.exists(save_path): shutil.rmtree(save_path)
            model_current.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"    [NEW BEST] {target_ratio*100}% sauvegardé (Loss: {loss:.5f})")

# ==========================================
# 5. GRAPHIQUE LISSÉ (SCIPY STYLE)
# ==========================================
print("\n[>] Génération du graphique 'benchmark_random_scipy.png'...")

final_ratios = [0.0] + TARGET_RATIOS
means = [np.mean(aggregated_results[r]) if aggregated_results[r] else 0 for r in final_ratios]
stds = [np.std(aggregated_results[r]) if aggregated_results[r] else 0 for r in final_ratios]

plt.figure(figsize=(10, 6))
x_vals = np.array([r * 100 for r in final_ratios])
y_vals = np.array(means)

# LISSAGE SCIPY
try:
    # On crée 300 points pour faire une courbe douce
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
    # Spline cubique (k=3) ou quadratique (k=2)
    spl = make_interp_spline(x_vals, y_vals, k=2) 
    y_smooth = spl(x_smooth)
    # On évite que la courbe plonge sous 0 (impossible pour une Loss)
    y_smooth = np.clip(y_smooth, 0, None)
    
    plt.plot(x_smooth, y_smooth, color='gray', linestyle='-', linewidth=2, label='Random (Lissé SciPy)', alpha=0.8)
except Exception as e:
    print(f"Info: Lissage impossible ({e}), utilisation ligne brisée.")
    plt.plot(x_vals, y_vals, color='gray', linestyle='-', linewidth=2, label='Random')

# On affiche quand même les vrais points
plt.errorbar(x_vals, y_vals, yerr=stds, fmt='o', color='black', capsize=5, label='Mesures réelles')

plt.title(f"Random Sampling Performance ({N_REPEATS} runs)")
plt.xlabel("% Domain B")
plt.ylabel("Loss MSE")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("benchmark_random_scipy.png", dpi=300)

# ==========================================
# 6. CSV OUTPUT
# ==========================================
csv_filename = "resultats_random.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Ratio", "Mean", "Std"] + [f"Run {i+1}" for i in range(N_REPEATS)])
    for r in final_ratios:
        vals = aggregated_results[r]
        writer.writerow([r, np.mean(vals), np.std(vals)] + [f"{v:.5f}" for v in vals])
print(f"[OK] Terminé. CSV: {csv_filename}")