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

EPOCHS = 20      
BATCH_SIZE = 64
LEARNING_RATE = 1e-4 
NUM_WORKERS = 0

SAVE_DIR = "best_models_random"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# ==========================================
# 1. CHEMINS & DATASET
# ==========================================
current_script_dir = Path(__file__).resolve().parent
dataset_dir = current_script_dir.parent.parent / "FINAL_DATASET"

print(f"--- PROTOCOLE RANDOM (1080p | MAE | BEST OF 3 RUNS) ---")
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
        # --- RÉSOLUTION 1920x1080 ---
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

# Validation sur Test B
ds_test_B = ImageRegressionDataset(test_files_B, processor)
loader_test_B = DataLoader(ds_test_B, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ==========================================
# 3. TRAINING & EVAL (MAE - L1LOSS)
# ==========================================
def train_model(train_files, desc_bar):
    model = ResNetForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    ).to(DEVICE)
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
    criterion = nn.L1Loss() # MAE
    with torch.no_grad():
        for pv, targets in loader:
            pv, targets = pv.to(DEVICE), targets.to(DEVICE)
            out = model(pv).logits
            preds = torch.sigmoid(out)
            loss = criterion(preds, targets)
            total_loss += loss.item(); count += 1
    return total_loss / count if count > 0 else 0

# ==========================================
# 4. EXECUTION
# ==========================================
aggregated_results = {0.0: []}
for r in TARGET_RATIOS: aggregated_results[r] = []

# RECORDS (Pour savoir qui est le champion actuel)
best_losses = {r: float('inf') for r in [0.0] + TARGET_RATIOS}

for run_idx in range(N_REPEATS):
    print(f"\n████ RUN {run_idx+1} / {N_REPEATS} ████")
    
    current_train_set = list(base_train_files_A)
    current_pool = list(base_pool_files_B)
    
    # Stratégie Random : on mélange le pool au début
    random.shuffle(current_pool)
    
    # --- STEP 0: BASELINE ---
    print("[Step 0] Training Baseline (A only)...")
    model_current = train_model(current_train_set, "Training Baseline")
    loss = evaluate(model_current, loader_test_B)
    aggregated_results[0.0].append(loss)
    
    # LOGIQUE DE CHAMPIONNAT : On écrase si meilleur
    if loss < best_losses[0.0]:
        best_losses[0.0] = loss
        save_path = os.path.join(SAVE_DIR, "best_baseline")
        if os.path.exists(save_path): shutil.rmtree(save_path)
        model_current.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"    [NEW RECORD] Baseline sauvegardée (MAE: {loss:.5f})")
    else:
        print(f"    [INFO] Baseline MAE: {loss:.5f} (Record actuel: {best_losses[0.0]:.5f})")

    accumulated_count = 0
    total_pool_size = len(base_pool_files_B)
    
    # --- STEP 1..N: ACTIVE LEARNING (RANDOM) ---
    for target_ratio in TARGET_RATIOS:
        target_count = int(total_pool_size * target_ratio)
        n_to_add = target_count - accumulated_count
        
        # Incremental : On ajoute les données au set existant
        if n_to_add > 0:
            new_samples = current_pool[:n_to_add]
            current_pool = current_pool[n_to_add:]
            current_train_set.extend(new_samples)
            accumulated_count += len(new_samples)
        
        desc = f"Run {run_idx+1} - Random {target_ratio*100}%"
        model_current = train_model(current_train_set, desc)
        loss = evaluate(model_current, loader_test_B)
        aggregated_results[target_ratio].append(loss)
        
        # LOGIQUE DE CHAMPIONNAT : On écrase si meilleur
        if loss < best_losses[target_ratio]:
            best_losses[target_ratio] = loss
            save_path = os.path.join(SAVE_DIR, f"best_model_ratio_{int(target_ratio*100)}")
            if os.path.exists(save_path): shutil.rmtree(save_path)
            model_current.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"    [NEW RECORD] {target_ratio*100}% sauvegardé (MAE: {loss:.5f})")
        else:
            print(f"    [INFO] {target_ratio*100}% MAE: {loss:.5f} (Record actuel: {best_losses[target_ratio]:.5f})")

# ==========================================
# 5. GRAPHIQUE & CSV
# ==========================================
print("\n[>] Génération du graphique 'benchmark_random.png'...")

final_ratios = [0.0] + TARGET_RATIOS
# Moyenne pour la courbe scientifique
random_means = [np.mean(aggregated_results[r]) for r in final_ratios]
random_stds  = [np.std(aggregated_results[r]) for r in final_ratios]
# On garde aussi les records pour info
random_bests = [best_losses[r] for r in final_ratios]

x_vals = np.array([r * 100 for r in final_ratios])

plt.figure(figsize=(10, 6))

try:
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
    spl = make_interp_spline(x_vals, random_means, k=2)
    y_smooth = np.clip(spl(x_smooth), 0, None)
    plt.plot(x_smooth, y_smooth, color='gray', linestyle='--', linewidth=2, label='Random Sampling (Moyenne)')
except:
    plt.plot(x_vals, random_means, color='gray', linestyle='--', linewidth=2, label='Random Sampling (Moyenne)')

# On affiche aussi les "Meilleurs Records" en petits points rouges pour info
plt.scatter(x_vals, random_bests, color='red', marker='*', s=80, label='Record Absolu (Sauvegardé)')

plt.errorbar(x_vals, random_means, yerr=random_stds, fmt='o', color='black', capsize=5)

plt.title(f"Random Sampling : Performance sur Domaine B\nMAE Loss")
plt.xlabel("Pourcentage de données du Domaine B ajoutées (%)")
plt.ylabel("Erreur MAE")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig("benchmark_random2.png", dpi=300)

csv_filename = "resultats_random2.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Ratio", "Mean MAE", "Std MAE", "Best MAE"] + [f"Run {i+1}" for i in range(N_REPEATS)])
    for r in final_ratios:
        vals = aggregated_results[r]
        writer.writerow([r, np.mean(vals), np.std(vals), best_losses[r]] + [f"{v:.5f}" for v in vals])
print(f"[OK] Terminé. CSV: {csv_filename}")