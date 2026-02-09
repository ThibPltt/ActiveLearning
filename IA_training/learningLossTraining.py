import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import time  # <--- AJOUT POUR LE CHRONO

# ==========================================
# 0. CONFIGURATION
# ==========================================
logging.set_verbosity_error()
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> GPU ACTIVÉ : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

MODEL_NAME = "microsoft/resnet-18"
TARGET_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
N_REPEATS = 3 

EPOCHS = 20      
BATCH_SIZE = 10  
LEARNING_RATE = 1e-4 
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9

# Paramètres spécifiques "Learning Loss" (Slide 19)
LAMBDA_LOSS = 1.0  
MARGIN = 1.0       

SAVE_DIR = "best_models_learning_loss"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# ==========================================
# 1. CHEMINS & DATASET
# ==========================================
current_script_dir = Path(__file__).resolve().parent
dataset_dir = current_script_dir.parent / "FINAL_DATASET"
if not dataset_dir.exists():
    dataset_dir = current_script_dir.parent.parent / "FINAL_DATASET"

print(f"--- PROTOCOLE LEARNING LOSS (CVPR 2019) | 1080p ---")
dir_train_A = dataset_dir / "Domain_A_Train"
dir_pool_B  = dataset_dir / "Domain_B_Pool"
dir_test_B  = dataset_dir / "Domain_B_Test"

if not dir_train_A.exists(): raise FileNotFoundError("Dataset introuvable.")

def load_images(folder):
    exts = ["*.jpg", "*.JPG", "*.png", "*.jpeg"]
    files = []
    for ext in exts: files.extend([str(f) for f in folder.glob(ext)])
    return sorted(list(set(files)))

base_train_files_A = load_images(dir_train_A)
base_pool_files_B  = load_images(dir_pool_B)
test_files_B       = load_images(dir_test_B)

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
            x = float(parts[-2]); y = float(parts[-1])
            return x / self.max_x, y / self.max_y
        except: return 0.5, 0.5

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        target = torch.tensor(self.extract_coords(os.path.basename(path)), dtype=torch.float32)
        inputs = self.processor(image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), target

try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
except:
    from transformers import ConvNextImageProcessor
    processor = ConvNextImageProcessor.from_pretrained(MODEL_NAME)

ds_test_B = ImageRegressionDataset(test_files_B, processor)
loader_test_B = DataLoader(ds_test_B, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 2. ARCHITECTURE DU MODULE (SLIDE 17)
# ==========================================
class LossPredictionModule(nn.Module):
    def __init__(self): 
        super(LossPredictionModule, self).__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.FC1 = nn.Linear(64, 128)
        self.FC2 = nn.Linear(128, 128)
        self.FC3 = nn.Linear(256, 128)
        self.FC4 = nn.Linear(512, 128)
        self.linear = nn.Linear(4 * 128, 1) 

    def forward(self, features):
        out1 = self.GAP(features[0]).view(features[0].size(0), -1)
        out1 = F.relu(self.FC1(out1))
        out2 = self.GAP(features[1]).view(features[1].size(0), -1)
        out2 = F.relu(self.FC2(out2))
        out3 = self.GAP(features[2]).view(features[2].size(0), -1)
        out3 = F.relu(self.FC3(out3))
        out4 = self.GAP(features[3]).view(features[3].size(0), -1)
        out4 = F.relu(self.FC4(out4))
        out = torch.cat((out1, out2, out3, out4), 1)
        out = self.linear(out)
        return out

class ResNetWithFeatures(nn.Module):
    def __init__(self):
        super(ResNetWithFeatures, self).__init__()
        self.backbone = ResNetForImageClassification.from_pretrained(
            MODEL_NAME, num_labels=2, output_hidden_states=True, ignore_mismatched_sizes=True
        )
        
    def forward(self, x):
        outputs = self.backbone(x)
        logits = outputs.logits
        features = outputs.hidden_states[1:] 
        return logits, features

# ==========================================
# 3. LOSS FUNCTION (SLIDE 18)
# ==========================================
def LossPredLoss(input, target, margin=1.0):
    input = input.view(-1) 
    target = target.view(-1) 
    half_size = input.size(0) // 2
    if half_size == 0: return torch.tensor(0.0).to(DEVICE)
    
    input_1 = input[:half_size]
    input_2 = input[half_size:2*half_size]
    target_1 = target[:half_size]
    target_2 = target[half_size:2*half_size]
    
    one = 2 * torch.sign(torch.clamp(target_1 - target_2, min=0)) - 1 
    loss = torch.clamp(margin - one * (input_1 - input_2), min=0)
    loss = loss.mean()
    return loss

# ==========================================
# 4. TRAINING & SELECTION
# ==========================================
def train_learning_loss(train_files, desc_bar):
    model = ResNetWithFeatures().to(DEVICE)
    module = LossPredictionModule().to(DEVICE)
    model.train()
    module.train()
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(module.parameters()), 
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion_task = nn.L1Loss(reduction='none') 
    
    train_ds = ImageRegressionDataset(train_files, processor)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    for _ in tqdm(range(EPOCHS), desc=desc_bar, leave=False, ascii=True):
        for pv, targets in train_ld:
            pv, targets = pv.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
            preds_logits, features = model(pv)
            preds = torch.sigmoid(preds_logits)
            target_loss = criterion_task(preds, targets) 
            target_loss = torch.mean(target_loss, dim=1) 
            
            pred_loss = module(features) 
            loss_module = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            loss_task = torch.mean(target_loss)
            
            total_loss = loss_task + LAMBDA_LOSS * loss_module
            total_loss.backward()
            optimizer.step()
    return model, module

def query_learning_loss(model, module, pool_files, n_to_add):
    model.eval()
    module.eval()
    ds_pool = ImageRegressionDataset(pool_files, processor)
    ld_pool = DataLoader(ds_pool, batch_size=BATCH_SIZE, shuffle=False)
    all_pred_losses = []
    
    with torch.no_grad():
        for pv, _ in tqdm(ld_pool, desc="Predicting Loss on Pool", leave=False):
            pv = pv.to(DEVICE)
            _, features = model(pv)
            pred_l = module(features)
            all_pred_losses.extend(pred_l.cpu().numpy().flatten())
            
    sorted_indices = np.argsort(all_pred_losses)[::-1] 
    top_indices = sorted_indices[:n_to_add]
    return list(top_indices)

def evaluate(model, loader):
    model.eval() 
    total_loss = 0; count = 0
    criterion = nn.L1Loss()
    with torch.no_grad():
        for pv, targets in loader:
            pv, targets = pv.to(DEVICE), targets.to(DEVICE)
            logits, _ = model(pv)
            preds = torch.sigmoid(logits)
            loss = criterion(preds, targets)
            total_loss += loss.item(); count += 1
    return total_loss / count if count > 0 else 0

# ==========================================
# 5. EXECUTION
# ==========================================
aggregated_results = {0.0: []}
for r in TARGET_RATIOS: aggregated_results[r] = []
best_losses = {r: float('inf') for r in [0.0] + TARGET_RATIOS}

for run_idx in range(N_REPEATS):
    print(f"\n████ RUN {run_idx+1} / {N_REPEATS} ████")
    start_time = time.time() # CHRONO DEBUT
    
    current_train_set = list(base_train_files_A)
    current_pool = list(base_pool_files_B)
    
    print("[Step 0] Training Baseline (Learning Loss Style)...")
    model_curr, module_curr = train_learning_loss(current_train_set, "Baseline")
    loss = evaluate(model_curr, loader_test_B)
    aggregated_results[0.0].append(loss)
    
    if loss < best_losses[0.0]:
        best_losses[0.0] = loss
        save_path = os.path.join(SAVE_DIR, "best_baseline")
        if os.path.exists(save_path): shutil.rmtree(save_path)
        model_curr.backbone.save_pretrained(save_path) # Sauvegarde du cerveau utile
        processor.save_pretrained(save_path)

    accumulated_count = 0
    total_pool_size = len(base_pool_files_B)
    
    for target_ratio in TARGET_RATIOS:
        target_count = int(total_pool_size * target_ratio)
        n_to_add = target_count - accumulated_count
        
        if n_to_add > 0:
            selected_indices = query_learning_loss(model_curr, module_curr, current_pool, n_to_add)
            new_samples = []
            remaining_pool = []
            selected_set = set(selected_indices)
            for idx, f in enumerate(current_pool):
                if idx in selected_set: new_samples.append(f)
                else: remaining_pool.append(f)
            
            current_pool = remaining_pool
            current_train_set.extend(new_samples)
            accumulated_count += len(new_samples)
        
        desc = f"Run {run_idx+1} - LL {target_ratio*100}%"
        model_curr, module_curr = train_learning_loss(current_train_set, desc)
        loss = evaluate(model_curr, loader_test_B)
        aggregated_results[target_ratio].append(loss)
        
        if loss < best_losses[target_ratio]:
            best_losses[target_ratio] = loss
            save_path = os.path.join(SAVE_DIR, f"best_model_ratio_{int(target_ratio*100)}")
            if os.path.exists(save_path): shutil.rmtree(save_path)
            model_curr.backbone.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"    [NEW BEST] {target_ratio*100}% sauvegardé (MAE: {loss:.5f})")
        else:
            print(f"    [INFO] {target_ratio*100}% MAE: {loss:.5f}")
    
    end_time = time.time() # CHRONO FIN
    duration = end_time - start_time
    print(f"[TIME] Durée Run {run_idx+1}: {duration/60:.1f} min")

# ==========================================
# 6. GRAPHIQUE COMPARATIF
# ==========================================
print("\n[>] Génération graphique 'benchmark_learning_loss.png'...")
final_ratios = [0.0] + TARGET_RATIOS
ll_means = [np.mean(aggregated_results[r]) for r in final_ratios]
ll_stds  = [np.std(aggregated_results[r]) for r in final_ratios]
ll_bests = [best_losses[r] for r in final_ratios]

random_best_values = [0.22662, 0.18325, 0.14990, 0.11003, 0.07957, 0.05597, 0.03629]
cluster_l2_vals = [0.21659, 0.16733, 0.12849, 0.09188, 0.06720, 0.05573, 0.03316]
dropout_025_best_values = [0.23847, 0.19540, 0.15526, 0.10915, 0.08544, 0.07129, 0.04272]

x_vals = np.array([r * 100 for r in final_ratios])

plt.figure(figsize=(12, 8))
plt.plot(x_vals, random_best_values, color='gray', linestyle='--', label='Random')
plt.plot(x_vals, cluster_l2_vals, color='orange', linestyle='--', label='Clustering V2')
plt.plot(x_vals, dropout_025_best_values, color='blue', linestyle='--', label='Dropout 0.25')

try:
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
    spl = make_interp_spline(x_vals, ll_means, k=2)
    y_smooth = np.clip(spl(x_smooth), 0, None)
    plt.plot(x_smooth, y_smooth, color='crimson', linestyle='-', linewidth=3, label='Learning Loss (Ours)')
except:
    plt.plot(x_vals, ll_means, color='crimson', linestyle='-', linewidth=3, label='Learning Loss (Ours)')

plt.errorbar(x_vals, ll_means, yerr=ll_stds, fmt='o', color='darkred', capsize=5)
plt.scatter(x_vals, ll_bests, color='pink', marker='*', s=200, edgecolors='black', label='Record LL', zorder=10)

plt.title("Benchmark: Learning Loss (CVPR 2019) vs Others")
plt.xlabel("% Domain B Added"); plt.ylabel("Loss MAE")
plt.legend(); plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig("benchmark_learning_loss.png", dpi=300)

csv_filename = "resultats_learning_loss.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Ratio", "Mean", "Std", "Best"] + [f"Run {i+1}" for i in range(N_REPEATS)])
    for r in final_ratios:
        vals = aggregated_results[r]
        writer.writerow([r, np.mean(vals), np.std(vals), best_losses[r]] + [f"{v:.5f}" for v in vals])

print("[OK] Terminé.")