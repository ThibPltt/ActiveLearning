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
from sklearn.preprocessing import normalize

# NOUVEAUX IMPORTS POUR LE CLUSTERING
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# ==========================================
# 0. CONFIGURATION
# ==========================================

# Objectif : Protocole de Diversité (Clustering K-Means) pour la localisation intelligente.

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
BATCH_SIZE = 10
LEARNING_RATE = 1e-4 
NUM_WORKERS = 0

SAVE_DIR = "best_models_clustering_L2"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# ==========================================
# 1. CHEMINS & DATASET
# ==========================================

current_script_dir = Path(__file__).resolve().parent
# Recherche intelligente du dataset
dataset_dir = current_script_dir.parent / "FINAL_DATASET"
if not dataset_dir.exists():
    dataset_dir = current_script_dir.parent.parent / "FINAL_DATASET"

print(f"--- PROTOCOLE DIVERSITY (CLUSTERING K-MEANS) | 1080p | MAE ---")
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
loader_test_B = DataLoader(ds_test_B, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. FONCTIONS
# ==========================================

def train_model(train_files, desc_bar):
    model = ResNetForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
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
    criterion = nn.L1Loss()
    with torch.no_grad():
        for pv, targets in loader:
            pv, targets = pv.to(DEVICE), targets.to(DEVICE)
            out = model(pv).logits
            preds = torch.sigmoid(out)
            loss = criterion(preds, targets)
            total_loss += loss.item(); count += 1
    return total_loss / count if count > 0 else 0

# --- LOGIQUE CLUSTERING ---
def get_features(model, loader):
    """Extrait les vecteurs (embeddings) avant la couche de classification"""
    model.eval()
    features_list = []
    with torch.no_grad():
        for pv, _ in tqdm(loader, desc="Extraction Features", leave=False):
            pv = pv.to(DEVICE)
            # On récupère la sortie du backbone ResNet (pooler_output)
            # C'est un vecteur de taille 512 pour ResNet18
            embedding = model.resnet(pv).pooler_output.flatten(1)
            features_list.append(embedding.cpu().numpy())
    return np.concatenate(features_list, axis=0)

def query_clustering_sampling(model, pool_files, n_to_add):
    """
    K-MEANS SAMPLING :
    1. Calcule les features de tout le pool.
    2. Fait un K-Means avec K = n_to_add (nombre d'images à ajouter).
    3. Prend l'image la plus proche du centre de chaque cluster.
    """
    if n_to_add >= len(pool_files):
        return list(range(len(pool_files))) # Si on demande tout, on prend tout

    # 1. Extraction
    ds_pool = ImageRegressionDataset(pool_files, processor)
    # Batch size plus grand pour l'extraction car pas de backprop (plus rapide)
    ld_pool = DataLoader(ds_pool, batch_size=64, shuffle=False)
    features = get_features(model, ld_pool)

    features = normalize(features, norm='l2')

    # 2. K-Means
    # On veut trouver 'n_to_add' représentants
    kmeans = KMeans(n_clusters=n_to_add, init='k-means++', n_init=20, random_state=42)
    kmeans.fit(features)

    # 3. Sélection des plus proches des centroïdes
    # argmin_min retourne l'index du point le plus proche pour chaque centre
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features)
    
    # On retourne les indices (uniques et triés)
    return sorted(list(set(closest_indices)))

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
            # ICI : Appel à la fonction CLUSTERING
            selected_indices = query_clustering_sampling(model_current, current_pool, n_to_add)
            
            new_samples = []
            remaining_pool = []
            selected_set = set(selected_indices)
            
            for idx, f in enumerate(current_pool):
                if idx in selected_set: new_samples.append(f)
                else: remaining_pool.append(f)
            
            # Mise à jour
            current_pool = remaining_pool
            current_train_set.extend(new_samples)
            accumulated_count += len(new_samples)
        
        desc = f"Run {run_idx+1} - Clustering {target_ratio*100}%"
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
# 5. GRAPHIQUE FINAL COMPARATIF (4 COURBES)
# ==========================================

print("\n[>] Génération du graphique 'benchmark_clustering_L2.png'...")

final_ratios = [0.0] + TARGET_RATIOS

# Données Clustering (Notre test actuel)
cluster_means = [np.mean(aggregated_results[r]) for r in final_ratios]
cluster_stds  = [np.std(aggregated_results[r]) for r in final_ratios]
cluster_bests = [best_losses[r] for r in final_ratios]

# --- BENCHMARKS ---
# Random Best (MAE 1080p)
random_best_values = [0.22662, 0.18325, 0.14990, 0.11003, 0.07957, 0.05597, 0.03629]

# Dropout 0.25
dropout_025_best_values = [0.23847, 0.19540, 0.15526, 0.10915, 0.08544, 0.07129, 0.04272]

# Data Augmentation 
data_aug_best_values = [0.24126, 0.20826, 0.19518, 0.17650, 0.15241, 0.09309, 0.04062]

# Clustering V1)
clustering_v1_values = [0.22838, 0.16766, 0.13846, 0.09015, 0.07230, 0.05592, 0.03622]

min_len = min(len(random_best_values), len(final_ratios))
x_vals = np.array([r * 100 for r in final_ratios[:min_len]])

plt.figure(figsize=(12, 8))

# 1. RANDOM (Gris)
plt.plot(x_vals, random_best_values[:min_len], color='gray', linestyle='--', label='Random (Ref)', alpha=0.6)
# 2. DROPOUT 0.25 (Rouge)
plt.plot(x_vals, dropout_025_best_values[:min_len], color='red', linestyle='--', label='Dropout 0.25 (Ref)', alpha=0.6)
# 3. DATA AUGMENTATION (Bleu)
plt.plot(x_vals, data_aug_best_values[:min_len], color='blue', linestyle='--', label='Data Augmentation (Ref)', alpha=0.6)
# 4. CLUSTERING V1 (Violet)
plt.plot(x_vals, clustering_v1_values[:min_len], color='purple', linestyle='-.', label='Clustering V1 (Ref)', alpha=0.7)

# 5. CLUSTERING (Orange - Ligne Pleine - ACTUEL)
try:
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
    spl_clus = make_interp_spline(x_vals, cluster_means[:min_len], k=2)
    y_smooth_clus = np.clip(spl_clus(x_smooth), 0, None)
    plt.plot(x_smooth, y_smooth_clus, color='orange', linestyle='-', linewidth=3, label='Clustering K-Means (Moyenne)')
except:
    plt.plot(x_vals, cluster_means[:min_len], color='orange', linestyle='-', linewidth=3, label='Clustering K-Means (Moyenne)')

# Barres d'erreur Clustering
plt.errorbar(x_vals, cluster_means[:min_len], yerr=cluster_stds[:min_len], fmt='o', color='darkorange', capsize=5)
# Étoiles Records Clustering
plt.scatter(x_vals, cluster_bests[:min_len], color='gold', marker='*', s=180, edgecolors='black', label='Record Absolu (Clustering)', zorder=10)

plt.title(f"Benchmark : Clustering vs Dropout vs Augmentation vs Random")
plt.xlabel("Pourcentage de données du Domaine B ajoutées (%)")
plt.ylabel("Loss MAE")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.savefig("benchmark_clustering_L2.png", dpi=300)

# CSV
csv_filename = "resultats_clustering_L2.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Ratio", "Mean", "Std", "Best"] + [f"Run {i+1}" for i in range(N_REPEATS)])
    for r in final_ratios:
        vals = aggregated_results[r]
        writer.writerow([r, np.mean(vals), np.std(vals), best_losses[r]] + [f"{v:.5f}" for v in vals])

print(f"[OK] Terminé. Fichiers : {csv_filename} et benchmark_clustering_L2.png")