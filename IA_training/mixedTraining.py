import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification, logging
from torchvision import transforms
import random
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
import csv
from scipy.interpolate import make_interp_spline
import time

# Imports Fusionnés (Clustering + Data Aug)
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import normalize

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

# --- PARAMÈTRES CROISÉS (FIDÈLES AUX ORIGINAUX) ---
# 1. Paramètres Data Augmentation (Issus de dataAugmTraining.py)
MC_PASSES = 10                 # Nombre de passages pour l'incertitude
BATCH_SIZE_UNCERTAINTY = 64    # Batch size accéléré pour l'inférence DataAug

# 2. Paramètres Clustering L2 (Issus de clusteringTraining.py)
# (Implicites : Normalisation L2 + K-Means)

# 3. Paramètre de Croisement (Le lien entre les deux)
UNCERTAINTY_FACTOR = 3         # On garde les 3x plus incertains pour le clustering

SAVE_DIR = "best_models_mixed"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# ==========================================
# 1. CHEMINS & DATASET
# ==========================================
current_script_dir = Path(__file__).resolve().parent
dataset_dir = current_script_dir.parent / "FINAL_DATASET"
if not dataset_dir.exists():
    dataset_dir = current_script_dir.parent.parent / "FINAL_DATASET"

print(f"--- PROTOCOLE MIXTE (MC_PASSES={MC_PASSES} | FACTOR={UNCERTAINTY_FACTOR}) ---")
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

# Pipeline de Data Augmentation
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True)
])

class ImageRegressionDataset(Dataset):
    def __init__(self, image_paths, processor, transform=None, use_processor=True):
        self.image_paths = image_paths
        self.processor = processor
        self.transform = transform
        self.use_processor = use_processor
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
        
        if self.use_processor:
            inputs = self.processor(image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0), target
        else:
            return self.transform(image), target

try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
except:
    from transformers import ConvNextImageProcessor
    processor = ConvNextImageProcessor.from_pretrained(MODEL_NAME)

ds_test_B = ImageRegressionDataset(test_files_B, processor)
loader_test_B = DataLoader(ds_test_B, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 2. FONCTIONS UTILES
# ==========================================
def train_model(train_files, desc_bar):
    model = ResNetForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
    ).to(DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()
    
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

def get_features(model, files):
    """Extrait les embeddings pour le Clustering"""
    model.eval()
    ds = ImageRegressionDataset(files, processor)
    ld = DataLoader(ds, batch_size=32, shuffle=False)
    features_list = []
    
    with torch.no_grad():
        for pv, _ in tqdm(ld, desc="Extraction Features (Clustering)", leave=False):
            pv = pv.to(DEVICE)
            embedding = model.resnet(pv).pooler_output.flatten(1)
            features_list.append(embedding.cpu().numpy())
    
    return np.concatenate(features_list, axis=0)

def get_uncertainties(model, pool_files):
    """Calcule l'incertitude via Data Augmentation (Optimisé Batch)"""
    model.eval()
    uncertainties = []
    
    # On utilise le dataset en mode 'Augmentation Manuelle'
    ds_aug = ImageRegressionDataset(pool_files, processor=None, transform=augment_transform, use_processor=False)
    
    # Pour optimiser, on traite image par image mais on batch les augmentations
    # C'est plus sûr pour garder l'ordre des fichiers
    
    for i in tqdm(range(len(pool_files)), desc=f"Computing Uncertainty (MC={MC_PASSES})", leave=False):
        # 1. On charge l'image brute une fois
        img_path = pool_files[i]
        image = Image.open(img_path).convert("RGB")
        
        # 2. On crée le batch d'augmentations
        batch_tensors = []
        for _ in range(MC_PASSES):
            batch_tensors.append(augment_transform(image))
        
        # [MC_PASSES, 3, 224, 224]
        batch_input = torch.stack(batch_tensors).to(DEVICE)
        
        # 3. Inférence en un seul bloc (Rapide grâce à BATCH_SIZE_UNCERTAINTY implicite ici car MC_PASSES est petit)
        # Si MC_PASSES était > 64, il faudrait sous-batcher, mais à 10 ça passe large.
        with torch.no_grad():
            out = model(batch_input).logits
            preds = torch.sigmoid(out) # [MC_PASSES, 2]
            
        # 4. Calcul de l'écart-type
        std_dev = preds.std(dim=0).mean().item()
        uncertainties.append(std_dev)
        
    return np.array(uncertainties)

# ==========================================
# 3. STRATEGIE MIXTE (CROISEMENT DES PARAMETRES)
# ==========================================
def query_mixed_strategy(model, pool_files, n_to_add):
    if n_to_add >= len(pool_files):
        return list(range(len(pool_files)))

    # --- ETAPE 1 : FILTRAGE PAR INCERTITUDE (Paramètres DataAug) ---
    # On garde 3x le budget (UNCERTAINTY_FACTOR)
    n_candidates = min(len(pool_files), n_to_add * UNCERTAINTY_FACTOR)
    
    uncertainties = get_uncertainties(model, pool_files)
    
    # Tri décroissant (les plus incertains en premier)
    sorted_indices = np.argsort(uncertainties)[::-1]
    candidate_indices = sorted_indices[:n_candidates]
    candidate_files = [pool_files[i] for i in candidate_indices]
    
    # --- ETAPE 2 : SELECTION PAR DIVERSITÉ (Paramètres Cluster L2) ---
    features = get_features(model, candidate_files)
    features = normalize(features, norm='l2') # Normalisation L2
    
    kmeans = KMeans(n_clusters=n_to_add, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(features)
    
    closest_indices_in_candidates, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features)
    
    # Remapping vers les indices originaux
    final_indices = [candidate_indices[i] for i in closest_indices_in_candidates]
    
    return sorted(list(set(final_indices)))

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

# ==========================================
# 4. EXECUTION
# ==========================================
aggregated_results = {0.0: []}
for r in TARGET_RATIOS: aggregated_results[r] = []
best_losses = {r: float('inf') for r in [0.0] + TARGET_RATIOS}

for run_idx in range(N_REPEATS):
    print(f"\n████ RUN {run_idx+1} / {N_REPEATS} ████")
    start_time = time.time()
    
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
    
    for target_ratio in TARGET_RATIOS:
        target_count = int(total_pool_size * target_ratio)
        n_to_add = target_count - accumulated_count
        
        if n_to_add > 0:
            selected_indices = query_mixed_strategy(model_current, current_pool, n_to_add)
            
            new_samples = []
            remaining_pool = []
            selected_set = set(selected_indices)
            for idx, f in enumerate(current_pool):
                if idx in selected_set: new_samples.append(f)
                else: remaining_pool.append(f)
            
            current_pool = remaining_pool
            current_train_set.extend(new_samples)
            accumulated_count += len(new_samples)
        
        desc = f"Run {run_idx+1} - Mixed {target_ratio*100}%"
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
            print(f"    [INFO] {target_ratio*100}% MAE: {loss:.5f}")
            
    end_time = time.time()
    duration = end_time - start_time
    print(f"[TIME] Durée Run {run_idx+1}: {duration/60:.1f} min")

# ==========================================
# 5. GRAPHIQUE COMPARATIF
# ==========================================
print("\n[>] Génération graphique 'benchmark_mixed_L2_DA.png'...")
final_ratios = [0.0] + TARGET_RATIOS
mix_means = [np.mean(aggregated_results[r]) for r in final_ratios]
mix_stds  = [np.std(aggregated_results[r]) for r in final_ratios]
mix_bests = [best_losses[r] for r in final_ratios]

# Valeurs CSV injectées
random_best_values = [0.22662, 0.18325, 0.14990, 0.11003, 0.07957, 0.05597, 0.03629]
cluster_l2_vals = [0.21659, 0.16733, 0.12849, 0.09188, 0.06720, 0.05573, 0.03316]
data_aug_best_values = [0.24126, 0.20826, 0.19518, 0.17650, 0.15241, 0.09309, 0.04062]

x_vals = np.array([r * 100 for r in final_ratios])

plt.figure(figsize=(12, 8))
plt.plot(x_vals, cluster_l2_vals, color='green', linestyle='--', label='Clustering L2', alpha=0.6)
plt.plot(x_vals, data_aug_best_values, color='blue', linestyle='--', label='Data Augmentation', alpha=0.6)
plt.plot(x_vals, random_best_values, color='red', linestyle=':', label='Random', alpha=0.4)

try:
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), 300)
    spl = make_interp_spline(x_vals, mix_means, k=2)
    y_smooth = np.clip(spl(x_smooth), 0, None)
    plt.plot(x_smooth, y_smooth, color='magenta', linestyle='-', linewidth=3, label='Mixed Strategy')
except:
    plt.plot(x_vals, mix_means, color='magenta', linestyle='-', linewidth=3, label='Mixed Strategy')

plt.errorbar(x_vals, mix_means, yerr=mix_stds, fmt='o', color='purple', capsize=5)
plt.scatter(x_vals, mix_bests, color='pink', marker='*', s=200, edgecolors='black', label='Record Mix', zorder=10)

plt.title("Benchmark: Mixed Strategy vs Components")
plt.xlabel("% Domain B Added"); plt.ylabel("Loss MAE")
plt.legend(); plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig("benchmark_mixed_L2_DA.png", dpi=300)

csv_filename = "resultats_mixed_L2_DA.csv"
with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Ratio", "Mean", "Std", "Best"] + [f"Run {i+1}" for i in range(N_REPEATS)])
    for r in final_ratios:
        vals = aggregated_results[r]
        writer.writerow([r, np.mean(vals), np.std(vals), best_losses[r]] + [f"{v:.5f}" for v in vals])

print("[OK] Terminé.")