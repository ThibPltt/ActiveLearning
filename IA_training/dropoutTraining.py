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
# 0. CONFIGURATION
# ==========================================
logging.set_verbosity_error()
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/resnet-18"

# Ratios d'Active Learning (Doit matcher Random)
TARGET_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
N_REPEATS = 3  # Nombre de runs complets

# Config Dropout
MC_PASSES = 10           # Nombre de passages pour l'incertitude
BATCH_SIZE_UNCERTAINTY = 32 
DROPOUT_RATE = 0.5       # >>> ICI : On force 50% de neurones éteints

EPOCHS = 20      
BATCH_SIZE = 10  
LEARNING_RATE = 1e-4 

# ==========================================
# 1. DATASET
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
# 2. PRÉPARATION DES DONNÉES (SPLIT PDF)
# ==========================================
current_script_dir = Path(__file__).resolve().parent
processed_dir = current_script_dir.parent / "processed_frames"

# --- DOMAINE A (Base de départ) ---
# >>> ICI : Reprise du Domaine A
path_A = processed_dir / "Domain_A"
if not path_A.exists(): raise FileNotFoundError(f"Manque: {path_A}")
files_A = [str(f) for f in path_A.rglob("*.jpg")]
random.shuffle(files_A)

# PDF : "Divisée en 3/4 pour le train et 1/4 pour le test"
split_A = int(len(files_A) * 0.75) 
base_train_files_A = files_A[:split_A]  # Sera le point de départ de l'entraînement
test_files_A = files_A[split_A:]

# --- DOMAINE B (Pool Active Learning) ---
# >>> ICI : Différenciation Pool vs Test
path_B = processed_dir / "Domain_B"
if not path_B.exists(): raise FileNotFoundError(f"Manque: {path_B}")
files_B = [str(f) for f in path_B.rglob("*.jpg")]
random.shuffle(files_B)

# PDF : "80% Active Learning set et 20% Test set"
split_B = int(len(files_B) * 0.80)
base_pool_files_B = files_B[:split_B]   # Le "bassin" où on va piocher avec Dropout
test_files_B = files_B[split_B:]        # Le "juge" fixe (jamais utilisé pour train)

print(f"--- PROTOCOLE ACTIVE LEARNING : MC DROPOUT ---")
print(f"Base Fixe (A - Train) : {len(base_train_files_A)} images")
print(f"Pool Variable (B)     : {len(base_pool_files_B)} images (Candidats)")
print(f"Test Set Fixe (B)     : {len(test_files_B)} images (Evaluation)")
print("-" * 60)

ds_test_B = ImageRegressionDataset(test_files_B, processor)
loader_test_B = DataLoader(ds_test_B, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 3. FONCTIONS
# ==========================================
def train_model(train_files, desc_bar):
    """Entraîne le modèle from scratch avec ajout manuel de Dropout."""
    
    # 1. Chargement standard (SANS l'argument invalide)
    model = ResNetForImageClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2, 
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    
    # 2. AJOUT MANUEL DU DROPOUT (C'est ici qu'on force l'incertitude)
    # On récupère la taille de la dernière couche cachée (512 pour ResNet-18)
    hidden_size = model.config.hidden_sizes[-1]
    
    # On remplace le classifieur par : Flatten -> DROPOUT -> Linear
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(p=DROPOUT_RATE),  # Utilise le taux défini (0.5)
        nn.Linear(hidden_size, 2)
    ).to(DEVICE)
    
    # 3. Suite classique de l'entraînement
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
    model.eval() # Ici on éteint le dropout pour avoir la "vraie" perf
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

def compute_uncertainty_dropout(model, candidate_files):
    """Calcule l'incertitude via MC Dropout."""
    # >>> ICI : On active le mode train pour que le Dropout fonctionne pendant l'inférence
    model.train() 
    
    uncertainty_scores = []
    ds_pool = ImageRegressionDataset(candidate_files, processor)
    # Batch size plus gros ici pour accélérer le scan
    ld_pool = DataLoader(ds_pool, batch_size=BATCH_SIZE_UNCERTAINTY, shuffle=False)
    
    start_idx = 0
    with torch.no_grad():
        for pv, _ in tqdm(ld_pool, desc="Scoring Uncertainty", leave=False):
            pv = pv.to(DEVICE)
            batch_len = pv.size(0)
            batch_preds = []
            
            # Monte Carlo : 10 passages par image
            for _ in range(MC_PASSES):
                out = model(pv).logits
                preds = torch.sigmoid(out)
                batch_preds.append(preds.cpu().numpy())
            
            # Variance des prédictions = Incertitude
            variances = np.var(np.stack(batch_preds), axis=0).sum(axis=1)
            
            for i in range(batch_len):
                uncertainty_scores.append((start_idx + i, variances[i]))
            start_idx += batch_len
            
    # Tri décroissant (Plus incertain en premier)
    uncertainty_scores.sort(key=lambda x: x[1], reverse=True)
    return uncertainty_scores

# ==========================================
# 4. EXECUTION (3 RUNS)
# ==========================================
aggregated_results = {0.0: []}
for r in TARGET_RATIOS: aggregated_results[r] = []

for run_idx in range(N_REPEATS):
    print(f"\n████ RUN {run_idx+1} / {N_REPEATS} ████")
    
    # 1. Reset : On repart de A propre et du Pool B complet
    current_train_set = list(base_train_files_A)
    current_pool = list(base_pool_files_B)
    
    # 2. Baseline (Zero-Shot)
    print("[Step 0] Training Baseline (A)...")
    model_current = train_model(current_train_set, "Training Baseline")
    loss_base = evaluate(model_current, loader_test_B)
    aggregated_results[0.0].append(loss_base)
    print(f"  > Baseline Loss: {loss_base:.5f}")
    
    accumulated_count = 0
    total_pool_size = len(base_pool_files_B)
    
    # 3. Boucle Active Learning
    for target_ratio in TARGET_RATIOS:
        target_count = int(total_pool_size * target_ratio)
        n_to_add = target_count - accumulated_count
        
        if n_to_add <= 0: continue
        
        # A. Sélection Dropout (Active Learning)
        # On utilise le modèle actuel pour scanner le pool restant
        ranked_indices = compute_uncertainty_dropout(model_current, current_pool)
        top_indices = [idx for idx, score in ranked_indices[:n_to_add]]
        
        # B. Transfert Pool -> Train
        new_samples = []
        remaining_pool = []
        top_indices_set = set(top_indices)
        
        for idx, f in enumerate(current_pool):
            if idx in top_indices_set:
                new_samples.append(f)
            else:
                remaining_pool.append(f)
        
        current_pool = remaining_pool
        current_train_set.extend(new_samples)
        accumulated_count += len(new_samples)
        
        print(f"  > [Ratio {target_ratio*100}%] Ajout de {len(new_samples)} samples incertains.")

        # C. Ré-entraînement
        desc = f"Run {run_idx+1} - Ratio {target_ratio*100}%"
        model_current = train_model(current_train_set, desc)
        
        # D. Evaluation
        loss = evaluate(model_current, loader_test_B)
        aggregated_results[target_ratio].append(loss)
        print(f"    -> Loss: {loss:.5f}")

# ==========================================
# 5. GRAPHIQUE FINAL
# ==========================================
print("\n[>] Génération du graphique 'benchmark_comparatif.png'...")

final_ratios = [0.0] + TARGET_RATIOS
dropout_means = []
dropout_stds = []

for r in final_ratios:
    losses = aggregated_results[r]
    dropout_means.append(np.mean(losses) if losses else 0)
    dropout_stds.append(np.std(losses) if losses else 0)

# --- DONNÉES RANDOM (EXTRAITES DE VOS LOGS) ---
random_means = [0.20460, 0.09660, 0.07259, 0.04295, 0.03793, 0.03206, 0.01708]
random_stds  = [0.00000, 0.00592, 0.00355, 0.00321, 0.01370, 0.01254, 0.00065]

# Sécurité tailles
min_len = min(len(random_means), len(final_ratios))
random_means = random_means[:min_len]
random_stds = random_stds[:min_len]

plt.figure(figsize=(12, 7))
x_vals = [r * 100 for r in final_ratios[:min_len]]

# Random (Gris)
plt.errorbar(x_vals, random_means, yerr=random_stds, 
             fmt='-o', color='gray', ecolor='darkgray', capsize=5, linewidth=2, 
             label='Random Sampling', alpha=0.6)

# Dropout (Bleu)
plt.errorbar(x_vals, dropout_means[:min_len], yerr=dropout_stds[:min_len], 
             fmt='-o', color='blue', ecolor='navy', capsize=5, linewidth=2, 
             label='Uncertainty (Dropout 50%)', zorder=5)

plt.title(f"Comparaison Active Learning : ({N_REPEATS} runs)")
plt.xlabel("Cost Ratio (%)")
plt.ylabel("Loss MSE")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.savefig("benchmark_comparatif.png", dpi=300)
print(f"[OK] Graphique sauvegardé : benchmark_comparatif.png")