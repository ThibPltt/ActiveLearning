import os
import sys

# ==========================================
# 1. CORRECTIFS SYSTÈME
# ==========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import certifi
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================
# 2. CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chemins
current_script_dir = Path(__file__).resolve().parent
processed_frames_dir = current_script_dir.parent / "processed_frames"
path_domain_A = processed_frames_dir / "Domain_A"
path_domain_B = processed_frames_dir / "Domain_B"
output_dir_A = "./modele_pretrain_source_A" 

# Paramètres
BUDGET_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
N_REPEATS = 3
EPOCHS_A = 15       
EPOCHS_ADAPT = 5    # Réduit à 5 car le modèle est déjà chaud ! (Gain de temps énorme)
BATCH_SIZE = 10

# ==========================================
# 3. PRÉPARATION DES DONNÉES
# ==========================================
if not path_domain_A.exists() or not path_domain_B.exists():
    raise FileNotFoundError("Dossiers Domain_A ou Domain_B introuvables.")

# --- Chargement A ---
domain_a_paths = [str(f) for f in path_domain_A.rglob("*.jpg")]
print(f"--- Domaine A : {len(domain_a_paths)} images ---")

# --- Chargement B ---
domain_b_paths = []
for i in range(1, 13):
    p = path_domain_B / str(i)
    if p.exists():
        domain_b_paths.extend([str(f) for f in p.glob("*.jpg")])

if not domain_b_paths: raise ValueError("Pas d'images dans Domaine B.")
print(f"--- Domaine B : {len(domain_b_paths)} images ---")

# Split B
pool_b_paths, test_b_paths = train_test_split(domain_b_paths, test_size=0.2, random_state=42)
print(f"-> Pool B: {len(pool_b_paths)} | Test B: {len(test_b_paths)}")

# --- Dataset ---
model_name = "microsoft/resnet-18"
global_processor = AutoImageProcessor.from_pretrained(model_name)

class ImageRegressionDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
        self.max_x = 1280.0 
        self.max_y = 720.0

    def __len__(self): return len(self.image_paths)

    def extract_coords(self, filename):
        try:
            parts = filename.replace('.jpg', '').split('_')
            return float(parts[-2]) / self.max_x, float(parts[-1]) / self.max_y
        except: return 0.0, 0.0

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        norm_x, norm_y = self.extract_coords(os.path.basename(path))
        target = torch.tensor([norm_x, norm_y], dtype=torch.float32)
        inputs = self.processor(image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), target

# Test Dataloader
test_dataloader = DataLoader(ImageRegressionDataset(test_b_paths, global_processor), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 4. PHASE 1 : PRÉ-ENTRAÎNEMENT A (SKIP SI DÉJÀ FAIT)
# ==========================================
print("\n" + "="*60)
print(" PHASE 1 : PRÉ-ENTRAÎNEMENT A")
print("="*60)

# Vérifie si le modèle existe déjà pour gagner du temps
model_path_exists = os.path.exists(output_dir_A) and os.listdir(output_dir_A)

if model_path_exists:
    print(f"-> Modèle A déjà trouvé dans {output_dir_A}. On saute l'entraînement !")
else:
    print("-> Entraînement initial sur le Domaine A (1 seule fois)...")
    model_a = ResNetForImageClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True).to(device)
    loader_a = DataLoader(ImageRegressionDataset(domain_a_paths, global_processor), batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model_a.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    model_a.train()
    for epoch in tqdm(range(EPOCHS_A), desc="Training A"):
        for px, tg in loader_a:
            optimizer.zero_grad()
            loss = criterion(torch.sigmoid(model_a(px.to(device)).logits), tg.to(device))
            loss.backward()
            optimizer.step()
            
    model_a.save_pretrained(output_dir_A)
    print("-> Modèle A sauvegardé.")
    del model_a, loader_a
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ==========================================
# 5. PHASE 2 : APPROCHE ITÉRATIVE (RAPIDE)
# ==========================================
results = {ratio: [] for ratio in BUDGET_RATIOS}

print("\n" + "="*60)
print(" PHASE 2 : ADAPTATION ITÉRATIVE (On garde le modèle en mémoire)")
print("="*60)

for run in range(N_REPEATS):
    print(f"\n--- SÉQUENCE COMPLÈTE {run+1}/{N_REPEATS} ---")
    
    # 1. On charge le modèle A (Point de départ)
    model = ResNetForImageClassification.from_pretrained(output_dir_A).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 2. On prépare le Pool
    available_pool = list(pool_b_paths) # Copie pour ne pas modifier l'original
    random.seed(run + 42)
    random.shuffle(available_pool) # On mélange le pool au début
    
    current_b_samples = []
    
    # 3. Boucle Cumulative (1% -> 2% -> 5%...)
    for ratio in BUDGET_RATIOS:
        # Combien d'images on veut au total pour ce ratio ?
        target_count = int(len(pool_b_paths) * ratio)
        needed = target_count - len(current_b_samples)
        
        if needed > 0:
            # On pioche 'needed' images dans le pool restant
            new_samples = available_pool[:needed]
            available_pool = available_pool[needed:] # On les retire du pool
            current_b_samples.extend(new_samples)
            
            print(f"   [Ratio {ratio*100:04.1f}%] Ajout de {needed} images (Total B: {len(current_b_samples)})")
        else:
            print(f"   [Ratio {ratio*100:04.1f}%] Pas d'ajout nécessaire.")

        # Création du dataset (A complet + B cumulé)
        train_loader = DataLoader(
            ImageRegressionDataset(domain_a_paths + current_b_samples, global_processor),
            batch_size=BATCH_SIZE, shuffle=True
        )
        
        # Entraînement RAPIDE (Iterative update)
        model.train()
        for epoch in range(EPOCHS_ADAPT):
            for px, tg in train_loader:
                optimizer.zero_grad()
                loss = criterion(torch.sigmoid(model(px.to(device)).logits), tg.to(device))
                loss.backward()
                optimizer.step()
        
        # Évaluation
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for px, tg in test_dataloader:
                val_loss += criterion(torch.sigmoid(model(px.to(device)).logits), tg.to(device)).item()
        
        avg_loss = val_loss / len(test_dataloader)
        results[ratio].append(avg_loss)
        print(f"    -> Loss: {avg_loss:.6f}")

# ==========================================
# 6. RÉSULTATS
# ==========================================
print("\n" + "="*60)
print(f"{'Ratio B':<10} | {'Loss Moyenne':<15} | {'Std Dev':<15}")
print("-" * 60)
txt_out = "RÉSULTATS ITÉRATIFS (CUMULATIFS)\n"
for r in BUDGET_RATIOS:
    mean = np.mean(results[r])
    std = np.std(results[r])
    line = f"{r*100:>5.0f}%     | {mean:.6f}        | +/- {std:.6f}"
    print(line)
    txt_out += line + "\n"

with open("resultats_iteratifs.txt", "w") as f:
    f.write(str(results) + "\n\n" + txt_out)