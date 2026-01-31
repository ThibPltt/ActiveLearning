import os
import sys

# ==========================================
# FIX OBLIGATOIRE POUR LE PROBLÈME SSL
# ==========================================
# On force Python à ignorer la configuration cassée de PostgreSQL
# et à utiliser ses propres certificats valides.
try:
    import certifi
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()
    print(f"DEBUG: Certificats SSL forcés sur : {certifi.where()}")
except ImportError:
    # Si certifi n'est pas là, on tente de désactiver la vérification (moins sûr mais fonctionnel)
    print("DEBUG: Certifi introuvable, désactivation de la vérification SSL")
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ... La suite du code (section 1. CONFIGURATION...) reste identique ...

# ==========================================
# 1. CONFIGURATION ET PARAMÈTRES
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chemins
current_script_dir = Path(__file__).resolve().parent
base_data_path = current_script_dir.parent / "processed_frames" / "Domain_B"

# Paramètres de l'Active Learning
BUDGET_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
N_REPEATS = 3       # Nombre de répétitions pour lisser les résultats
EPOCHS_PER_RUN = 10 # Nombre d'époques
BATCH_SIZE = 10     # <--- MODIFIÉ ICI (était 16)

# ==========================================
# 2. CHARGEMENT DES DONNÉES (DOSSIERS 1 à 12)
# ==========================================
if not base_data_path.exists():
    raise FileNotFoundError(f"Dossier introuvable : {base_data_path}")

print(f"--- Recherche des images dans : {base_data_path} ---")

all_image_paths = []
folders_count = 0

for i in range(1, 13):
    folder_path = base_data_path / str(i)
    if folder_path.exists():
        imgs = [str(f) for f in folder_path.glob("*.jpg")]
        if imgs:
            all_image_paths.extend(imgs)
            folders_count += 1

if len(all_image_paths) == 0:
    raise ValueError("ERREUR CRITIQUE : Aucune image trouvée dans les dossiers 1 à 12.")

print(f"-> {len(all_image_paths)} images trouvées dans {folders_count} dossiers.")

# ==========================================
# 3. SPLIT ACTIVE LEARNING (POOL vs TEST FIXE)
# ==========================================
# Test Set fixe de 20% pour comparer équitablement les résultats
pool_paths, test_paths = train_test_split(all_image_paths, test_size=0.2, random_state=42)

print(f"-> Pool d'Active Learning : {len(pool_paths)} images")
print(f"-> Test Set (Fixe)        : {len(test_paths)} images")
print("------------------------------------------------------")

# ==========================================
# 4. DATASET & UTILITAIRES
# ==========================================
model_name = "microsoft/resnet-18"
global_processor = AutoFeatureExtractor.from_pretrained(model_name)

class ImageRegressionDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
        self.max_x = 1280.0 
        self.max_y = 720.0

    def __len__(self):
        return len(self.image_paths)

    def extract_coords(self, filename):
        name_no_ext = filename.replace('.jpg', '')
        parts = name_no_ext.split('_')
        try:
            x = float(parts[-2])
            y = float(parts[-1])
            return x / self.max_x, y / self.max_y
        except (ValueError, IndexError):
            return 0.0, 0.0

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        norm_x, norm_y = self.extract_coords(os.path.basename(path))
        target = torch.tensor([norm_x, norm_y], dtype=torch.float32)
        inputs = self.processor(image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), target

def get_fresh_model():
    """Renvoie un modèle vierge pour repartir de zéro à chaque run"""
    model = ResNetForImageClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)
    return model

def evaluate_model(model, dataloader):
    """Évalue le modèle sur le Test Set avec une barre de progression"""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    with torch.no_grad():
        for pixel_values, targets in tqdm(dataloader, desc="Evaluation Test", leave=False):
            pixel_values, targets = pixel_values.to(device), targets.to(device)
            outputs = model(pixel_values).logits
            preds = torch.sigmoid(outputs)
            loss = criterion(preds, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# DataLoader de Test (Fixe)
test_dataset = ImageRegressionDataset(test_paths, global_processor)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 5. BOUCLE PRINCIPALE (EXPÉRIENCE)
# ==========================================
results = {}

print("\n=== DÉBUT DE L'EXPÉRIENCE : RANDOM SAMPLING ===")
print(f"Batch Size : {BATCH_SIZE}")
print(f"Ratios : {BUDGET_RATIOS}")
print("="*60)

for ratio in BUDGET_RATIOS:
    losses_for_ratio = []
    subset_size = int(len(pool_paths) * ratio)
    
    print(f"\n>>> BUDGET : {ratio*100}% ({subset_size} images)")
    
    for run in range(N_REPEATS):
        print(f"  Configuration Run {run+1}/{N_REPEATS}...", end=" ")
        
        # 1. SÉLECTION ALÉATOIRE
        random.seed(run + int(ratio*1000)) 
        train_subset_paths = random.sample(pool_paths, subset_size)
        
        # 2. PRÉPARATION DATALOADER
        train_dataset = ImageRegressionDataset(train_subset_paths, global_processor)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # 3. MODÈLE VIERGE
        model = get_fresh_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        # 4. ENTRAÎNEMENT
        model.train()
        total_steps = len(train_dataloader) * EPOCHS_PER_RUN
        
        print("") 
        
        # Barre de progression fluide sur le total des steps (batches)
        with tqdm(total=total_steps, desc=f"Train R{run+1}", unit="batch", leave=True) as pbar:
            for epoch in range(EPOCHS_PER_RUN):
                epoch_loss = 0
                for pixel_values, targets in train_dataloader:
                    pixel_values, targets = pixel_values.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(pixel_values).logits
                    predictions = torch.sigmoid(outputs)
                    loss = criterion(predictions, targets)
                    loss.backward()
                    optimizer.step()
                    
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 5. ÉVALUATION
        final_test_loss = evaluate_model(model, test_dataloader)
        losses_for_ratio.append(final_test_loss)
        
        print(f"  -> Résultat Run {run+1}: MSE Loss = {final_test_loss:.6f}")

    results[ratio] = losses_for_ratio

# ==========================================
# 6. RAPPORT FINAL
# ==========================================
print("\n" + "="*55)
print(" RÉSULTATS FINAUX (MSE Loss sur Test Set)")
print("="*55)
print(f"{'Budget':<10} | {'Nb Images':<10} | {'Moyenne Loss':<15} | {'Std Dev (CI)':<15}")
print("-" * 55)

txt_output = "RÉSULTATS RANDOM SAMPLING\n"
for ratio in BUDGET_RATIOS:
    avg_loss = np.mean(results[ratio])
    std_dev = np.std(results[ratio])
    n_imgs = int(len(pool_paths) * ratio)
    
    line = f"{ratio*100:>5.0f}%     | {n_imgs:<10} | {avg_loss:.6f}        | +/- {std_dev:.6f}"
    print(line)
    txt_output += line + "\n"

print("-" * 55)

with open("resultats_random.txt", "w") as f:
    f.write(str(results) + "\n\n" + txt_output)

print("Expérience terminée.")