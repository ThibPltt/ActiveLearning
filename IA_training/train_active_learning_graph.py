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
import matplotlib.pyplot as plt

# ==========================================
# 2. CONFIGURATION GPU (RTX 4060 Ti)
# ==========================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"--- 🚀 GPU DÉTECTÉ : {torch.cuda.get_device_name(0)} ---")
    torch.backends.cudnn.benchmark = True 
else:
    device = torch.device("cpu")
    print("--- ⚠️ ATTENTION : GPU NON DÉTECTÉ ---")

current_script_dir = Path(__file__).resolve().parent
processed_frames_dir = current_script_dir.parent / "processed_frames"
path_domain_A = processed_frames_dir / "Domain_A"
path_domain_B = processed_frames_dir / "Domain_B"
output_dir_A = "./modele_pretrain_source_A"
checkpoints_dir = "./checkpoints_active_learning"

# Paramètres
BUDGET_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
N_REPEATS = 3
EPOCHS_A = 15       
EPOCHS_ADAPT = 10
BATCH_SIZE = 32     

# Création dossier checkpoints
os.makedirs(checkpoints_dir, exist_ok=True)

# ==========================================
# 3. DONNÉES
# ==========================================
if not path_domain_A.exists() or not path_domain_B.exists():
    raise FileNotFoundError("Dossiers introuvables.")

print("\n--- Chargement ---")
domain_a_paths = []
for img in tqdm(list(path_domain_A.rglob("*.jpg")), desc="Scan A", unit="img"):
    domain_a_paths.append(str(img))

domain_b_paths = []
for i in range(1, 13):
    p = path_domain_B / str(i)
    if p.exists():
        domain_b_paths.extend([str(f) for f in p.glob("*.jpg")])

pool_b_paths, test_b_paths = train_test_split(domain_b_paths, test_size=0.2, random_state=42)
print(f"-> Pool B: {len(pool_b_paths)} | Test B: {len(test_b_paths)}")

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

test_dataloader = DataLoader(
    ImageRegressionDataset(test_b_paths, global_processor), 
    batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
)

# ==========================================
# 4. PHASE 1 : PRÉ-ENTRAÎNEMENT A (AVEC VÉRIFICATION)
# ==========================================
print("\n" + "="*60)
print(" PHASE 1 : PRÉ-ENTRAÎNEMENT A")
print("="*60)

# VÉRIFICATION ROBUSTE : Dossier existe + contient config.json
if os.path.exists(output_dir_A) and os.path.exists(os.path.join(output_dir_A, "config.json")):
    print(f"-> ✅ Modèle pré-entraîné A trouvé dans : {output_dir_A}")
    print("-> ⏩ ON SAUTE L'ENTRAÎNEMENT DU DOMAINE A (Gain de temps !)")
else:
    print("-> ⏳ Pas de modèle A trouvé (ou incomplet). Démarrage de l'entraînement...")
    model_a = ResNetForImageClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True).to(device)
    loader_a = DataLoader(
        ImageRegressionDataset(domain_a_paths, global_processor), 
        batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    optimizer = torch.optim.AdamW(model_a.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda') 
    
    model_a.train()
    with tqdm(total=len(loader_a)*EPOCHS_A, desc="Train A", unit="batch", colour="green") as pbar:
        for epoch in range(EPOCHS_A):
            for px, tg in loader_a:
                px, tg = px.to(device), tg.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model_a(px).logits
                    loss = criterion(torch.sigmoid(outputs), tg)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    # SAUVEGARDE COMPLÈTE (Modèle + Processeur)
    print(f"-> Sauvegarde du modèle A dans {output_dir_A}")
    model_a.save_pretrained(output_dir_A)
    global_processor.save_pretrained(output_dir_A)
    del model_a, loader_a, scaler
    torch.cuda.empty_cache()

# ==========================================
# 5. PHASE 2 : EXPÉRIENCE PAR RATIO
# ==========================================
results = {}

print("\n" + "="*60)
print(" PHASE 2 : EXPÉRIENCE PAR RATIO (Avec Sauvegarde & Graph)")
print("="*60)

for ratio in BUDGET_RATIOS:
    subset_size = int(len(pool_b_paths) * ratio)
    print(f"\n>>> 📊 RATIO : {ratio*100}% ({subset_size} images B)")
    
    losses_for_ratio = []
    best_loss_for_this_ratio = float('inf') 
    
    for run in range(N_REPEATS):
        
        # 1. Sélection & Dataset
        random.seed(run + int(ratio*1000)) 
        selected_b = random.sample(pool_b_paths, subset_size)
        full_paths = domain_a_paths + selected_b
        
        # 2. Rechargement propre (Depuis le modèle A vérifié)
        model = ResNetForImageClassification.from_pretrained(output_dir_A).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler('cuda')
        
        train_loader = DataLoader(
            ImageRegressionDataset(full_paths, global_processor),
            batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
        )
        
        # 3. Train
        model.train()
        desc_bar = f" Run {run+1}/{N_REPEATS}"
        with tqdm(total=len(train_loader)*EPOCHS_ADAPT, desc=desc_bar, unit="batch", colour="cyan", leave=True) as pbar:
            for epoch in range(EPOCHS_ADAPT):
                for px, tg in train_loader:
                    px, tg = px.to(device), tg.to(device)
                    optimizer.zero_grad()
                    with torch.amp.autocast('cuda'):
                        outputs = model(px).logits
                        loss = criterion(torch.sigmoid(outputs), tg)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # 4. Eval
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for px, tg in test_dataloader:
                px, tg = px.to(device), tg.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(px).logits
                    loss = criterion(torch.sigmoid(outputs), tg)
                val_loss += loss.item()
        
        avg_loss = val_loss / len(test_dataloader)
        losses_for_ratio.append(avg_loss)
        
        # --- SAUVEGARDE MEILLEUR MODELE DU RATIO ---
        if avg_loss < best_loss_for_this_ratio:
            best_loss_for_this_ratio = avg_loss
            save_name = f"best_model_{ratio*100:.1f}percent"
            save_path = os.path.join(checkpoints_dir, save_name)
            
            model.save_pretrained(save_path)
            global_processor.save_pretrained(save_path)
            print(f"     -> 💾 Nouveau record ! Modèle sauvegardé dans : {save_name}")

        print(f"     -> Score Run {run+1}: {avg_loss:.6f}")

    mean_score = np.mean(losses_for_ratio)
    std_score = np.std(losses_for_ratio)
    results[ratio] = losses_for_ratio
    print(f"   ✅ MOYENNE {ratio*100}% : {mean_score:.6f} (+/- {std_score:.6f})")

# ==========================================
# 6. RÉSULTATS & GRAPHIQUE
# ==========================================
print("\n" + "="*60)
print(f"{'Ratio B':<10} | {'Loss Moyenne':<15} | {'Std Dev':<15}")
print("-" * 60)

x_ratios = [r * 100 for r in BUDGET_RATIOS]
y_means = [np.mean(results[r]) for r in BUDGET_RATIOS]
y_stds = [np.std(results[r]) for r in BUDGET_RATIOS]

txt_out = "RÉSULTATS INDÉPENDANTS (RTX 4060 Ti)\n"
for i, r in enumerate(BUDGET_RATIOS):
    line = f"{r*100:>5.0f}%     | {y_means[i]:.6f}        | +/- {y_stds[i]:.6f}"
    print(line)
    txt_out += line + "\n"

with open("resultats_gpu_independent.txt", "w") as f:
    f.write(str(results) + "\n\n" + txt_out)

# Génération du graphique
plt.figure(figsize=(10, 6))
plt.errorbar(x_ratios, y_means, yerr=y_stds, fmt='-o', capsize=5, color='b', label='MSE Loss')
plt.title('Performance Active Learning (Domain Adaptation)')
plt.xlabel('Données Domaine B Ajoutées (%)')
plt.ylabel('Erreur MSE (plus bas = meilleur)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plot_filename = "courbe_active_learning.png"
plt.savefig(plot_filename)
print(f"\n📊 Graphique généré : {plot_filename}")

print("--- ✅ EXPÉRIENCE TERMINÉE ---")