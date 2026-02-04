import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- SETUP ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 🚀 STRATÉGIE 3 : PROBABILISTIC REGRESSION ({device}) ---")

# --- CHEMINS (Idem) ---
current_dir = Path(__file__).resolve().parent
processed_dir = current_dir.parent / "processed_frames"
path_A, path_B = processed_dir / "Domain_A", processed_dir / "Domain_B"
output_dir_A = "./modele_pretrain_source_A"

# --- MODÈLE PROBABILISTE ---
# On change le modèle pour qu'il sorte 4 valeurs au lieu de 2
# Sorties : [Mu_X, Mu_Y, Sigma_X, Sigma_Y]
model_name = "microsoft/resnet-18"
processor = AutoImageProcessor.from_pretrained(model_name)

def get_probabilistic_model():
    # num_labels=4 : Moyenne X, Moyenne Y, Variance X, Variance Y
    model = ResNetForImageClassification.from_pretrained(
        output_dir_A, # On repart du pré-entraînement A
        num_labels=4, 
        ignore_mismatched_sizes=True
    ).to(device)
    return model

# --- LOSS FUNCTION : Gaussian NLL ---
# C'est ce qui permet d'apprendre l'incertitude
nll_loss = nn.GaussianNLLLoss()

def probabilistic_loss(outputs, targets):
    # Les sorties du modèle
    logits = outputs.logits
    mu = torch.sigmoid(logits[:, :2])  # Prédiction (0-1)
    # Variance : On utilise softplus pour s'assurer que c'est positif > 0
    var = F.softplus(logits[:, 2:]) + 1e-6 
    
    return nll_loss(mu, targets, var)

# --- SCORING D'INCERTITUDE RAPIDE ---
def get_probabilistic_uncertainty(model, paths):
    dataset = ImageRegressionDataset(paths, processor) # (Même classe Dataset que Code 1)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    
    uncertainties = []
    with torch.no_grad():
        for px, _ in tqdm(loader, desc="Scan Incertitude", leave=False):
            logits = model(px.to(device)).logits
            # La variance est directement dans les sorties 2 et 3
            var = F.softplus(logits[:, 2:]).cpu().numpy()
            # On somme la variance X et Y pour avoir un score global
            uncertainties.extend(var.sum(axis=1))
    return np.array(uncertainties)

# --- CLASS DATASET (Standard) ---
class ImageRegressionDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths = paths
        self.processor = processor
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        name = os.path.basename(self.paths[idx]).replace('.jpg','').split('_')
        target = torch.tensor([float(name[-2])/1280.0, float(name[-1])/720.0], dtype=torch.float32)
        return self.processor(img, return_tensors="pt")['pixel_values'].squeeze(0), target

# --- DATA ---
paths_A = [str(x) for x in path_A.rglob("*.jpg")]
paths_B = []
for i in range(1, 13):
    if (path_B/str(i)).exists(): paths_B.extend([str(x) for x in (path_B/str(i)).glob("*.jpg")])
pool_B, test_B = train_test_split(paths_B, test_size=0.2, random_state=42)
test_loader = DataLoader(ImageRegressionDataset(test_B, processor), batch_size=32)

# --- BOUCLE ACTIVE LEARNING ---
RATIOS = [0.01, 0.02, 0.05, 0.10]

for ratio in RATIOS:
    subset_size = int(len(pool_B) * ratio)
    print(f"\n>>> RATIO : {ratio*100}% ({subset_size} images)")
    
    # 1. Init Modèle Probabiliste
    model = get_probabilistic_model()
    
    # 2. Sélection par Variance prédite (Très rapide, pas de 10 passes)
    scores = get_probabilistic_uncertainty(model, pool_B)
    top_indices = np.argsort(scores)[::-1][:subset_size]
    selected_B = [pool_B[i] for i in top_indices]
    
    # 3. Entraînement avec Gaussian Loss
    train_loader = DataLoader(ImageRegressionDataset(paths_A + selected_B, processor), batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    for epoch in tqdm(range(10), desc="Train Probabiliste", leave=False):
        for px, tg in train_loader:
            optimizer.zero_grad()
            outputs = model(px.to(device))
            loss = probabilistic_loss(outputs, tg.to(device))
            loss.backward()
            optimizer.step()
            
    # 4. Eval (On évalue juste la précision X,Y ici)
    model.eval()
    mse_val = 0
    with torch.no_grad():
        for px, tg in test_loader:
            logits = model(px.to(device)).logits
            preds = torch.sigmoid(logits[:, :2]) # On ne regarde que X,Y pour le test
            mse_val += nn.MSELoss()(preds, tg.to(device)).item()
            
    print(f"   ✅ Score MSE: {mse_val/len(test_loader):.6f}")