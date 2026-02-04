import os
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

# --- SETUP ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 🚀 STRATÉGIE 1 : MC DROPOUT / VARIATION ({device}) ---")

# --- CHEMINS ---
current_dir = Path(__file__).resolve().parent
processed_dir = current_dir.parent / "processed_frames"
path_A, path_B = processed_dir / "Domain_A", processed_dir / "Domain_B"
output_dir_A = current_dir.parent / "modele_pretrain_source_A"

# Vérification de sécurité
if not output_dir_A.exists():
    # Si pas trouvé à la racine, on regarde si par hasard il est dans IA_training
    output_dir_A = current_dir / "modele_pretrain_source_A"
    
if not output_dir_A.exists():
    raise FileNotFoundError(f"Impossible de trouver le modèle A ici : {output_dir_A}")

checkpoints_dir = "./checkpoints_uncertainty"
os.makedirs(checkpoints_dir, exist_ok=True)

# --- PARAMÈTRES ---
RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20]
N_PASSES = 10  # Nombre de variations par image pour calculer l'incertitude
BATCH_SIZE = 32

# --- DATASET & MODEL ---
model_name = "microsoft/resnet-18"
processor = AutoImageProcessor.from_pretrained(model_name)

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

# --- FONCTIONS CLÉS POUR LA VARIATION ---
def enable_dropout(model):
    """Active le dropout en mode inférence pour créer de la variation."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_uncertainty_score(model, paths):
    """Calcule la variance (écart-type) des prédictions sur N passes."""
    dataset = ImageRegressionDataset(paths, processor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    enable_dropout(model) # ACTIVATION DE LA VARIATION
    
    variances = []
    with torch.no_grad():
        for px, _ in tqdm(loader, desc="Calcul Variation", leave=False):
            px = px.to(device)
            preds_stack = []
            # On prédit N fois la même image
            for _ in range(N_PASSES):
                out = torch.sigmoid(model(px).logits)
                preds_stack.append(out.cpu().numpy())
            
            # Calcul de la variance entre les N prédictions
            # Shape: (N_PASSES, Batch, 2) -> Variance sur l'axe 0
            batch_var = np.var(np.stack(preds_stack), axis=0).sum(axis=1)
            variances.extend(batch_var)
    return np.array(variances)

# --- CHARGEMENT DONNÉES ---
paths_A = [str(x) for x in path_A.rglob("*.jpg")]
paths_B = []
for i in range(1, 13):
    if (path_B/str(i)).exists(): paths_B.extend([str(x) for x in (path_B/str(i)).glob("*.jpg")])
pool_B, test_B = train_test_split(paths_B, test_size=0.2, random_state=42)
test_loader = DataLoader(ImageRegressionDataset(test_B, processor), batch_size=BATCH_SIZE)

# --- BOUCLE PRINCIPALE ---
results = {}
for ratio in RATIOS:
    subset_size = int(len(pool_B) * ratio)
    print(f"\n>>> RATIO : {ratio*100}% ({subset_size} images)")
    
    losses = []
    for run in range(3):
        # 1. Charger modèle A
        model = ResNetForImageClassification.from_pretrained(output_dir_A).to(device)
        
        # 2. SÉLECTION PAR FORTE VARIATION (UNCERTAINTY SAMPLING)
        scores = get_uncertainty_score(model, pool_B)
        # On prend les indices qui ont la plus grande variance (argsort inverse)
        top_indices = np.argsort(scores)[::-1][:subset_size]
        selected_B = [pool_B[i] for i in top_indices]
        
        # 3. Entraînement
        train_loader = DataLoader(ImageRegressionDataset(paths_A + selected_B, processor), batch_size=BATCH_SIZE, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        model.train()
        
        for epoch in tqdm(range(5), desc=f"Run {run+1}", leave=False):
            for px, tg in train_loader:
                optimizer.zero_grad()
                loss = criterion(torch.sigmoid(model(px.to(device)).logits), tg.to(device))
                loss.backward()
                optimizer.step()
        
        # 4. Test
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for px, tg in test_loader:
                val_loss += criterion(torch.sigmoid(model(px.to(device)).logits), tg.to(device)).item()
        losses.append(val_loss/len(test_loader))
        print(f"   -> Score: {losses[-1]:.6f}")

    results[ratio] = np.mean(losses)
    print(f"   ✅ Moyenne: {results[ratio]:.6f}")