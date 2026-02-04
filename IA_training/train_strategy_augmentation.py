import os
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification
from torchvision import transforms # Pour l'augmentation
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- SETUP ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 🚀 STRATÉGIE 2 : ROBUST DATA AUGMENTATION ({device}) ---")

# --- CHEMINS & SETUP ---
# (Même bloc que le précédent pour les chemins...)
current_dir = Path(__file__).resolve().parent
processed_dir = current_dir.parent / "processed_frames"
path_A, path_B = processed_dir / "Domain_A", processed_dir / "Domain_B"
output_dir_A = "./modele_pretrain_source_A"

# --- DATASET ROBUSTE (C'est ici que tout change) ---
model_name = "microsoft/resnet-18"
processor = AutoImageProcessor.from_pretrained(model_name)

class RobustDataset(Dataset):
    def __init__(self, paths, processor, is_training=False):
        self.paths = paths
        self.processor = processor
        self.is_training = is_training
        
        # AUGMENTATION "ADVERSARIALE" (Simulation de mauvaises conditions)
        self.augment = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2), # Problème lumière
            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),        # Problème flou
            transforms.RandomAdjustSharpness(sharpness_factor=2),                 # Problème netteté
            # Note: On ne fait pas de flip/rotation car ça fausse les coordonnées X,Y
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        
        # Appliquer le bruit SEULEMENT pendant l'entraînement
        if self.is_training:
            img = self.augment(img)
            
        name = os.path.basename(self.paths[idx]).replace('.jpg','').split('_')
        target = torch.tensor([float(name[-2])/1280.0, float(name[-1])/720.0], dtype=torch.float32)
        return self.processor(img, return_tensors="pt")['pixel_values'].squeeze(0), target

# --- CHARGEMENT ---
paths_A = [str(x) for x in path_A.rglob("*.jpg")]
paths_B = []
for i in range(1, 13):
    if (path_B/str(i)).exists(): paths_B.extend([str(x) for x in (path_B/str(i)).glob("*.jpg")])
pool_B, test_B = train_test_split(paths_B, test_size=0.2, random_state=42)

# Le Test set reste "propre" (is_training=False) pour une évaluation juste
test_loader = DataLoader(RobustDataset(test_B, processor, is_training=False), batch_size=32)

# --- BOUCLE PRINCIPALE (Sélection Random + Entraînement Robuste) ---
RATIOS = [0.01, 0.02, 0.05, 0.10]
results = {}

for ratio in RATIOS:
    subset_size = int(len(pool_B) * ratio)
    print(f"\n>>> RATIO : {ratio*100}% ({subset_size} images)")
    
    losses = []
    for run in range(3):
        # Sélection Random (Baseline)
        random.seed(run)
        selected_B = random.sample(pool_B, subset_size)
        
        # Dataset d'entraînement ROBUSTE (is_training=True)
        train_dataset = RobustDataset(paths_A + selected_B, processor, is_training=True)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        model = ResNetForImageClassification.from_pretrained(output_dir_A).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        model.train()
        
        for epoch in tqdm(range(10), desc=f"Train Augmenté R{run}", leave=False):
            for px, tg in train_loader:
                optimizer.zero_grad()
                loss = criterion(torch.sigmoid(model(px.to(device)).logits), tg.to(device))
                loss.backward()
                optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for px, tg in test_loader:
                val_loss += criterion(torch.sigmoid(model(px.to(device)).logits), tg.to(device)).item()
        losses.append(val_loss/len(test_loader))
        print(f"   -> Score: {losses[-1]:.6f}")

    results[ratio] = np.mean(losses)
    print(f"   ✅ Moyenne: {results[ratio]:.6f}")