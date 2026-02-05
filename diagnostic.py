import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification, logging
from tqdm import tqdm
import sys

# ==========================================
# 0. CONFIGURATION
# ==========================================
logging.set_verbosity_error()
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> DIAGNOSTIC DOMAIN GAP (A vs B) sur {DEVICE}")

MODEL_NAME = "microsoft/resnet-18"
BATCH_SIZE = 64
EPOCHS = 15     
LEARNING_RATE = 1e-4
NUM_WORKERS = 0

# ==========================================
# 1. CHEMINS & DATASET
# ==========================================
current_script_dir = Path(__file__).resolve().parent
dataset_dir = current_script_dir.parent / "FINAL_DATASET"
if not dataset_dir.exists():
    dataset_dir = current_script_dir.parent.parent / "FINAL_DATASET"

dir_train_A = dataset_dir / "Domain_A_Train"
dir_test_B  = dataset_dir / "Domain_B_Test"
dir_test_A  = dataset_dir / "Domain_A_Test" 

if not dir_train_A.exists():
    print(f"[ERREUR] Dossier introuvable : {dir_train_A}")
    sys.exit(1)

def load_images(folder):
    if not folder.exists(): return []
    exts = ["*.jpg", "*.JPG", "*.png", "*.jpeg"]
    files = []
    for ext in exts: files.extend([str(f) for f in folder.glob(ext)])
    return sorted(list(set(files)))

train_files_A = load_images(dir_train_A)
test_files_B  = load_images(dir_test_B)
test_files_A  = load_images(dir_test_A)

if len(test_files_A) == 0:
    print("[INFO] Création automatique du Test A (Split 80/20).")
    import random
    random.shuffle(train_files_A)
    split = int(len(train_files_A) * 0.8)
    test_files_A = train_files_A[split:]
    train_files_A = train_files_A[:split]

print(f"--- DONNÉES ---")
print(f"Train A : {len(train_files_A)} images")
print(f"Test A  : {len(test_files_A)} images")
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
        target = torch.tensor(self.extract_coords(os.path.basename(path)), dtype=torch.float32)
        inputs = self.processor(image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), target

try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
except:
    from transformers import ConvNextImageProcessor
    processor = ConvNextImageProcessor.from_pretrained(MODEL_NAME)

ld_train_A = DataLoader(ImageRegressionDataset(train_files_A, processor), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
ld_test_A  = DataLoader(ImageRegressionDataset(test_files_A, processor), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
ld_test_B  = DataLoader(ImageRegressionDataset(test_files_B, processor), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# ==========================================
# 2. ENTRAINEMENT BASELINE
# ==========================================
print("\n[ETAPE 1] Entraînement Baseline (Domaine A)...")
model = ResNetForImageClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True).to(DEVICE)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.L1Loss() 

for epoch in tqdm(range(EPOCHS), desc="Training"):
    for pv, targets in ld_train_A:
        pv, targets = pv.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        out = model(pv).logits
        preds = torch.sigmoid(out)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

# ==========================================
# 3. CALCULS ET AFFICHAGE
# ==========================================
def get_mae(loader, name):
    model.eval()
    total_loss = 0; count = 0
    criterion = nn.L1Loss()
    with torch.no_grad():
        for pv, targets in tqdm(loader, desc=f"Test {name}"):
            pv, targets = pv.to(DEVICE), targets.to(DEVICE)
            preds = torch.sigmoid(model(pv).logits)
            total_loss += criterion(preds, targets).item()
            count += 1
    return total_loss / count if count > 0 else 0

mae_A = get_mae(ld_test_A, "Domaine A")
mae_B = get_mae(ld_test_B, "Domaine B")

gap = mae_B - mae_A
multiplier = mae_B / mae_A if mae_A > 0 else 0

print("\n" + "="*60)
print(f" DIAGNOSTIC TERMINÉ")
print("="*60)
print(f"Erreur A : {mae_A:.5f}")
print(f"Erreur B : {mae_B:.5f}")
print(f"MULTIPLICATEUR : x{multiplier:.2f}")

# ==========================================
# 4. GRAPHIQUE ÉPURÉ (SANS FLÈCHE)
# ==========================================
plt.figure(figsize=(10, 7))

# Données
domains = ['Domaine A\n(Référence)', 'Domaine B\n(Cible)']
scores = [mae_A, mae_B]
colors = ['#2ca02c', '#d62728'] # Vert / Rouge

# Barres
bars = plt.bar(domains, scores, color=colors, alpha=0.85, width=0.6, edgecolor='black')

# Valeurs sur les barres
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

# BOITE D'INFO CENTRALE (Pas de flèche)
# On place le texte bien haut, au milieu
max_height = max(mae_A, mae_B)
plt.text(
    0.5, max_height * 0.6, # Position X, Y
    f"GAP: +{gap:.4f}\nLOSS x{multiplier:.1f}", 
    ha='center', va='center', 
    fontsize=20, fontweight='bold', color='black',
    bbox=dict(boxstyle="square,pad=0.5", fc="white", ec="black", lw=2)
)

plt.title(f"Diagnostic du Domain Gap (MAE)", fontsize=16, fontweight='bold')
plt.ylabel("Erreur Moyenne Absolue (MAE)", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("compare_trainA_testA_testB.png", dpi=300)
print("[OK] Graphique généré : 'compare_trainA_testA_testB.png'")