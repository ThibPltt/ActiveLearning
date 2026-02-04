import os
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification, logging
import random
import sys
from tqdm import tqdm

# ==========================================
# 0. CONFIGURATION
# ==========================================
logging.set_verbosity_error()
sys.stdout.reconfigure(encoding='utf-8')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/resnet-18"
BATCH_SIZE = 10
EPOCHS = 15  # Suffisant pour que A devienne bon

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


# ... (Imports et Config identiques) ...

# ==========================================
# 2. CHARGEMENT DEPUIS FINAL_DATASET
# ==========================================
current_script_dir = Path(__file__).resolve().parent
# On pointe vers le nouveau dossier
dataset_dir = current_script_dir.parent / "FINAL_DATASET"

print("--- CHARGEMENT DES DONNÉES FIGÉES ---")

# Chargement direct sans calcul de split
path_train_A = dataset_dir / "Domain_A_Train"
path_test_A  = dataset_dir / "Domain_A_Test"
path_test_B  = dataset_dir / "Domain_B_Test" # On utilise le vrai test set B

# Vérification
if not path_train_A.exists():
    raise FileNotFoundError("Lancez d'abord split_folders.py !")

train_files_A = [str(f) for f in path_train_A.glob("*.jpg")]
test_files_A  = [str(f) for f in path_test_A.glob("*.jpg")]
test_files_B  = [str(f) for f in path_test_B.glob("*.jpg")]

print(f"Domaine A (Train) : {len(train_files_A)} images")
print(f"Domaine A (Test)  : {len(test_files_A)} images")
print(f"Domaine B (Test)  : {len(test_files_B)} images")
print("-" * 60)

# ... (Le reste du code reste IDENTIQUE : Entraînement et Evaluate) ...

# ==========================================
# 3. ENTRAINEMENT SUR A
# ==========================================
print("\n[ETAPE 1] Entraînement sur les 3/4 du Domaine A...")
print("L'objectif est que le modèle devienne excellent sur A.")

model = ResNetForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True
).to(DEVICE)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

train_ds = ImageRegressionDataset(train_files_A, processor)
train_ld = DataLoader(train_ds, batch_size=10, shuffle=True)

for epoch in range(EPOCHS):
    total_loss = 0
    for pv, targets in tqdm(train_ld, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
        pv, targets = pv.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        out = model(pv).logits
        preds = torch.sigmoid(out)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

# ==========================================
# 4. VERIFICATION (TEST A vs TEST B)
# ==========================================
def evaluate_files(model, files, desc):
    ds = ImageRegressionDataset(files, processor)
    ld = DataLoader(ds, batch_size=10, shuffle=False)
    model.eval()
    total_loss = 0
    count = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for pv, targets in tqdm(ld, desc=desc):
            pv, targets = pv.to(DEVICE), targets.to(DEVICE)
            out = model(pv).logits
            preds = torch.sigmoid(out)
            loss = criterion(preds, targets)
            total_loss += loss.item()
            count += 1
    return total_loss / count if count > 0 else 0

print("\n" + "="*60)
print(" RÉSULTATS DU DOMAIN GAP")
print("="*60)

# TEST 1 : A -> A (Doit être BON)
loss_A = evaluate_files(model, test_files_A, "Test sur A (1/4)")
print(f"\n1. TEST SUR LE DOMAINE A (Attendu: TRÈS BON / FAIBLE LOSS)")
print(f"   MSE Loss = {loss_A:.5f}")

if loss_A < 0.05:
    print("   -> [OK] Le modèle a bien appris son domaine d'origine.")
else:
    print("   -> [ATTENTION] Le modèle galère même sur A. Vérifiez vos données A.")

# TEST 2 : A -> B (Doit être NUL)
loss_B = evaluate_files(model, test_files_B, "Test sur B (Zero-Shot)")
print(f"\n2. TEST SUR LE DOMAINE B (Attendu: TRÈS NUL / HAUTE LOSS)")
print(f"   MSE Loss = {loss_B:.5f}")

if loss_B > 0.10:
    print(f"   -> [OK] CONFIRMÉ : Le modèle est perdu sur B.")
    print(f"   -> Domain Gap = {loss_B / loss_A:.1f}x plus d'erreur sur B que sur A.")
else:
    print("   -> [BIZARRE] Le modèle est déjà bon sur B sans entraînement ??")

print("="*60)