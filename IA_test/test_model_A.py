import os
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification
# Fallback pour les anciennes versions
try:
    from transformers import AutoFeatureExtractor
except ImportError:
    pass
from tqdm import tqdm
import math

# ==========================================
# 1. FIXES SYSTÈME
# ==========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import certifi
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    pass

# ==========================================
# 2. CHEMINS ET MODÈLE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Test sur : {device} ---")

current_script_dir = Path(__file__).resolve().parent

# --- A. LOCALISATION DU MODÈLE ---
target_folder_name = "modele_pretrain_source_A"
path_root = current_script_dir.parent / target_folder_name
path_training = current_script_dir.parent / "IA_training" / target_folder_name

if path_root.exists():
    model_path = path_root
    print(f"✅ Modèle trouvé à la racine : {model_path}")
elif path_training.exists():
    model_path = path_training
    print(f"✅ Modèle trouvé dans IA_training : {model_path}")
else:
    print(f"\n❌ ERREUR : Le dossier '{target_folder_name}' est introuvable.")
    exit()

# --- B. LOCALISATION DES IMAGES ---
data_path = current_script_dir.parent / "processed_frames" / "Domain_A" / "video_20260113_103624"
if not data_path.exists():
    raise FileNotFoundError(f"Images introuvables ici : {data_path}")

# ==========================================
# 3. CHARGEMENT DU PROCESSEUR
# ==========================================
original_model_name = "microsoft/resnet-18"
print(f"-> Chargement du processeur standard : {original_model_name}")

try:
    processor = AutoImageProcessor.from_pretrained(original_model_name)
except:
    processor = AutoFeatureExtractor.from_pretrained(original_model_name)

# ==========================================
# 4. DATASET
# ==========================================
class ImageRegressionDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
        self.max_x = 1280.0 
        self.max_y = 720.0

    def __len__(self): return len(self.image_paths)

    def extract_coords(self, filename):
        name_no_ext = filename.replace('.jpg', '')
        parts = name_no_ext.split('_')
        try:
            x = float(parts[-2])
            y = float(parts[-1])
            return x / self.max_x, y / self.max_y
        except: return 0.0, 0.0

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        norm_x, norm_y = self.extract_coords(os.path.basename(path))
        target = torch.tensor([norm_x, norm_y], dtype=torch.float32)
        inputs = self.processor(image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), target, str(path)

all_image_paths = [str(f) for f in data_path.glob("*.jpg")]
test_dataset = ImageRegressionDataset(all_image_paths, processor)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==========================================
# 5. CHARGEMENT DU MODÈLE
# ==========================================
print("-> Chargement des poids du modèle...")
model = ResNetForImageClassification.from_pretrained(model_path).to(device)
model.eval()
criterion = nn.MSELoss() # L'outil pour calculer la MSE

# ==========================================
# 6. CALCULS (MSE + PIXELS)
# ==========================================
print(f"\n--- Test sur {len(all_image_paths)} images ---")

distances_pixels = []
total_mse_loss = 0
total_samples = 0
max_x = 1280.0
max_y = 720.0

with torch.no_grad():
    for pixel_values, targets, paths in tqdm(test_dataloader, desc="Calcul...", unit="batch"):
        pixel_values, targets = pixel_values.to(device), targets.to(device)
        
        # Prédiction
        outputs = model(pixel_values).logits
        preds = torch.sigmoid(outputs)
        
        # --- 1. CALCUL MSE LOSS (Sur GPU) ---
        loss = criterion(preds, targets)
        # On multiplie par la taille du batch pour avoir la somme exacte
        total_mse_loss += loss.item() * pixel_values.size(0)
        total_samples += pixel_values.size(0)
        
        # --- 2. CALCUL PIXELS (Sur CPU) ---
        preds_cpu = preds.cpu().numpy()
        targets_cpu = targets.cpu().numpy()
        
        for i in range(len(preds_cpu)):
            pred_x = preds_cpu[i][0] * max_x
            pred_y = preds_cpu[i][1] * max_y
            real_x = targets_cpu[i][0] * max_x
            real_y = targets_cpu[i][1] * max_y
            
            dist = math.sqrt((pred_x - real_x)**2 + (pred_y - real_y)**2)
            distances_pixels.append(dist)
            
            if len(distances_pixels) % 200 == 0:
                fname = os.path.basename(paths[i])
                print(f"\n[Img] {fname}")
                print(f"   Vrai : ({real_x:6.1f}, {real_y:6.1f})")
                print(f"   Pred : ({pred_x:6.1f}, {pred_y:6.1f})")
                print(f"   Err  : {dist:.2f} px")

# ==========================================
# 7. RÉSULTATS
# ==========================================
avg_mse = total_mse_loss / total_samples
avg_pixel_error = sum(distances_pixels) / len(distances_pixels)
min_error = min(distances_pixels)
max_error = max(distances_pixels)

print("\n" + "="*50)
print(f" RÉSULTATS : {target_folder_name}")
print("="*50)
print(f"📉 MSE LOSS       : {avg_mse:.6f}")
print("-" * 50)
print(f"📏 Erreur Pixels  : {avg_pixel_error:.2f} px")
print(f"   Min            : {min_error:.2f} px")
print(f"   Max            : {max_error:.2f} px")
print("="*50)