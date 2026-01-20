import os
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import time

# ==========================================
# 1. CONFIGURATION ET CHEMINS
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calcul du chemin pour éviter les erreurs "File Not Found"
# On part du dossier de ce script -> parent -> processed_frames
current_script_dir = Path(__file__).resolve().parent
data_path = current_script_dir.parent / "processed_frames" / "Domain_A" / "video_20260113_103624"

if not data_path.exists():
    raise FileNotFoundError(f"Dossier introuvable : {data_path}")

# Liste de toutes les images JPG
all_image_paths = [str(f) for f in data_path.glob("*.jpg")]
print(f"--- {len(all_image_paths)} images trouvées. Entraînement sur : {device} ---")

# ==========================================
# 2. LOGIQUE DU DATASET (RÉGRESSION)
# ==========================================
class ImageRegressionDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
        # Ajustez ces valeurs selon la résolution réelle de vos images (ex: 1280x720)
        # La normalisation aide énormément le modèle à converger.
        self.max_x = 1280.0 
        self.max_y = 720.0

    def __len__(self):
        return len(self.image_paths)

    def extract_coords(self, filename):
        # Format attendu : ..._X_Y.jpg
        name_no_ext = filename.replace('.jpg', '')
        parts = name_no_ext.split('_')
        x = float(parts[-2])
        y = float(parts[-1])
        return x / self.max_x, y / self.max_y  # Normalisation 0-1

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        
        # Extraction et normalisation des labels
        norm_x, norm_y = self.extract_coords(os.path.basename(path))
        target = torch.tensor([norm_x, norm_y], dtype=torch.float32)
        
        # Preprocessing image pour ResNet (Resize 224x224, Normalisation ImageNet)
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return pixel_values, target

# ==========================================
# 3. MODÈLE ET CHARGEMENT
# ==========================================
model_name = "microsoft/resnet-18"
processor = AutoFeatureExtractor.from_pretrained(model_name)

# On demande 2 sorties (X et Y)
model = ResNetForImageClassification.from_pretrained(
    model_name, 
    num_labels=2, 
    ignore_mismatched_sizes=True
).to(device)

dataset = ImageRegressionDataset(all_image_paths, processor)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# ==========================================
# 4. BOUCLE D'ENTRAÎNEMENT SIMPLIFIÉE
# ==========================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss() # Perte standard pour la régression

model.train()
epochs = 20

print("Début de l'entraînement...")
start_time = time.time()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_idx, (pixel_values, targets) in enumerate(dataloader):
        pixel_values, targets = pixel_values.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(pixel_values).logits
        predictions = torch.sigmoid(outputs)
        
        # Calcul de l'erreur
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print(f"Époque [{epoch+1}/{epochs}] - Perte Moyenne: {epoch_loss/len(dataloader):.6f}")

end_time = time.time()
print("Entraînement terminé.")

# --- SAUVEGARDE ---
output_dir = "./mon_modele_regression"

# Créer le dossier s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Sauvegarde du modèle (poids + configuration)
model.save_pretrained(output_dir)

# Sauvegarde du processeur (indispensable pour retrouver les mêmes transformations d'images)
processor.save_pretrained(output_dir)

print(f"Modèle sauvegardé avec succès dans : {output_dir}")

elapsed_time = end_time - start_time
print(f"Temps écoulé : {elapsed_time} secondes")