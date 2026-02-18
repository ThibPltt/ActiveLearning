# utils.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from PIL import Image
import os

# Objectif : Ce fichier contient les classes et fonctions nécessaires pour préparer les données, 
# définir le modèle et entraîner le modèle de régression d'images. 
# Il est conçu pour être utilisé dans un script d'entraînement principal.
# resnet-18 est utilisé comme modèle de base pour la régression d'images.

# --- DATASET ---
class ImageRegressionDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
        self.max_x = 1920.0 
        self.max_y = 1080.0

    def __len__(self):
        return len(self.image_paths)

    def extract_coords(self, filename):
        name_no_ext = filename.replace('.jpg', '')
        parts = name_no_ext.split('_')
        x = float(parts[-2])
        y = float(parts[-1])
        return x / self.max_x, y / self.max_y

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        target = torch.tensor(self.extract_coords(os.path.basename(path)), dtype=torch.float32)
        inputs = self.processor(image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), target

# --- FONCTIONS UTILITAIRES ---
def get_model_and_processor(device):
    model_name = "microsoft/resnet-18"
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = ResNetForImageClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)
    return model, processor

def train_one_cycle(model, train_loader, device, epochs=5):
    """Entraîne le modèle pour un nombre donné d'époques"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    model.train()
    
    for epoch in range(epochs):
        for pixel_values, targets in train_loader:
            pixel_values, targets = pixel_values.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values).logits
            predictions = torch.sigmoid(outputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
    return model