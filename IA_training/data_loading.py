from torch.utils.data import Dataset, DataLoader
import os
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
import random

class ImageRegressionDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
        self.max_x = 1920.0 
        self.max_y = 1080.0

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
    




class ActiveLearningDataset(Dataset):
    def __init__(self, image_paths, processor, return_labels=False):
        self.image_paths = image_paths
        self.processor = processor
        self.return_labels = return_labels
        self.max_x, self.max_y = 1920.0, 1080.0

    def __len__(self):
        return len(self.image_paths)

    def extract_coords(self, filename):
        name_no_ext = filename.replace('.jpg', '')
        parts = name_no_ext.split('_')
        return float(parts[-2]) / self.max_x, float(parts[-1]) / self.max_y

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        
        # Preprocessing
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)

        if self.return_labels:
            norm_x, norm_y = self.extract_coords(os.path.basename(path))
            target = torch.tensor([norm_x, norm_y], dtype=torch.float32)
            return pixel_values, target
        
        return pixel_values # Retourne juste l'image pour l'inférence AL
    



class UncertaintyDataset(Dataset):
    def __init__(self, file_list, extractor):
        self.file_list = file_list
        self.extractor = extractor

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        
        # 1. Image originale
        img_orig = Image.open(path).convert("RGB")
        feat_orig = self.extractor(img_orig, return_tensors="pt")['pixel_values'].squeeze(0)
        
        # 2. Image bruitée (on réutilise ta fonction OpenCV)
        img_bruit_np = augment_for_uncertainty(str(path))
        img_bruit_pil = Image.fromarray(cv2.cvtColor(img_bruit_np, cv2.COLOR_BGR2RGB))
        feat_bruit = self.extractor(img_bruit_pil, return_tensors="pt")['pixel_values'].squeeze(0)
        
        return feat_orig, feat_bruit, str(path)
    
def augment_for_uncertainty(image_path):
    img = cv2.imread(image_path)
    
    # Choix aléatoire des dégradations
    alpha = random.uniform(0.5, 1.5) # Contraste aléatoire
    beta = random.randint(-50, 50)   # Luminosité aléatoire
    
    # Application
    img_aug = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    
    # Flou pour simuler un mouvement ou une mauvaise mise au point
    if random.random() > 0.5:
        img_aug = cv2.GaussianBlur(img_aug, (5, 5), 0)
        
    return img_aug