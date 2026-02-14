import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, ResNetForImageClassification
from torchvision import transforms
import random
import sys
from tqdm import tqdm
import gc

# ==========================================
# 0. CONFIGURATION
# ==========================================
sys.stdout.reconfigure(encoding='utf-8')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">>> GPU ACTIVÉ : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

MAX_X = 1920.0
MAX_Y = 1080.0
BATCH_SIZE = 32

# DOSSIERS
PARENT_FOLDER = "." 
PATHS = {
    "Random": os.path.join(PARENT_FOLDER, "best_models_random"),
    "Dropout": os.path.join(PARENT_FOLDER, "best_models_dropout_0.25"),
    "DataAug": os.path.join(PARENT_FOLDER, "best_models_data_aug"),
    "ClusterL2": os.path.join(PARENT_FOLDER, "best_models_clustering_L2"),
    "Mixte": os.path.join(PARENT_FOLDER, "best_models_mixed")
}

# ==========================================
# 1. SETUP DONNÉES
# ==========================================
dataset_dir = Path(r"C:\Users\Megaport\Desktop\PROJET_AL\FINAL_DATASET")
if not dataset_dir.exists():
    print("ERREUR : Dataset introuvable.")
    sys.exit()

dir_test_B = dataset_dir / "Domain_B_Test"
all_files_B = sorted(list(dir_test_B.glob("*.jpg")) + list(dir_test_B.glob("*.JPG")))

random.seed(42)
# On garde 100 images pour avoir des stats solides
sample_imgs = random.sample(all_files_B, min(100, len(all_files_B)))
print(f">>> Échantillon de test : {len(sample_imgs)} images")

augment_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True)
])

def get_true_coords(filename):
    try:
        name = filename.replace('.jpg', '').replace('.JPG', '')
        parts = name.split('_')
        return float(parts[-2]), float(parts[-1])
    except: return MAX_X/2, MAX_Y/2

def cleanup():
    torch.cuda.empty_cache()
    gc.collect()

# Fonction générique pour récupérer les erreurs d'un modèle
def get_errors(model_path, images):
    if not os.path.exists(model_path): return None
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = ResNetForImageClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()
    
    errors = []
    for img_path in tqdm(images, leave=False):
        img = Image.open(img_path).convert("RGB")
        true_x, true_y = get_true_coords(img_path.name)
        inputs = processor(img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            p = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
        dist = np.sqrt((p[0]*MAX_X - true_x)**2 + (p[1]*MAX_Y - true_y)**2)
        errors.append(dist)
    
    del model
    cleanup()
    return errors

# Fonction pour calculer la variance (robustesse)
def get_robustness_score(model_path, images):
    if not os.path.exists(model_path): return None
    model = ResNetForImageClassification.from_pretrained(model_path).to(DEVICE)
    model.eval()
    
    variances = []
    # On teste sur 20 images seulement pour aller vite (c'est lourd x10)
    for img_path in tqdm(images[:20], leave=False):
        img = Image.open(img_path).convert("RGB")
        batch = torch.stack([augment_transform(img) for _ in range(10)]).to(DEVICE)
        with torch.no_grad():
            preds = torch.sigmoid(model(batch).logits).cpu().numpy()
        
        # Ecart-type moyen (en pixels)
        std = np.mean([np.std(preds[:,0]*MAX_X), np.std(preds[:,1]*MAX_Y)])
        variances.append(std)
        
    del model
    cleanup()
    return np.mean(variances)

# ==========================================
# 2. GÉNÉRATION DES GRAPHIQUES COMPLETS
# ==========================================

def plot_all_strategies():
    strategies = ["Random", "Dropout", "DataAug", "ClusterL2", "Mixte"]
    
    # Stockage des données
    data_baseline = [] # Pour Boxplot
    data_final = []    # Pour Boxplot
    
    means_baseline = [] # Pour BarChart
    means_final = []    # Pour BarChart
    
    robustness_baseline = [] # Pour Robustesse
    robustness_final = []    # Pour Robustesse
    
    valid_strategies = []

    print("\n=== CALCUL DES STATISTIQUES POUR TOUTES LES STRATÉGIES ===")
    
    for strat in strategies:
        print(f">>> Traitement : {strat}")
        path_base = os.path.join(PATHS[strat], "best_baseline")
        path_final = os.path.join(PATHS[strat], "best_model_ratio_50")
        
        # 1. Erreurs (Précision)
        err_b = get_errors(path_base, sample_imgs)
        err_f = get_errors(path_final, sample_imgs)
        
        if err_b and err_f:
            data_baseline.append(err_b)
            data_final.append(err_f)
            means_baseline.append(np.mean(err_b))
            means_final.append(np.mean(err_f))
            valid_strategies.append(strat)
            
            # 2. Robustesse (Stabilité)
            print(f"    Calcul Robustesse {strat}...")
            rob_b = get_robustness_score(path_base, sample_imgs)
            rob_f = get_robustness_score(path_final, sample_imgs)
            robustness_baseline.append(rob_b)
            robustness_final.append(rob_f)
        else:
            print(f"    [SKIP] Dossiers manquants pour {strat}")

    if not valid_strategies: return print("Aucune donnée valide trouvée.")

    # --- GRAPHIQUE 1 : MOYENNES (Barres) ---
    x = np.arange(len(valid_strategies))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    rects1 = plt.bar(x - width/2, means_baseline, width, label='Avant (Domain A)', color='salmon', edgecolor='black')
    rects2 = plt.bar(x + width/2, means_final, width, label='Après (A + 50% B)', color='lightgreen', edgecolor='black')
    
    # MODIF: Titre avec sous-titre explicatif
    plt.title("1. Précision Moyenne : Comparatif Complet\n(Plus c'est bas, meilleur est le modèle)")
    plt.ylabel("Erreur Moyenne (Pixels)")
    plt.xticks(x, valid_strategies)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # MODIF: Ajout des valeurs sur les barres
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height,
                     f'{height:.0f}',
                     ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig("FULL_1_MOYENNES.png")
    print("[OK] FULL_1_MOYENNES.png")

    # --- GRAPHIQUE 2 : DISTRIBUTIONS (Boxplots) ---
    plt.figure(figsize=(14, 8))
    
    # On doit alterner les positions : 1, 2,  4, 5,  7, 8...
    positions_b = [i*3 + 1 for i in range(len(valid_strategies))]
    positions_f = [i*3 + 2 for i in range(len(valid_strategies))]
    
    bplot1 = plt.boxplot(data_baseline, positions=positions_b, widths=0.6, patch_artist=True, boxprops=dict(facecolor="salmon"))
    bplot2 = plt.boxplot(data_final, positions=positions_f, widths=0.6, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    
    # Légende manuelle
    plt.plot([], [], color='salmon', linewidth=10, label='Avant (Domain A)')
    plt.plot([], [], color='lightgreen', linewidth=10, label='Après (A + 50% B)')
    
    plt.title("2. Stabilité et Dispersion : Comparatif Complet\n(Boîte petite = Modèle constant | Boîte grande = Modèle instable)")
    plt.ylabel("Erreur (Pixels)")
    
    # Étiquettes au milieu des paires
    plt.xticks([i*3 + 1.5 for i in range(len(valid_strategies))], valid_strategies)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # MODIF: Ajout des médianes sur les boîtes
    def add_median_labels(bplot):
        for median in bplot['medians']:
            # Récupérer les coordonnées de la ligne médiane
            x, y = median.get_xdata()[0], median.get_ydata()[0]
            # On centre le texte sur la boîte (moyenne des x)
            x_center = np.mean(median.get_xdata())
            # Affichage avec un petit fond blanc pour la lisibilité
            plt.text(x_center, y, f'{y:.0f}', ha='center', va='bottom', 
                     fontweight='bold', fontsize=9, color='black',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    add_median_labels(bplot1)
    add_median_labels(bplot2)

    plt.tight_layout()
    plt.savefig("FULL_2_DISTRIBUTIONS.png")
    print("[OK] FULL_2_DISTRIBUTIONS.png")

    # --- GRAPHIQUE 3 : ROBUSTESSE (Score) ---
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, robustness_baseline, width, label='Avant', color='orange', alpha=0.6)
    plt.bar(x + width/2, robustness_final, width, label='Après', color='teal', alpha=0.8)
    
    plt.title("3. Score de Tremblement (Robustesse)\n(Plus c'est bas, plus le modèle est solide face aux perturbations)")
    plt.ylabel("Variance Moyenne (Pixels)")
    plt.xticks(x, valid_strategies)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Ajout du gain en %
    for i in range(len(valid_strategies)):
        gain = robustness_baseline[i] - robustness_final[i]
        color = 'green' if gain > 0 else 'red'
        plt.text(i + width/2, robustness_final[i] + 0.5, f"{'-' if gain>0 else '+'}{abs(gain):.1f}", 
                 ha='center', color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig("FULL_3_ROBUSTESSE.png")
    print("[OK] FULL_3_ROBUSTESSE.png")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    plot_all_strategies()
    print("\n--- ANALYSE COMPLÈTE TERMINÉE ---")