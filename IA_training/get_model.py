import torch
from PIL import Image
import torch.nn as nn
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import data_loading as dt_loading 
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

def evaluate_model(model, data_loader, device, desc="Evaluation"):
    model.eval()
    total_mse = 0
    total_pixel_error = 0
    # On récupère les dimensions pour dé-normaliser
    max_x, max_y = 1920.0, 1080.0 
    
    with torch.no_grad():
        for pixel_values, targets in data_loader:
            pixel_values, targets = pixel_values.to(device), targets.to(device)
            
            outputs = model(pixel_values).logits
            predictions = torch.sigmoid(outputs)
            
            # MSE sur valeurs normalisées (0-1)
            loss = nn.functional.mse_loss(predictions, targets)
            total_mse += loss.item()

            # Erreur en pixels (Distance Euclidienne)
            pred_px = predictions * torch.tensor([max_x, max_y]).to(device)
            true_px = targets * torch.tensor([max_x, max_y]).to(device)
            dist = torch.sqrt(torch.sum((pred_px - true_px)**2, dim=1))
            total_pixel_error += dist.mean().item()

    avg_mse = total_mse / len(data_loader)
    avg_px_dist = total_pixel_error / len(data_loader)
    
    print(f"--- {desc} ---")
    print(f"MSE (norm): {avg_mse:.6f}")
    print(f"Erreur Moyenne: {avg_px_dist:.2f} pixels\n")
    return avg_mse, avg_px_dist


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

current_dir = Path(__file__).resolve().parent

# --- Domaine A ---
path_a = current_dir.parent / "datasets" / "Domain_A" / "test"
files_a = list(path_a.glob("*.jpg"))

# --- Domaine B ---
path_b = current_dir.parent / "datasets" / "Domain_B" / "test"
files_b = list(path_b.glob("*.jpg"))


def test(files, strategy, percent, seed = 0):
    if(percent == 0):
        model_dir = current_dir / "model_baseline"
    else:
        if(seed != 0):
            model_dir = current_dir / f"models_{strategy}{seed}" / f"model_{strategy}_{percent}percent"
        else: model_dir = current_dir / f"models_{strategy}" / f"model_{strategy}_{percent}percent"
    print(f"Modèle et extracteur chargés depuis {model_dir}")
    model = ResNetForImageClassification.from_pretrained(model_dir).to(device)
    extractor = AutoFeatureExtractor.from_pretrained(model_dir)

    # --- Préparation du DataLoader ---
    test_dataset = dt_loading.ImageRegressionDataset(files, extractor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # --- Lancement de l'Évaluation ---
    if len(files) > 0:
        print("\n--- RÉSULTATS DES TESTS ---")
        print(f"Model = {strategy} - Percent = {percent} - Seed = {seed}")
        return evaluate_model(model, test_loader, device, desc="Domaine B (Source)")
    else:
        print("Aucune image trouvée pour le test.")

mixed = "mixed_strategy"
diversity = "diversity"
uncertainty = "uncertainty"
mixed2 = "mixed_strategy_bis"
random_strat = "random"
idk = "idk"
list_strategies = [
    #mixed, 
    diversity, 
    uncertainty, 
    #mixed2, 
    random_strat,
    #idk
    ]

pourcentages = [1,5,10,20,50]
seeds = [42,43,44,45,46]


import csv
import os

def enregistrer_donnee(nom_ligne, nom_colonne, valeur):
    fichier = 'resultats.csv'
    file_exists = os.path.isfile(fichier)
    
    with open(fichier, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Ligne', 'Colonne', 'Valeur'])
        writer.writerow([nom_ligne, nom_colonne, valeur])

def pipeline_test():

    avg_mse,_ = test(files_b, "baseline",0)
    
    for strategy in list_strategies:
        if (strategy == random_strat or strategy == diversity): 
                for seed in seeds:
                    enregistrer_donnee(f"{strategy} seed {seed}","0%",avg_mse)
        else: enregistrer_donnee(strategy,"0%",avg_mse)

    for strategy in list_strategies:
        for pourcentage in pourcentages:
            if (strategy == random_strat or strategy == diversity): # or strategy == diversity
                for seed in seeds:
                    avg_mse, _ = test(files_b, strategy,pourcentage,seed)
                    enregistrer_donnee(f"{strategy} seed {seed}", f"{pourcentage}%", avg_mse)
            else: 
                avg_mse,_ = test(files_b,strategy,pourcentage)
                enregistrer_donnee(strategy,f"{pourcentage}%",avg_mse)



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from scipy import stats

def mean_confidence_interval(data, confidence=0.95):
    # Transformation en array numpy pour la sécurité
    a = np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a) # Moyenne et Erreur Standard
    # Calcul de l'intervalle avec la loi de Student
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def plot():

    renommer = {
        mixed: 'mixed strategy 1',
        mixed2: 'mixed strategy 2',
        idk: 'Discriminative Active Learning'
    }

    couleurs = {
        random_strat: 'green'
    }


    # Charger les données enregistrées
    df = pd.read_csv('resultats.csv')

    df['Strategie_Pure'] = df['Ligne'].str.split(' seed').str[0]
    df['Pct_Num'] = df['Colonne'].str.replace('%', '').astype(float)

    df = df[df['Strategie_Pure'].isin(list_strategies)]
    stats_df = df.groupby(['Strategie_Pure', 'Pct_Num'])['Valeur'].apply(mean_confidence_interval).reset_index()
    stats_df[['Moyenne', 'IC']] = pd.DataFrame(stats_df['Valeur'].tolist(), index=stats_df.index)


    plt.figure(figsize=(10, 6))


    for strat in stats_df['Strategie_Pure'].unique():
        subset = stats_df[stats_df['Strategie_Pure'] == strat]

        
        couleur_imposee = couleurs.get(strat)

        nom_affiche = renommer.get(strat, strat)

        line, = plt.plot(
            subset['Pct_Num'], 
            subset['Moyenne'], 
            '-o', 
            label=nom_affiche,
            color = couleur_imposee,
            linewidth=2
        )

        color = line.get_color()

        plt.fill_between(
            subset['Pct_Num'], 
            subset['Moyenne'] - subset['IC'], 
            subset['Moyenne'] + subset['IC'], 
            color=couleur_imposee, 
            alpha=0.1  # Transparence pour voir les croisements
        )

    plt.xlabel("Ratio du dataset Active Learning utilisé (%)", fontsize=12)
    plt.ylabel("MSELoss moyenne (avec IC à 95%)", fontsize=12)
    plt.title("Évolution de la performance par stratégie", fontsize=14, pad=20)

    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("resultats_scipy_unc_div.png", dpi=300)
    plt.show()

    plt.savefig("result.png", dpi=300, bbox_inches='tight')


# pipeline_test()
plot()