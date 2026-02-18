import os
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, ResNetForImageClassification
import random
import time
import data_loading as dt_loading

import prepareData

mixed = "mixed_strategy"
diversity = "diversity"
uncertainty = "uncertainty"
mixed2 = "mixed_strategy_bis"
random_strat = "random"
idk = "idk"


def train(strategy, percentage, previous_percentage = 0, seed = 0):
    print("-------- ENTRAINEMENT ---------")
    if (seed != 0):
        save_strategy = strategy + str(seed)
    else: save_strategy = strategy
    save_model_name = f"models_{save_strategy}/model_{strategy}_{percentage}percent"

    current_dir = Path(__file__).resolve().parent
    
    if previous_percentage > 0:
        model_name = f"model_{strategy}_{previous_percentage}percent"
        old_model_path = current_dir / f"models_{save_strategy}" / model_name
    else:
        old_model_path = current_dir / "model_baseline"
    if (seed != 0):
        dataset_folder = f"{strategy}_datasets_{seed}"
    else: dataset_folder = f"{strategy}_datasets"
    
    dataset_name = f"{strategy}_dataset_{percentage}percent"
    
    path_b = current_dir.parent / "ActiveLearningDatasets" / dataset_folder / dataset_name

    # ==========================================
    # 1. CONFIGURATION ET CHEMINS
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)

    # --- Domaine A (75% Train / 25% Test) ---
    path_a_train = current_dir.parent / "datasets" / "Domain_A" / "train"
    path_a_test = current_dir.parent / "datasets" / "Domain_A" / "test"
    train_a = list(path_a_train.glob("*.jpg"))
    test_a = list(path_a_test.glob("*.jpg"))

    # --- Domaine B (80% Train / 20% Test) ---
    # On va chercher dans le dossier déjà créé

    if not path_b.exists():
        raise FileNotFoundError(f"Le dossier {path_b} n'existe pas dans {current_dir}")

    train_b = list(path_b.glob("*.jpg"))
    path_b_test = current_dir.parent / "Domaine_B" / "test"
    test_b = list(path_b_test.glob("*jpg"))

    print(len(train_a))
    print(len(train_b))

    # --- Fusion ---
    final_train_list = train_a  + train_b
    final_test_list = test_a + test_b


    # ==========================================
    # 3. MODÈLE ET CHARGEMENT
    # ==========================================
    # old_model_path = "microsoft/resnet-18"
    # processor = AutoFeatureExtractor.from_pretrained(old_model_path)

    if os.path.exists(old_model_path):
        print(f"Chargement du modèle existant depuis {old_model_path}...")
        # On charge vos poids déjà entraînés sur le Domaine A
        model = ResNetForImageClassification.from_pretrained(
            old_model_path, 
            num_labels=2, 
            ignore_mismatched_sizes=True
        ).to(device)
        processor = AutoFeatureExtractor.from_pretrained(old_model_path)
    else:
        print("Modèle existant non trouvé, départ de zéro avec resnet-18.")
        model_name = "microsoft/resnet-18"
        processor = AutoFeatureExtractor.from_pretrained(model_name)
        model = ResNetForImageClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True
        ).to(device)

    dataset = dt_loading.ImageRegressionDataset(final_train_list, processor)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

    dataset = dt_loading.ImageRegressionDataset(final_test_list, processor)
    test_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    # ==========================================
    # 4. BOUCLE D'ENTRAÎNEMENT SIMPLIFIÉE
    # ==========================================


    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss() # Perte standard pour la régression

    model.train()
    epochs = 20

    print(len(final_train_list))

    print("Début de l'entraînement...")
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (pixel_values, targets) in enumerate(train_loader):
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
            
        print(f"Epoch [{epoch+1}/{epochs}] - Loss Moyenne: {epoch_loss/len(train_loader):.6f}")

    end_time = time.time()
    print("Entraînement terminé.")

    # --- SAUVEGARDE ---
    output_dir = current_dir / save_model_name

    # Créer le dossier s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarde du modèle (poids + configuration)
    model.save_pretrained(output_dir)

    # Sauvegarde du processeur (indispensable pour retrouver les mêmes transformations d'images)
    processor.save_pretrained(output_dir)

    print(f"Modèle sauvegardé avec succès dans : {output_dir}")

    elapsed_time = end_time - start_time
    print(f"Temps écoulé : {elapsed_time} secondes")


list_strategies = [
    # mixed, 
    # diversity, 
    # uncertainty, 
    # mixed2, 
    # random_strat,
    idk]

couples_pourcentages = [(1,0), (5,1), (10,5),(20,10),(50,20)]
seeds = [42,43,44,45,46]

def automatisation():

    for strategy in list_strategies:
        for pourcentage in couples_pourcentages:
            if (strategy == random_strat or strategy == diversity):
                for seed in seeds:

                    print("=" * 50)
                    print(f"Stratégie : {strategy} - Seed : {seed} - Pourcentage : {pourcentage}")
                    print("=" * 50)

                    print()
                    prepareData.prep_data(strategy, pourcentage, seed)
                    train(strategy, pourcentage[0], pourcentage[1], seed)
            else: 

                print("=" * 50)
                print(f"Stratégie : {strategy} - Pourcentage : {pourcentage}")
                print("=" * 50)

                prepareData.prep_data(strategy, pourcentage)
                train(strategy, pourcentage[0], pourcentage[1])

automatisation()