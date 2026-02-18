import shutil
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from pathlib import Path
from operator import itemgetter
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist
import data_loading as loading


mixed = "mixed_strategy"
diversity = "diversity"
uncertainty = "uncertainty"
mixed2 = "mixed_strategy_bis"
random_strat = "random"
idk = "idk"



"""
Méthodes utilitaires
"""

current_dir = Path(__file__).resolve().parent
source_root = current_dir.parent / "datasets" / "Domain_B" / "active_learning"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_files = list(source_root.glob("*.jpg"))

def prepare_strategy(strategy, percentage, previous_percentage = 0, seed = 0):
    if seed == 0:
        target_dir = current_dir.parent / "ActiveLearningDatasets" / f"{strategy}_datasets" /  f"{strategy}_dataset_{int(percentage)}percent"
        previous_target_dir = current_dir.parent / "ActiveLearningDatasets" / f"{strategy}_datasets" / f"{strategy}_dataset_{int(previous_percentage)}percent"

    else:
        target_dir = current_dir.parent / "ActiveLearningDatasets" / f"{strategy}_datasets_{seed}" /  f"{strategy}_dataset_{int(percentage)}percent"
        previous_target_dir = current_dir.parent / "ActiveLearningDatasets" / f"{strategy}_datasets_{seed}" / f"{strategy}_dataset_{int(previous_percentage)}percent"

    return target_dir, previous_target_dir

def write_new_data(target_dir, selected_data):
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_path in selected_data:
        if file_path.is_file():
            shutil.copy2(file_path, target_dir / file_path.name)

def copy_previous(target_dir, percentage, previous_target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)

    # Au début de la fonction augmentation_data, après la création de target_dir
    if percentage > 1:
        # Trouver le dossier précédent (ex: si 10%, on cherche le 5%)
        if previous_target_dir.exists():
            print(f"Copie de la base précédente : {previous_target_dir.name}")
            for f in previous_target_dir.glob("*.jpg"):
                shutil.copy2(f, target_dir / f.name)

def get_remaining_files(target_dir, percentage):
    already_selected_names = {f.name for f in target_dir.glob("*.jpg")}
    remaining_files = [f for f in all_files if f.name not in already_selected_names]
    
    # Calcul du nombre d'images à ajouter
    total_needed = int(len(all_files) * percentage / 100)
    n_to_add = total_needed - len(already_selected_names)
    return remaining_files, n_to_add

def get_model_extractor(previous_percentage, strategy, seed = 0):
    if(previous_percentage == 0):
        model_dir = current_dir / "model_baseline"
    elif (seed == 0): model_dir = current_dir / f"models_{strategy}" / f"model_{strategy}_{previous_percentage}percent"
    else: model_dir = current_dir / f"models_{strategy}{seed}" / f"model_{strategy}_{previous_percentage}percent"
    
    model = ResNetForImageClassification.from_pretrained(str(model_dir.resolve().as_posix())).to(device)
    extractor = AutoFeatureExtractor.from_pretrained(str(model_dir.resolve().as_posix()))

    return model, extractor

"""
Sélection random
"""

def get_random_data(percentage, data, seed = 42):
    data.sort()
    
    # On utilise une graine fixe pour que le 'shuffle' soit identique à chaque exécution
    random.seed(seed)
    random.shuffle(data)
    
    sample_size = round(len(data) * percentage / 100)
    return data[:sample_size]

def get_random_data_nb(number, data, seed = 42):
    random.seed(seed)
    random.shuffle(data)
    return data[:number]

def random_data(percentage, seed = 42):
    target_dir, _ = prepare_strategy(random_strat,percentage, 0,seed)

    data = []
    data = list(source_root.glob("*.jpg"))

    selected_data = get_random_data(percentage, data)

    write_new_data(target_dir,selected_data)

""" 
Sélection "uncertainty sampling" avec la stratégie "augmentation" : on donne une image normale puis la même image altérée, et on note si le modèle change d'avis. On sélectionne les images ayant les plus grandes disparités.
"""

def adjust_quality(image, contrast=1.2, brightness=30):
    # contrast (1.0-3.0), brightness (0-100)
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adjusted

def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss * 255, 0, 255).astype(np.uint8)

def get_distance(coord_1, coord_2):
    return np.sqrt((coord_2[0] - coord_1[0])**2 + (coord_2[1] - coord_1[1])**2)

def get_augmentation_data(number, file_list, model, extractor):

    dataset = loading.UncertaintyDataset(file_list, extractor)
    # On garde num_workers=2 pour que le CPU prépare les images bruitées en avance
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    
    liste_scores = []

    with torch.no_grad():
        for batch_orig, batch_bruit, paths in loader:
            # Inférence par batch
            logits_orig = model(batch_orig).logits
            logits_bruit = model(batch_bruit).logits
            
            # Calcul de distance Euclidienne vectorisé sur le batch
            # sqrt(sum((p1 - p2)^2))
            diff = logits_orig - logits_bruit
            distances = torch.sqrt(torch.sum(diff**2, dim=1)).numpy()
            
            # On stocke les résultats
            for i in range(len(paths)):
                liste_scores.append((Path(paths[i]), distances[i]))


    # Tri par score décroissant (les plus instables en premier)
    liste_scores.sort(key=itemgetter(1), reverse=True)
    
    # Sélection selon le pourcentage
    return [item[0] for item in liste_scores[:number]]

def augmentation_data(percentage, previous_percentage = 0):
    random.seed(42)
    target_dir, previous_target_dir = prepare_strategy(uncertainty, percentage,previous_percentage)

    copy_previous(target_dir,percentage,previous_target_dir)

    model, extractor = get_model_extractor(previous_percentage, diversity)

    remaining_files, n_to_add = get_remaining_files(target_dir, percentage)
    if n_to_add > 0:
        # On modifie get_augmentation_data pour qu'il prenne un NOMBRE et non un %
        selected_data = get_augmentation_data(n_to_add, remaining_files, model, extractor)
        write_new_data(target_dir,selected_data)

"""
Sélection "diversity sampling" avec la stratégie de clustering : on crée des clusters à l'aide de la similarité entre les images, puis on choisit les centroïdes.
"""

def get_embeddings(file_list, model, extractor, batch_size=16):
    model.eval()
    # On utilise ton dataset en mode "sans labels" pour la sélection
    dataset = loading.ActiveLearningDataset(file_list, extractor, return_labels=False)
    # num_workers=2 permet de charger les images en parallèle pendant que le CPU calcule
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
    
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            # batch contient ici directement les pixel_values
            outputs = model.resnet(batch)
            pooled_output = torch.flatten(outputs.pooler_output, 1)
            embeddings.append(pooled_output.numpy())
            
    return np.concatenate(embeddings, axis=0)

"""
def get_embeddings(file_list, model, extractor):
    model.eval()
    embeddings = []
    
    # On retire la dernière couche (head) pour avoir les features pures
    # Pour ResNet chez Hugging Face, on utilise souvent model.resnet
    with torch.no_grad():
        for path in file_list:
            img = Image.open(path).convert("RGB")
            inputs = extractor(img, return_tensors="pt").to(device)
            
            # On récupère la sortie avant la couche de classification/régression
            outputs = model.resnet(inputs['pixel_values'])
            # Global Average Pooling pour avoir un vecteur plat
            pooled_output = torch.flatten(outputs.pooler_output, 1)
            embeddings.append(pooled_output.cpu().numpy()[0])
            
    return np.array(embeddings)
"""
    
def get_diversity_data(n_to_select, file_list, model, extractor, seed = 42):
    
    print(f"Extraction des caractéristiques pour {len(file_list)} images...")
    X = get_embeddings(file_list, model, extractor)
    
    print(f"Création de {n_to_select} clusters...")
    kmeans = KMeans(n_clusters=n_to_select, random_state=seed, n_init=10)
    kmeans.fit(X)
    
    # Trouver l'image la plus proche du centre de chaque cluster (le centroïde)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    
    selected_paths = [file_list[idx] for idx in closest]
    return selected_paths

def diversity_data_selection(percentage, previous_percentage = 0, seed = 42):

    target_dir, previous_target_dir = prepare_strategy(diversity, percentage,previous_percentage, seed)

    copy_previous(target_dir,percentage,previous_target_dir)
    
    remaining_files, n_to_add = get_remaining_files(target_dir,percentage)

    model, extractor = get_model_extractor(previous_percentage, diversity, seed)

    if n_to_add > 0:
        new_selected = get_diversity_data(n_to_add, remaining_files, model, extractor, seed)
        write_new_data(target_dir,new_selected)

"""
PREMIERE STRATEGIE MIXTE : dans les données à ajouter, 10% de random, 50% de diversity sélectionné parmi 180% de uncertainty
"""

def mixed_strategy_selection(percentage, previous_percentage = 0):
    target_dir, previous_target_dir = prepare_strategy(mixed,percentage, previous_percentage)
    copy_previous(target_dir,percentage,previous_target_dir)

    remaining_files, n_to_add = get_remaining_files(target_dir,percentage)
    
    nb_random = 0
    nb_diversity = 0
    nb_uncertainty = 0
    nb_final_hybrid = 0

    model, extractor = get_model_extractor(previous_percentage, mixed)

    if n_to_add > 0:

        # 10% de l'ajout est aléatoire
        nb_random = round(n_to_add * 0.10)
        # Les 90% restants sont pour la stratégie hybride
        nb_final_hybrid = n_to_add - nb_random
        # On prend le double pour le pool d'incertitude (ratio 2:1)
        nb_uncertainty = nb_final_hybrid * 2
        # La diversité ramènera ce nombre à nb_final_hybrid
        nb_diversity = nb_final_hybrid

        random_selected = get_random_data_nb(nb_random, remaining_files)
        random_names = {f.name for f in random_selected}
        remaining_files_random =  [f for f in remaining_files if f.name not in random_names]
        uncertainty_selected = get_augmentation_data(nb_uncertainty,remaining_files_random, model, extractor)
        diversity_selected = get_diversity_data(nb_diversity,uncertainty_selected, model, extractor, 42)
        new_selected = random_selected + diversity_selected
        write_new_data(target_dir,new_selected)

"""
SECONDE STRATEGY MIXTE : score normalisé uncertainty * diversity
"""

def get_mixed_strategy_bis_data(n_to_select, file_list, model, extractor):
    # --- ÉTAPE 1 : CALCUL DE LA DIVERSITÉ ---
    print("Calcul des scores de diversité...")
    X = get_embeddings(file_list, model, extractor)
    
    # On crée les clusters
    kmeans = KMeans(n_clusters=n_to_select, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    # Score de diversité = inverse de la distance au centroïde le plus proche
    # Plus on est proche d'un centre de cluster, plus le score est haut
    distances_to_centers = np.min(cdist(X, centers, metric='euclidean'), axis=1)
    # On normalise pour éviter les échelles disproportionnées (ex: entre 0 et 1)
    score_diversity = 1 / (1 + distances_to_centers)

    # --- ÉTAPE 2 : CALCUL DE L'INCERTITUDE ---
    print("Calcul des scores d'incertitude...")
    model.eval()
    score_uncertainty = []

    with torch.no_grad():
        for path in file_list:
            # Prédiction originale
            img_orig = Image.open(path).convert("RGB")
            inputs_orig = extractor(img_orig, return_tensors="pt").to(device)
            pred_orig = model(inputs_orig['pixel_values']).logits.cpu().numpy()[0]
            
            # Prédiction bruitée
            img_bruit_np = loading.augment_for_uncertainty(str(path))
            img_bruit_pil = Image.fromarray(cv2.cvtColor(img_bruit_np, cv2.COLOR_BGR2RGB))
            inputs_bruit = extractor(img_bruit_pil, return_tensors="pt").to(device)
            pred_bruit = model(inputs_bruit['pixel_values']).logits.cpu().numpy()[0]
            
            # Distance (votre fonction get_distance)
            score_uncertainty.append(get_distance(pred_orig, pred_bruit))

    score_uncertainty = np.array(score_uncertainty)

    # --- ÉTAPE 3 : COMBINAISON ET SÉLECTION ---
    # Normalisation min-max pour que les deux scores pèsent autant
    s_div_norm = (score_diversity - score_diversity.min()) / (score_diversity.max() - score_diversity.min())
    s_unc_norm = (score_uncertainty - score_uncertainty.min()) / (score_uncertainty.max() - score_uncertainty.min())
    
    final_scores = calculate_score_one(s_div_norm, s_unc_norm)
    
    # Trier les indices par score décroissant
    best_indices = np.argsort(final_scores)[::-1][:n_to_select]
    
    return [file_list[i] for i in best_indices]

# Méthode utilisée
def calculate_score_one(s_div_norm,s_unc_norm):
    return s_div_norm * s_unc_norm

# Ajout inutilisé
def calculate_score_two(s_div_norm,s_unc_norm, weight = 0.5):
    return weight * s_div_norm + (1 - weight) * s_unc_norm

def mixed_strategy_bis_selection(percentage, previous_percentage = 0):
    
    target_dir, previous_target_dir = prepare_strategy(mixed2,percentage, previous_percentage)

    copy_previous(target_dir,percentage,previous_target_dir)
    
    remaining_files, n_to_add = get_remaining_files(target_dir,percentage)

    model, extractor = get_model_extractor(previous_percentage, mixed2)

    if n_to_add > 0:
        new_selected = get_mixed_strategy_bis_data(n_to_add, remaining_files,model,extractor)
        write_new_data(target_dir,new_selected)

"""
SELECTION IDK
"""

class FamiliarityModel(nn.Module):
    def __init__(self, input_dim=512): # 512 pour ResNet-18
        super(FamiliarityModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # Sortie entre 0 (inconnu) et 1 (connu/familier)
        )
    
    def forward(self, x):
        return self.net(x)
    
def train_familiarity_model(X_known, X_unknown):
    print("---- IDK training----")
    fam_model = FamiliarityModel(input_dim=X_known.shape[1]).to(device)
    optimizer = torch.optim.Adam(fam_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    n_known = len(X_known)
    n_unknown = len(X_unknown)

    if n_unknown > n_known:
        indices = np.random.choice(n_unknown, n_known, replace = False)
        X_unknown_balanced = X_unknown[indices]
        X_known_balanced = X_known
    elif n_known > n_unknown:
        indices = np.random.choice(n_known,n_unknown, replace = False)
        X_known_balanced = X_known[indices]
        X_unknown_balanced = X_unknown
    else: 
        X_unknown_balanced = X_unknown
        X_known_balanced = X_known
    
    # Création du dataset binaire
    X = np.concatenate([X_known_balanced, X_unknown_balanced])
    y = np.concatenate([np.ones(len(X_known_balanced)), np.zeros(len(X_unknown_balanced))])
    
    # Boucle d'entraînement rapide (le modèle est très léger)
    # ... code d'entraînement standard ...

    # Étape 2 : Conversion en tenseurs PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Étape 3 : Boucle d'entraînement
    fam_model.train()
    epochs = 50  # Un petit nombre suffit pour un classifieur binaire
    batch_size = 16
    
    print(f"Entraînement du modèle de familiarité sur {len(X)} échantillons...")
    
    for epoch in range(epochs):
        # Mélange des données à chaque époque
        permutation = torch.randperm(X_tensor.size()[0])
        epoch_loss = 0
        
        for i in range(0, X_tensor.size()[0], batch_size):
            optimizer.zero_grad()
            
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_tensor[indices], y_tensor[indices]
            
            # Forward
            outputs = fam_model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Époque [{epoch+1}/{epochs}], Loss: {epoch_loss/len(X_tensor):.4f}")
    return fam_model

def idk_selection(percentage, previous_percentage = 0):
    print("---- IDK selection----")
    # 1. Préparation des dossiers (similaire à votre code actuel)
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    target_dir = root_dir / "ActiveLearningDatasets" / "idk_datasets" / f"idk_dataset_{percentage}percent"
    domain_a_dir = root_dir / "datasets" / "Domain_A" / "train"
    path_sources_b = root_dir / "datasets" / "Domain_B" / "active_learning"

    target_dir.mkdir(parents=True, exist_ok=True)

    # 2. Gestion de l'inclusion (récupérer le dossier précédent)
    if percentage > 1:
        previous_target_dir = current_dir.parent / "ActiveLearningDatasets" / "idk_datasets" /  f"idk_dataset_{previous_percentage}percent"
        if previous_target_dir.exists():
            print(f"Copie de la base précédente : {previous_target_dir.name}")
            for f in previous_target_dir.glob("*.jpg"):
                shutil.copy2(f, target_dir / f.name)
    else: previous_target_dir = current_dir.parent / "datasets" / "Domain_A" / "train"
    
    # 3. Filtrage des fichiers restants
    all_files_b = list(path_sources_b.glob("*.jpg"))
    already_selected_names = {f.name for f in target_dir.glob("*.jpg")}
    remaining_files = [f for f in all_files_b if f.name not in already_selected_names]
    
    known_paths = list(domain_a_dir.glob("*.jpg")) + list(target_dir.glob("*.jpg"))

    # 4. Calcul du nombre d'images à ajouter
    total_needed = round(len(all_files_b) * percentage / 100)
    n_to_add = total_needed - len(already_selected_names)

    if(previous_percentage == 0):
        model_dir = current_dir / "model_baseline"
    else: model_dir = current_dir / "models_idk" / f"model_idk_{previous_percentage}percent"
    
    
    if n_to_add > 0:
        model = ResNetForImageClassification.from_pretrained(str(model_dir.resolve().as_posix())).to(device)
        extractor = AutoFeatureExtractor.from_pretrained(str(model_dir.resolve().as_posix()))

        X_known = get_embeddings(known_paths, model, extractor)
        X_remaining = get_embeddings(remaining_files, model, extractor)
        fam_judge = train_familiarity_model(X_known, X_remaining)

        if fam_judge:
            fam_judge.eval()
            with torch.no_grad():
                X_rem_tensor = torch.tensor(X_remaining, dtype=torch.float32).to(device)
                scores = fam_judge(X_rem_tensor).cpu().numpy().flatten()
                
                # Pair paths with scores and sort (Ascending: 0 = most unknown)
                path_score_pairs = list(zip(remaining_files, scores))
                path_score_pairs.sort(key=lambda x: x[1])
                new_selected = [p for p, s in path_score_pairs[:n_to_add]]

            # 5. Copie finale
            target_dir.mkdir(parents=True, exist_ok=True)
            for f in new_selected:
                shutil.copy2(f, target_dir / f.name)
    print("---- IDK selected----")

"""
Mixed strategy 3 : sélection aléatoire parmi les clusters, avec une préférence pour les images incertaines
"""

def get_mixed_strategy_cluster_uncertainty(n_to_select, file_list, model, extractor):
    # --- ÉTAPE 1 : CLUSTERING (DIVERSITÉ) ---
    print("Calcul des embeddings et création des clusters...")
    X = get_embeddings(file_list, model, extractor)
    
    # On crée autant de clusters que de données à sélectionner
    # Ainsi, on ira chercher un échantillon représentatif dans chaque zone de l'espace
    kmeans = KMeans(n_clusters=n_to_select, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    # --- ÉTAPE 2 : CALCUL DE L'INCERTITUDE ---
    print("Calcul des scores d'incertitude...")
    model.eval()
    uncertainty_scores = []

    with torch.no_grad():
        for path in file_list:
            # Prédiction originale
            img_orig = Image.open(path).convert("RGB")
            inputs_orig = extractor(img_orig, return_tensors="pt").to(device)
            pred_orig = model(inputs_orig['pixel_values']).logits.cpu().numpy()[0]
            
            # Prédiction avec augmentation (bruit)
            img_bruit_np = augment_for_uncertainty(str(path))
            img_bruit_pil = Image.fromarray(cv2.cvtColor(img_bruit_np, cv2.COLOR_BGR2RGB))
            inputs_bruit = extractor(img_bruit_pil, return_tensors="pt").to(device)
            pred_bruit = model(inputs_bruit['pixel_values']).logits.cpu().numpy()[0]
            
            # Calcul de la distance (votre fonction habituelle)
            uncertainty_scores.append(get_distance(pred_orig, pred_bruit))
    
    uncertainty_scores = np.array(uncertainty_scores)

    # --- ÉTAPE 3 : SÉLECTION (UN PAR CLUSTER AVEC BOOST) ---
    selected_indices = []
    
    for i in range(n_to_select):
        # On récupère les indices de toutes les images appartenant au cluster i
        indices_in_cluster = np.where(cluster_labels == i)[0]
        
        if len(indices_in_cluster) > 0:
            # Au lieu de prendre une donnée purement aléatoire, 
            # on prend celle qui a le score d'incertitude le plus élevé dans ce cluster.
            # C'est ici que se fait le "coup de pouce" (boost).
            idx_best_in_cluster = indices_in_cluster[np.argmax(uncertainty_scores[indices_in_cluster])]
            selected_indices.append(idx_best_in_cluster)
            
    return [file_list[idx] for idx in selected_indices]

"""
SELECTION AUTOMATIQUE
"""

def prep_data(strategy, percentage, seed = 42):
    match strategy:
        case "random":
            random_data(percentage[0], seed)
        case "diversity":
            diversity_data_selection(percentage[0], percentage[1], seed)
        case "uncertainty":
            augmentation_data(percentage[0], percentage[1])
        case "mixed_strategy":
            mixed_strategy_selection(percentage[0], percentage[1])
        case "mixed_strategy_bis":
            mixed_strategy_bis_selection(percentage[0], percentage[1])
        case "idk":
            idk_selection(percentage[0],percentage[1])