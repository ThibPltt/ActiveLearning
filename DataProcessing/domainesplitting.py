import os
import shutil
import random
from pathlib import Path

def split_dataset(domain_path, folder_1, folder_2, ratio_1):
    """
    Répartit les images d'un domaine dans deux nouveaux sous-dossiers.
    """
    # 1. Lister toutes les images de manière récursive
    all_images = []
    for root, dirs, files in os.walk(domain_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_images.append(os.path.join(root, file))
    
    if not all_images:
        print(f"Aucune image trouvée dans {domain_path}")
        return

    # 2. Créer les nouveaux dossiers cibles
    path_1 = os.path.join(domain_path, folder_1)
    path_2 = os.path.join(domain_path, folder_2)
    os.makedirs(path_1, exist_ok=True)
    os.makedirs(path_2, exist_ok=True)

    # 3. Mélanger et calculer l'index de séparation
    random.shuffle(all_images)
    split_idx = int(len(all_images) * ratio_1)

    # 4. Déplacer les fichiers
    for i, img_path in enumerate(all_images):
        dest_folder = path_1 if i < split_idx else path_2
        # On utilise shutil.move pour déplacer le fichier
        try:
            shutil.move(img_path, os.path.join(dest_folder, os.path.basename(img_path)))
        except shutil.Error:
            # Gère le cas où un fichier avec le même nom existe déjà dans la destination
            pass

    print(f"Terminé pour {os.path.basename(domain_path)} :")
    print(f" - {folder_1}: {split_idx} images")
    print(f" - {folder_2}: {len(all_images) - split_idx} images")

# --- CONFIGURATION ---
current_dir = Path(__file__).resolve().parent
base_dir = current_dir.parent / "datasets" 


"""
# Domaine A : 3/4 (0.75) train, 1/4 test
split_dataset(
    domain_path=os.path.join(base_dir, "Domain_A"),
    folder_1="train",
    folder_2="test",
    ratio_1=0.75
)

# Domaine B : 80% (0.80) active_learning, 20% test
split_dataset(
    domain_path=os.path.join(base_dir, "Domain_B"),
    folder_1="active_learning",
    folder_2="test",
    ratio_1=0.80
)
"""



import os
import shutil
from pathlib import Path

def split_domain_b_by_folder(domain_path):
    """
    Répartit les images du Domaine B :
    - Dossiers 1 à 9 -> active_learning
    - Dossiers 10 à 12 -> test (le dossier 9 inclus selon votre demande)
    """
    # 1. Créer les dossiers cibles
    path_al = os.path.join(domain_path, "active_learning")
    path_test = os.path.join(domain_path, "test")
    os.makedirs(path_al, exist_ok=True)
    os.makedirs(path_test, exist_ok=True)

    count_al = 0
    count_test = 0

    # 2. Parcourir les dossiers numérotés
    # On liste les dossiers dans Domain_B
    for entry in os.scandir(domain_path):
        if entry.is_dir() and entry.name.isdigit():
            folder_num = int(entry.name)
            
            # Définition de la destination selon votre règle
            # Note : Si vous voulez 1-9 dans AL et 10-12 dans Test, utilisez 1 <= folder_num <= 9
            if 1 <= folder_num <= 9:
                dest_folder = path_al
            elif 10 <= folder_num <= 12:
                dest_folder = path_test
            else:
                continue # Ignore les autres dossiers

            # 3. Déplacer les images
            for file in os.listdir(entry.path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    src_path = os.path.join(entry.path, file)
                    # On évite les collisions de noms en préfixant par le numéro du dossier
                    new_name = f"{folder_num}_{file}"
                    shutil.move(src_path, os.path.join(dest_folder, new_name))
                    
                    if dest_folder == path_al:
                        count_al += 1
                    else:
                        count_test += 1

    print(f"Domaine B traité :")
    print(f" - active_learning (dossiers 1-9) : {count_al} images")
    print(f" - test (dossiers 10-12) : {count_test} images")

# --- EXÉCUTION ---
domain_b_path = os.path.join(base_dir, "Domain_B")

if os.path.exists(domain_b_path):
    split_domain_b_by_folder(domain_b_path)
else:
    print("Chemin Domaine B introuvable.")