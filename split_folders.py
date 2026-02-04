import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. CORRECTION DU CHEMIN RACINE
# ==========================================
# C'est ICI que je me trompais avant. J'ai rajouté "ActiveLearning".
project_root = Path(r"C:\Users\Thibaut\Desktop\ENSIM\5A_INFO\ACTIVE_INTERACTIVE_LEARNING\PROJET\ActiveLearning")

source_dir = project_root / "processed_frames"
dest_root = project_root.parent / "FINAL_DATASET" # On le met à la racine du PROJET pour être propre

# ==========================================
# 2. DEFINITION DES SOURCES SPÉCIFIQUES
# ==========================================
src_A_root = source_dir / "Domain_A"
src_B_root = source_dir / "Domain_B"

# Structure cible
dirs = {
    "train_A": dest_root / "Domain_A_Train",
    "test_A":  dest_root / "Domain_A_Test",
    "pool_B":  dest_root / "Domain_B_Pool",
    "test_B":  dest_root / "Domain_B_Test"
}

def setup_directories():
    if dest_root.exists():
        try:
            shutil.rmtree(dest_root)
        except:
            pass
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Dossiers créés dans {dest_root}")

def get_images_flattened(root_folder, domain_type):
    """
    Va chercher les images spécifiquement selon la structure décrite.
    """
    images = []
    valid_ext = [".jpg", ".JPG", ".jpeg", ".png"]
    
    if not root_folder.exists():
        print(f"!!! ERREUR : Le dossier {root_folder} n'existe pas !!!")
        return []

    # --- LOGIQUE POUR DOMAINE A (Sous-dossier vidéo) ---
    if domain_type == "A":
        # On prend tout ce qu'il y a dans les sous-dossiers (ex: video_2026...)
        for path in root_folder.rglob("*"):
            if path.suffix in valid_ext:
                images.append(path)
                
    # --- LOGIQUE POUR DOMAINE B (Dossiers 1 à 12) ---
    elif domain_type == "B":
        # On parcourt explicitement les dossiers 1, 2, ... 12
        # On va même chercher large : tous les dossiers qu'on trouve
        subfolders = [f for f in root_folder.iterdir() if f.is_dir()]
        print(f"   [INFO] Domaine B : trouvé {len(subfolders)} sous-dossiers (ex: {subfolders[0].name}...)")
        
        for sub in subfolders:
            for path in sub.glob("*"): # On regarde dedans
                if path.suffix in valid_ext:
                    images.append(path)

    return list(set(images))

def copy_files(file_list, destination, desc):
    for f in tqdm(file_list, desc=desc):
        try:
            shutil.copy2(f, destination / f.name)
        except shutil.SameFileError:
            pass

# ==========================================
# 3. EXECUTION
# ==========================================
print(f"--- DÉMARRAGE AVEC CHEMIN CORRIGÉ ---")
print(f"Racine : {source_dir}")

setup_directories()

# --- TRAITEMENT A ---
print("\n[1/2] Récupération Domaine A (Arthur)...")
files_A = get_images_flattened(src_A_root, "A")

if not files_A:
    print(f"ERREUR : Toujours rien dans A. Vérifie que le dossier s'appelle bien :")
    print(f"{src_A_root}")
    exit()

random.shuffle(files_A)
split_A = int(len(files_A) * 0.75)
copy_files(files_A[:split_A], dirs["train_A"], "Train A (75%)")
copy_files(files_A[split_A:], dirs["test_A"],  "Test A (25%)")

# --- TRAITEMENT B ---
print("\n[2/2] Récupération Domaine B (1-12)...")
files_B = get_images_flattened(src_B_root, "B")

if not files_B:
    print(f"ERREUR : Toujours rien dans B.")
    exit()

random.shuffle(files_B)
split_B = int(len(files_B) * 0.80)
copy_files(files_B[:split_B], dirs["pool_B"], "Pool B (80%)")
copy_files(files_B[split_B:], dirs["test_B"], "Test B (20%)")

print("\n" + "="*50)
print(f"TOUT EST BON.")
print(f"Images A trouvées : {len(files_A)}")
print(f"Images B trouvées : {len(files_B)}")
print(f"Dossier final : {dest_root}")
print("="*50)