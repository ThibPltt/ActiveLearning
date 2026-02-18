import os
import datetime
from pathlib import Path
import sys

# ==========================================
# CONFIGURATION
# ==========================================

# Objectif : Trouver les dossiers de modèles et estimer le temps 
# de formation à partir des timestamps des fichiers

TARGET_FOLDERS = [
    "best_models_random",
    "best_models_dropout_0.25",
    "best_models_dropout_0.5",
    "best_models_data_aug",
    "best_models_clustering",
    "best_models_clustering_L2",
    "best_models_learning_loss"
]

# Force l'encodage utf-8 pour la console Windows
sys.stdout.reconfigure(encoding='utf-8')

# On part du dossier du script et on remonte à la racine du projet
current_script_dir = Path(__file__).resolve().parent
project_root = current_script_dir.parent

print(f"--- RECHERCHE ET ANALYSE DES TEMPS ---")
print(f"Racine de recherche : {project_root}")

def find_folder_recursive(root_path, folder_name):
    """Cherche un dossier récursivement"""
    for root, dirs, files in os.walk(root_path):
        if folder_name in dirs:
            return Path(root) / folder_name
    return None

for folder_name in TARGET_FOLDERS:
    # On lance la recherche
    found_path = find_folder_recursive(project_root, folder_name)
    
    if found_path is None:
        print(f"\n[INTROUVABLE] : {folder_name}")
        continue

    print(f"\n[TROUVÉ] : {folder_name}")
    
    timestamps = []
    
    # Analyse des fichiers dans le dossier trouvé
    for root, dirs, files in os.walk(found_path):
        for name in files:
            full_path = os.path.join(root, name)
            try:
                stat = os.stat(full_path)
                timestamps.append(stat.st_mtime)
            except:
                pass

    if not timestamps:
        print("   -> Dossier vide.")
        continue

    start_time = min(timestamps)
    end_time = max(timestamps)
    
    total_seconds = end_time - start_time
    if total_seconds < 0: total_seconds = 0
    
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    
    # Affichage propre
    print(f"   Debut : {datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Fin   : {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   -----------------------------------------")
    if hours > 0:
        print(f"   > DUREE : {hours}h {remaining_minutes}min {seconds}s")
    else:
        print(f"   > DUREE : {minutes}min {seconds}s")