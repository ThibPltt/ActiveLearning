import cv2
import pandas as pd
import os
import glob
from pathlib import Path

def extract_and_name_frames(video_path, csv_path, output_root, subfolder_id):
    # Extraction du nom de base (ex: video_20240520_1430)
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Création du chemin : output_root / numéro_sous_dossier / nom_video
    # Cela permet de conserver la hiérarchie par numéro
    output_dir = os.path.join(output_root, subfolder_id)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    print(f"--- Traitement de [{subfolder_id}] : {video_base_name} ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recherche des coordonnées pour la frame actuelle
        frame_data = df[df['frame_id'] == frame_count]

        if not frame_data.empty:
            x = frame_data.iloc[0]['x']
            y = frame_data.iloc[0]['y']
            # Nom de fichier conservé : video_date_heure_x_y.jpg
            file_name = f"{video_base_name}_{x}_{y}.jpg"
            cv2.imwrite(os.path.join(output_dir, file_name), frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Terminé : {frame_count} frames analysées.\n")

# --- SCRIPT PRINCIPAL ---
current_dir = Path(__file__).resolve().parent
input_folder = current_dir.parent / "raw_data"/"Domain_B" 
output_folder = current_dir.parent / "datasets"/"Domain_B"

# 1. Trouver toutes les vidéos dans tous les sous-dossiers (recursive=True)
# Le pattern "**/*.mp4" cherche dans tous les niveaux de sous-répertoires
video_files = glob.glob(os.path.join(input_folder, "**/*.mp4"), recursive=True)

for video_path in video_files:
    # Récupérer le nom du dossier parent immédiat (le numéro)
    # os.path.dirname(video_path) donne le chemin du dossier, 
    # os.path.basename de ce chemin donne le nom du dernier dossier
    subfolder_id = os.path.basename(os.path.dirname(video_path))
    
    file_name = os.path.basename(video_path)
    
    # Extraction de l'identifiant pour le CSV
    identifier = file_name.replace("video_", "").replace(".mp4", "")
    csv_name = f"coords_{identifier}.csv"
    
    # On cherche le CSV dans le même sous-dossier que la vidéo
    csv_path = os.path.join(os.path.dirname(video_path), csv_name)
    
    # 3. Vérifier si le CSV existe avant de lancer le traitement
    if os.path.exists(csv_path):
        extract_and_name_frames(video_path, csv_path, output_folder, subfolder_id)
    else:
        print(f"⚠️ Attention : Pas de CSV trouvé dans {subfolder_id} pour {file_name}")

print("🚀 Tous les fichiers ont été traités avec succès !")