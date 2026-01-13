import cv2
import pandas as pd
import os
import glob

def extract_and_name_frames(video_path, csv_path, output_root):
    # Extraction du nom de base (ex: video_20240520_1430)
    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Création d'un sous-dossier spécifique par vidéo pour éviter les collisions
    output_dir = os.path.join(output_root, video_base_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    print(f"--- Traitement de : {video_base_name} ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recherche des coordonnées pour la frame actuelle
        frame_data = df[df['frame_id'] == frame_count]

        if not frame_data.empty:
            x = frame_data.iloc[0]['x']
            y = frame_data.iloc[0]['y']
            # Nom de fichier : video_date_heure_x_y.jpg
            file_name = f"{video_base_name}_{x}_{y}.jpg"
            cv2.imwrite(os.path.join(output_dir, file_name), frame)
        
        frame_count += 1
    
    cap.release()
    print(f"Terminé : {frame_count} frames analysées.\n")

# --- SCRIPT PRINCIPAL ---

input_folder = "data/Domain_B"          # Dossier contenant vos mp4 et csv
output_folder = "processed_frames/Domain_B"

# 1. Trouver toutes les vidéos
video_files = glob.glob(os.path.join(input_folder, "*.mp4"))

for video_path in video_files:
    # On récupère le nom du fichier sans extension (ex: video_2024_01_01)
    file_name = os.path.basename(video_path)
    
    # On extrait la partie 'date_heure' en supprimant le préfixe 'video_'
    # Si le format est strictement 'video_date_heure.mp4'
    identifier = file_name.replace("video_", "").replace(".mp4", "")
    
    # 2. Reconstruire le nom du CSV correspondant
    csv_name = f"coords_{identifier}.csv"
    csv_path = os.path.join(input_folder, csv_name)
    
    # 3. Vérifier si le CSV existe avant de lancer le traitement
    if os.path.exists(csv_path):
        extract_and_name_frames(video_path, csv_path, output_folder)
    else:
        print(f"⚠️ Attention : Pas de CSV trouvé pour {file_name} (attendu: {csv_name})")

print("🚀 Tous les fichiers ont été traités !")