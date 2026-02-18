# 👁️ Prédiction d'Eye-Tracking par Apprentissage Actif (Active Learning)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Task](https://img.shields.io/badge/Task-Regression-green)

Ce projet de recherche explore l'utilisation de l'**Active Learning (Apprentissage Actif)** pour optimiser l'entraînement d'un modèle de Deep Learning (**ResNet-18**) capable de prédire la position du regard (coordonnées X, Y) sur un écran à partir d'une simple webcam.

L'objectif est de remplacer les capteurs infrarouges coûteux par une solution logicielle, en réduisant le coût d'annotation et en améliorant la robustesse du modèle face au **Domain Gap** (changements de personne, luminosité, fond, accessoires).

---

## 📂 Architecture du Projet

L'organisation des dossiers est critique pour le bon fonctionnement des scripts d'analyse.


PROJET_AL/
│
├── IA_Training/                  # Scripts d'entraînement des modèles
│   ├── randomTraining.py         # Stratégie Baseline (Aléatoire)
│   ├── mixedTraining.py          # Stratégie Gagnante (Mixte L2 + DA)
│   ├── dataAugmTraining.py       # Stratégie Data Augmentation
│   ├── clusteringTraining.py     # Stratégie Clustering L2
│   ├── dropoutTraining.py        # Stratégie MC Dropout
│   └── best_models_xxx/          # Dossiers générés contenant les poids (.bin)
│       └── best_model_ratio_50/  # Modèle final (50% de données ajoutées)
│
├── analyze_bias.py               # Script d'analyse des biais (Spatial & Identité)
├── analyze_final.py              # Script de génération des graphiques comparatifs
├── README.md                     # Ce fichier
│
└── FINAL_DATASET/                # (Doit être situé à côté ou dans le dossier parent)
    ├── Domain_A_Train/           # Données de référence (1 personne - Source)
    ├── Domain_B_Pool/            # Pool d'Active Learning (Cible non annotée)
    └── Domain_B_Test/            # Set de test (Multiples personnes/conditions)

## 🚀 Installation

Assurez-vous d'avoir les dépendances nécessaires pour l'entraînement et l'analyse :

Bash
pip install torch torchvision transformers scikit-learn matplotlib seaborn pandas tqdm pillow

Stratégie,Type,Description
Random,Baseline,Sélection aléatoire. Sert de référence pour valider l'intelligence des autres méthodes.
MC Dropout,Uncertainty,Estime l'incertitude via la variance de plusieurs passes stochastiques (Dropout actif en inférence).
Data Augmentation,Uncertainty,"Mesure la stabilité des prédictions sur une image perturbée (rotations, bruit). Si la prédiction bouge beaucoup = Incertitude élevée."
Clustering L2,Diversity,Utilise K-Means sur les embeddings normalisés (L2) pour couvrir tout l'espace géométrique et éviter la redondance.
Mixte (L2 + DA),Hybride,"Approche en entonnoir : Sélectionne d'abord les 300 images les plus incertaines (via Data Aug), puis filtre pour garder les 100 plus diversifiées (via Clustering)."

## 📊 Résultats & Analyse
### 1. Performance Globale (Précision)
Le modèle Mixte et le Clustering L2 offrent la meilleure précision (Erreur Moyenne Absolue la plus basse).

Random : Performant mais stagne rapidement.

Dropout (0.25) : Échec massif. Le modèle "sous-apprend" (Underfitting) et prédit souvent une moyenne au centre de l'écran.

Mixte : Atteint une erreur < 70 pixels (sur un écran 1920x1080), validant l'approche "Incertitude Diversifiée".

(Graphe 1 : Plus la barre verte est basse, meilleur est le modèle)

### 2. Robustesse vs Précision (Le Trade-off)
Bien que la stratégie Mixte soit la plus précise, elle augmente légèrement la variance du modèle. La stratégie Data Augmentation est la seule à améliorer la robustesse (chiffres verts ci-dessous), créant un modèle plus stable face aux perturbations.

(Graphe 3 : Vert = Gain de stabilité, Rouge = Perte de stabilité au profit de la précision)

### 3. Analyse des Biais (Qui est sélectionné ?)
L'exécution du script analyze_bias.py révèle :

Biais Spatial : Une tendance générale des modèles à mieux prédire (ou à privilégier) le coin haut-gauche de l'écran.

Biais d'Identité : La stratégie Mixte couvre mieux les différents sujets (Arthur, Suzanne, Thibaut, Gaïa) en répartissant l'erreur, contrairement au Dropout qui échoue uniformément.

(Distribution spatiale des prédictions : On observe le biais "Central Tendency" du Dropout vs la couverture du Mixte)

## 🛠️ Utilisation
### 1. Lancer un Entraînement
Pour entraîner un modèle avec une stratégie spécifique (ex: Mixte), naviguez dans le dossier IA_Training :

Bash
cd IA_Training
python mixedTraining.py

Cela va générer un dossier best_models_mixed contenant les poids du modèle et un fichier CSV des résultats.

### 2. Comparer les Stratégies
Une fois les modèles entraînés (Random, Mixte, etc.), générez les graphiques globaux :

Bash
# Revenir à la racine du projet
python analyze_final.py
Génère : https://www.google.com/search?q=FULL_1_MOYENNES.png, FULL_2_DISTRIBUTIONS.png, https://www.google.com/search?q=FULL_3_ROBUSTESSE.png

### 3. Investiguer les Biais
Pour comprendre pourquoi un modèle échoue ou réussit (analyse par personne/position) :

Bash
python analyze_bias.py
Génère : https://www.google.com/search?q=BIAS_1_SPATIAL.png, BIAS_2_GROUPS.png

## 📚 Références
Auteur : Thibaut PLATET (5A IPS2 ENSIM)

Encadrant : Meysam SHAMSI

Papier de référence Active Learning : Tharwat, A., & Schenck, W. (2023). A Survey on Active Learning. Mathematics.

Papier de référence MC Dropout : Seoh, R. (2020). Qualitative analysis of Monte Carlo dropout. arXiv.

Projet réalisé dans le cadre du cursus ingénieur ENSIM - Le Mans Université.
