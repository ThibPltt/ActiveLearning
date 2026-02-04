# ActiveLearning

Cours d'active learning en 5A. Ce projet explore l'adaptation de domaine et la robustesse des modèles d'apprentissage.

## 📂 Structure du projet

Voici l'arborescence des fichiers telle qu'organisée actuellement :

```text
.
├── output.txt
├── README.md
├── resultats_gpu_independent.txt
├── data
│   ├── Domain_A
│   │   ├── coords_20260113_103624.csv
│   │   └── video_20260113_103624.mp4
│   └── Domain_B
│       ├── 1
│       ├── 2
│       ├── ... (Sous-dossiers 1 à 12 contenant vidéos et CSV)
├── DataCollection
│   └── dataCollection.py
├── DataProcessing
│   └── dataprocessing.py
├── IA_test
│   └── test_domain_A.py
├── IA_training
│   ├── ai_random_cpu.py
│   ├── ai_random_gpu.py
│   ├── ai_training_domain_A.py
│   ├── get_model.py
│   ├── testGpu.py
│   ├── resultats_random.txt
│   └── Logs
│       └── Random.txt
├── modele_pretrain_source_A
│   ├── config.json
│   └── model.safetensors
└── processed_frames
    └── Domain_A
        └── ... (Images extraites des vidéos)

```

## 📝 À faire (Roadmap)

> Attention, les données de test de doivent jamais changer, elle doivent être sur un corpus fixe

* Creer un dossier train et test pour le domaine A
* Creer un dossier train (ActiveLearning) et test pour le domaine B
* Refaire l'entrainement pour le domaine A
* Refaire le test pour le domaine A
* Faire test sur test/Domaine B avec le modèle A (voir que c'est nul)
* Faire de l'active learning
    * Randoms
    * les stratégies
* Prendre les deux meilleurs stratégie pour les mixer

> A chaque train il faut également effectuer des test

> Attention gestion des ration, incrémental c'est a dire 1% pour passer a 2%, nous rajoutons 1% puis pour passer a 5% nous rajoutons 3%, etc

### 1. Tests initiaux

* [ ] **Tests sur le Domaine B uniquement**
    * *Objectif :* Vérifier si le modèle se "perd" face à des environnements inconnus ou s'il conserve une certaine robustesse sans entraînement spécifique.



### 2. Adaptation de Domaine (Domain B sur A)

Entraînement incrémental du modèle pour l'adapter au nouveau domaine.

* [ ] **Protocole d'ajout de données**
    * Ajouter progressivement **1%, 2%, 5%, 10% et 20%** des données du Domaine B.


* [ ] **Analyse de la variance**
    * Répéter l'expérience **5 fois** pour chaque pourcentage.
    * *Objectif :* S'assurer que les résultats ne sont pas dus à la chance (variance statistique).
    * *Sauvegarde :* Enregistrer chaque modèle pour calculer la variance (ou garder uniquement le meilleur si la variance est négligeable).


* [ ] **Visualisation**
    * Afficher les graphiques de la **MSE Loss** pour analyser la convergence de l'apprentissage (la courbe doit diminuer).

* **Implémentation des différentes stratégies**
    * train_strategy_uncertainty
    * train_strategy_augmentation
    * train_strategy_probabilistic



### 3. Validation croisée

* [ ] **Back-testing**
    * Tester chaque nouveau modèle (fine-tuné sur le Domaine B) sur le **Domaine A** pour vérifier l'absence d'oubli catastrophique (régression des performances sur le domaine source).


## 📁 Répartition des Données

Voici comment les datasets sont divisés pour l'entraînement et le test. Pour le Domaine B, nous utilisons des sous-ensembles progressifs de la partie "Train" pour tester l'adaptation.

| Domaine | Jeu de données | Volume Global | Détails / Expérimentations |
| :---: | :--- | :---: | :--- |
| **Domaine A** | Train | **75%** | Base d'apprentissage source |
| | Test | **25%** | Validation baseline |
| **Domaine B** | **Train** | **80%** | **Incrémentation (Active Learning) :**<br>🔹 1% du volume Train<br>🔹 2% du volume Train<br>🔹 5% du volume Train<br>🔹 10% du volume Train<br>🔹 20% du volume Train<br>🔹 50% du volume Train |
| | Test | **20%** | Test de généralisation |

---

## 📊 Résultats

### Domaine A (Source)

Résultats de référence obtenus avec le modèle pré-entraîné sur la source A.

| Métrique | Valeur | Détails |
| :--- | :--- | :--- |
| **Modèle** | `modele_pretrain_source_A` | |
| **MSE LOSS** | `0.028582` | 📉 Plus bas est mieux |
| **Erreur Pixels** | **171.59 px** | Moyenne |
| | 0.56 px | Minimum |
| | 692.34 px | Maximum |

### Domaine B (Target)

Résultats de l'adaptation de domaine (Domain Adaptation) en ajoutant progressivement des données du Domaine B.

* **Matériel** : NVIDIA RTX 4060 Ti
* **Protocole** : 3 entraînements indépendants par pourcentage pour garantir la robustesse.

| % Données Ajoutées | MSE Loss (Moyenne) | Écart (+/-) |
| :---: | :---: | :---: |
| **1%** | 0.123820 | 0.008640 | 
| **2%** | 0.090929 | 0.008956 | 
| **5%** | 0.045907 | 0.002177 | 
| **10%** | 0.038439 | 0.005713 | 
| **20%** | 0.029378 | 0.005124 | 
| **50%** | 0.020274 | 0.001261 |