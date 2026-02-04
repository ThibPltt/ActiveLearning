import matplotlib.pyplot as plt

# Données extraites du fichier
# Les pourcentages (axe X)
x_labels = ['1%', '2%', '5%', '10%', '20%', '50%']
x_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

# Les moyennes (axe Y)
y_means = [0.123820, 0.090929, 0.045907, 0.038439, 0.029378, 0.020274]

# Les écarts-types (barres d'erreur)
y_errors = [0.008640, 0.008956, 0.002177, 0.005713, 0.005124, 0.001261]

# Création du graphique
plt.figure(figsize=(10, 6))

# Tracé de la courbe avec barres d'erreur
# fmt='-o' relie les points par une ligne et marque les points avec des cercles
# capsize=5 ajoute les petites barres horizontales aux extrémités des barres d'erreur
plt.errorbar(x_values, y_means, yerr=y_errors, fmt='-o', capsize=5, label='Random', color='blue')

# Titres et labels
plt.title('Résultats Indépendants (Random)')
plt.xlabel('Ratio / Pourcentage du domaine B ajouté au domaine A')
plt.ylabel('MSE Loss')  # Supposition que l'unité est en secondes

# Configuration de l'échelle logarithmique pour l'axe X pour mieux voir les petits pourcentages
plt.xscale('log')
plt.xticks(x_values, x_labels) # Utilise les labels "1%", "2%" etc. au lieu de 0.01, 0.02

# Ajout d'une grille pour la lisibilité
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# Sauvegarde ou affichage
plt.show()