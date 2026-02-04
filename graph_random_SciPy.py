import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Données (je les convertis en numpy array pour faciliter le travail avec Scipy)
x_labels = ['1%', '2%', '5%', '10%', '20%', '50%']
x_values = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
y_means = np.array([0.123820, 0.090929, 0.045907, 0.038439, 0.029378, 0.020274])
y_errors = np.array([0.008640, 0.008956, 0.002177, 0.005713, 0.005124, 0.001261])

# --- Partie Scipy ---
# Création d'un nouvel axe X plus dense pour avoir une courbe lisse.
# On utilise logspace car votre axe final est logarithmique.
x_new = np.logspace(np.log10(x_values.min()), np.log10(x_values.max()), 300)

# Création de l'interpolation (Spline cubique)
spl = make_interp_spline(x_values, y_means, k=3)
y_smooth = spl(x_new)
# --------------------

# Création du graphique
plt.figure(figsize=(10, 6))

# 1. On trace la courbe lissée calculée par Scipy
plt.plot(x_new, y_smooth, color='blue', label='Courbe lissée (Scipy)')

# 2. On superpose les points originaux et les barres d'erreur
# fmt='o' affiche juste les points sans les relier par des lignes droites (puisque la courbe lisse le fait)
plt.errorbar(x_values, y_means, yerr=y_errors, fmt='o', capsize=5, color='blue', label='Mesures (Random)')

# Titres et labels
plt.title('Résultats Indépendants (Random)')
plt.xlabel('Ratio / Pourcentage du domaine B ajouté au domaine A')
plt.ylabel('MSE Loss')

# Configuration de l'échelle logarithmique
plt.xscale('log')
plt.xticks(x_values, x_labels)

# Ajout d'une grille pour la lisibilité
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

# Affichage
plt.show()