from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re

mixed = "mixed_strategy"
diversity = "diversity"
uncertainty = "uncertainty"
mixed2 = "mixed_strategy_bis"
random_strat = "random"
idk = "idk"

strategies = [random_strat,diversity, uncertainty,mixed,mixed2,idk]
percentages = [1,5,10,20,50]
seeds = [42,43,44,45,46]
liste_noms = {random_strat:"Random", 
              diversity:"Diversité", 
              uncertainty:"Incertitude", 
              mixed:"Stratégie mixte 1", 
              mixed2:"Stratégie mixte 2", 
              idk:"DAL"}

def stat(data):
    numbers = []
    for f in data:
        if not f.name: continue
        prefix = f.name[:2]
        if len(prefix) == 2 and not prefix[1].isdigit():
            prefix = prefix[0] # On ne garde que le premier 
        numbers.append(prefix)
    return Counter(numbers)

def get_number_classes():
    current_dir = Path(__file__).resolve().parent
    reference = current_dir.parent / "datasets" / "Domain_B" / "active_learning"
    return stat(list(reference.glob("*.jpg")))

def explore_data(strategy, seed = 0):
    stats = {}
    current_dir = Path(__file__).resolve().parent
    if seed == 0:
        base_dir = current_dir / f"{strategy}_datasets"
    else: base_dir = current_dir /f"{strategy}_datasets_{str(seed)}"
    if not base_dir.exists():
        print(f"Le dossier {base_dir} n'existe pas")
        return
    for percentage in percentages:
        source_root = base_dir/f"{strategy}_dataset_{percentage}percent"
        number = stat(list(source_root.glob("*.jpg")))
        stats[f"{strategy}_{percentage}"] = number
    return stats


def extract_percentage(label):
    # On cherche le nombre juste avant le signe '%'
    match = re.search(r'(\d+)', label)
    if match:
        return int(match.group(1))
    return 0

def plot_results(results, ground_truth):
    # 1. Isoler la stratégie Random pour l'utiliser comme baseline
    # On crée un dictionnaire { '5%': Counter, '10%': Counter, ... }
    random_baseline = {}
    other_results = {}
    
    for label, counts in results.items():
        perc_match = re.search(r'(\d+)', label)
        perc = perc_match.group(1) if perc_match else "0"
        
        if "random" in label.lower():
            random_baseline[perc] = counts
        else:
            other_results[label] = counts

    n_plots = len(other_results)
    if n_plots == 0: return

    cols = 5
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    mandatory_classes = [str(i) for i in range(1, 10)]
    potential_keys = set(mandatory_classes).union(set(str(k) for k in ground_truth.keys()))
    all_keys = sorted(list(potential_keys), key=lambda x: int(x) if x.isdigit() else x)
    
    sorted_items = sorted(other_results.items(), key=lambda x: (x[0].split('(')[0], extract_percentage(x[0])))


    for i, (label, counts) in enumerate(sorted_items):
        ax = axes[i]
        ax.set_xlabel("Numéro vidéo", fontsize=8)
        ax.set_ylabel("Nombre d'images sélectionnées", fontsize=8)

        perc_match = re.search(r'(\d+)', label)
        current_perc = perc_match.group(1) if perc_match else None
        
        # Données à tracer
        strat_vals = [counts.get(c, 0) for c in all_keys]
        gt_vals = [ground_truth.get(c, 0) for c in all_keys]
        # On récupère le random correspondant au même pourcentage
        rand_vals = [random_baseline.get(current_perc, {}).get(c, 0) for c in all_keys]

        # A. Fond : Total disponible
        ax.bar(all_keys, gt_vals, color='lightgrey', label='Total (Domain_B)', alpha=0.3)
        
        # B. Milieu : Random (Baseline) - On l'affiche en "outline" (contour seul) pour la visibilité
        ax.step(all_keys, rand_vals, where='mid', color='red', linestyle='--', label='Baseline Random', linewidth=1.5, zorder=3)
        
        # C. Devant : La Stratégie étudiée
        bars_strat = ax.bar(all_keys, strat_vals, color='skyblue', edgecolor='navy', label='Stratégie active', alpha=0.8)

        # Annotations (Optionnel : seulement pour la stratégie active pour ne pas surcharger)
        for bar in bars_strat:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.1f}', ha='center', va='bottom', fontsize=7)

        ax.set_title(label, fontsize=10, fontweight='bold')
        if i == 0: ax.legend(loc='upper right', fontsize='small')

    for j in range(i + 1, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.savefig("analyse_repartition_finale.png")
    print(f"Graphique sauvegardé ({n_plots} sous-graphiques).")
    plt.show()

final_results = {}

for strategy in strategies:
    # On détermine quelles seeds utiliser pour cette stratégie
    current_seeds = [0]
    if strategy in [diversity, random_strat]:
        current_seeds = seeds
    elif strategy == idk:
        current_seeds = [0, 1]

    # Structure pour accumuler les compteurs par pourcentage
    # { percentage: [Counter_seed1, Counter_seed2, ...] }
    temp_accumulated = defaultdict(list)

    for s in current_seeds:
        seed_data = explore_data(strategy, s)
        for key, counter in seed_data.items():
            # key est "nomstrat_pourcentage"
            perc = key.split('_')[-1]
            temp_accumulated[perc].append(counter)

    # Calcul de la moyenne pour chaque pourcentage de la stratégie
    for perc, list_counters in temp_accumulated.items():
        if not list_counters: continue
        
        # Somme de tous les compteurs pour ce pourcentage
        mean_counter = Counter()
        for c in list_counters:
            mean_counter.update(c)
        
        # Division par le nombre de seeds pour avoir la moyenne
        nb_seeds = len(list_counters)
        for k in mean_counter:
            mean_counter[k] /= nb_seeds
            
        nom_propre = liste_noms.get(strategy,strategy)
        final_results[f"{nom_propre} ({perc} %)"] = mean_counter

number = get_number_classes()

plot_results(final_results, number)