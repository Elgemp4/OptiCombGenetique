import matplotlib.pyplot as plt
import numpy as np

def plot_images_comparison(X, x_recontructed):
    plt.figure(figsize=(12, 6))

    # --- Image A : La matrice Originale (Cible) ---
    plt.subplot(1, 3, 1)
    plt.title("Originale (Cible)")
    # 'cmap' définit les couleurs (ex: 'gray', 'viridis', 'plasma')
    plt.imshow(X, cmap='viridis', aspect='auto')
    plt.savefig("original.png")
    plt.colorbar()

    # --- Image B : La matrice Reconstruite (Résultat) ---
    plt.subplot(1, 3, 2)
    plt.title("Reconstruction (W x H)")
    plt.imshow(x_recontructed, cmap='viridis', aspect='auto')
    plt.savefig("reconstructed.png")
    plt.colorbar()

    # --- Image C : La différence (L'erreur visible) ---
    # C'est très utile pour voir ce que l'algo a raté
    plt.subplot(1, 3, 3)
    plt.title("Différence (Erreur)")
    plt.imshow(np.abs(X - x_recontructed), cmap='magma', aspect='auto')
    plt.savefig("error.png")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def plot_score_evolution(score_history, time_history, y_label='Score (Erreur L)'):
    """
    Crée le graphique de l'évolution du score en fonction du temps.
    """

    # Créer la figure et l'axe
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer les données (Score vs. Temps)
    ax.plot(time_history, score_history, marker='o', linestyle='-', color='b', markersize=3)

    # Ajouter des labels et un titre
    ax.set_title("Évolution du Meilleur Score au fil du Temps", fontsize=14)
    ax.set_xlabel("Temps écoulé (secondes)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    # Ajouter une grille pour faciliter la lecture
    ax.grid(True, linestyle='--', alpha=0.7)

    # Mettre l'axe Y à l'échelle logarithmique si le score change énormément
    # (Utile si vous commencez à 300 millions et finissez à 1 million)
    # ax.set_yscale('log')
    filename = "evolution_score.png"
    plt.savefig(filename)

    print(f"\nGraphique enregistré sous : {filename}")
    # Afficher le graphique
    plt.show()