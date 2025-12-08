import matplotlib.pyplot as plt
import numpy as np

def plot_images_comparison(X, x_recontructed):
    """
    Display the different images (the original, the reconstruction and the difference between them)
    :param X:
    :param x_recontructed:
    :return:
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Originale (X)")
    plt.imshow(X, cmap='viridis', aspect='auto')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Reconstruction (W x H)")
    plt.imshow(x_recontructed, cmap='viridis', aspect='auto')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Différence (L = X - W X H)")
    plt.imshow(np.abs(X - x_recontructed), cmap='magma', aspect='auto')
    plt.colorbar()

    plt.tight_layout()

    plt.savefig("./output/images.png")
    plt.show()

def plot_score_evolution(score_history, time_history, y_label='Score (Erreur L)'):
    """
    Create the graphics of the evolution of the best score through time
    :param score_history:
    :param time_history:
    :param y_label:
    :return:
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time_history, score_history, marker='o', linestyle='-', color='b', markersize=3)

    ax.set_title("Évolution du Meilleur Score au fil du Temps", fontsize=14)
    ax.set_xlabel("Temps écoulé (secondes)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    ax.grid(True, linestyle='--', alpha=0.7)

    filename = "./output/evolution_score.png"
    plt.savefig(filename)

    print(f"\nGraphique enregistré sous : {filename}")
    plt.show()