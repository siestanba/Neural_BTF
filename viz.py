# Visualisation en prenant les couleurs de l'extérieur vers l'intérieur de l'image

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

def visualize_tsne_with_distance_from_center(tsne_results_path, img_width, img_height):
    if not os.path.exists(tsne_results_path):
        print(f"Erreur : Le fichier '{tsne_results_path}' n'existe pas.")
        return

    # Charger les données
    try:
        df = pd.read_csv(tsne_results_path)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier CSV : {e}")
        return

    if not all(col in df.columns for col in ['tsne_x', 'tsne_y', 'image_x', 'image_y']):
        print("Erreur : Le fichier CSV ne contient pas toutes les colonnes requises ('tsne_x', 'tsne_y', 'image_x', 'image_y').")
        return

    print(f"Chargement des données depuis '{tsne_results_path}'...")
    print(f"Nombre de points : {len(df)}")

    # Préparer les données pour le tracé
    tsne_x = df['tsne_x']
    tsne_y = df['tsne_y']
    image_x = df['image_x']
    image_y = df['image_y']

    # Calculer le centre de l'image
    center_x = img_width / 2.0
    center_y = img_height / 2.0

    # Calculer la distance de chaque pixel au centre de l'image
    # np.sqrt((x - cx)^2 + (y - cy)^2)
    distances_from_center = np.sqrt(
        (image_x - center_x)**2 + (image_y - center_y)**2
    )

    # --- Visualisation ---
    plt.figure(figsize=(12, 10))

    # Utiliser les distances au centre pour la coloration
    scatter = plt.scatter(tsne_x, tsne_y, c=distances_from_center, cmap='inferno', s=10, alpha=0.7)

    plt.title("Visualisation t-SNE des vecteurs latents (colorés par la distance au centre de l'image)")
    plt.xlabel("Dimension 1 de t-SNE")
    plt.ylabel("Dimension 2 de t-SNE")
    plt.grid(True)
    plt.colorbar(scatter, label="Distance euclidienne au centre de l'image originale")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise les résultats t-SNE colorés par la distance des pixels au centre de l'image."
    )
    parser.add_argument(
        '-file',
        type=str,
        required=True,
        help="Chemin vers le fichier CSV des résultats t-SNE (e.g., tsne_results_1.csv)"
    )
    parser.add_argument(
        '-width',
        type=int,
        default=256,
        help="Largeur de l'image originale (e.g., 256)"
    )
    parser.add_argument(
        '-height',
        type=int,
        default=256,
        help="Hauteur de l'image originale (e.g., 256)"
    )
    args = parser.parse_args()

    visualize_tsne_with_distance_from_center(args.file, args.width, args.height)