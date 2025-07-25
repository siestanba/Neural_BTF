import numpy as np
import os
import re
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

# Importations Keras/TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import time

# =========================================================
# Configuration des paramètres
# =========================================================

img_width_test, img_height_test = 256, 256
total_pixels = img_height_test * img_width_test
epochs = 10
batch_size = 256
learning_rate = 0.005
validation_split = 0.2
latent_dim = 8
num_channels_rgb = 3

# =========================================================
# Configuration des arguments de ligne de commande
# =========================================================
parser = argparse.ArgumentParser(description="Entraîne un modèle pour la décomposition de l'ABRDF et génère une carte latente (Keras).")
parser.add_argument(
    '-img',
    type=str,
    default=None,
    help="Chemin du répertoire contenant les images d'entraînement (e.g., ../MANYFILES/tv000_pv000)"
)

parser.add_argument(
    '-pix',
    type=int,
    default=2500,
    help="Nombre de pixels à sélectionner aléatoirement (e.g., 2500)"
)

parser.add_argument(
    '-csv',
    type=str,
    help="Chemin du fichier contenant les coordonnées des pixels d'entraînement (e.g., data.csv)"
)

parser.add_argument(
    '-verb',
    type=int,
    default=0,
    help="1 si on veut les images de visualisation, 0 sinon"
)

parser.add_argument(
    '-tag',
    default=0,
    help="Nom ou tag du modèle pour le nommage lors de l'exportation des poids (on aura weights_model_{tag}.csv)"
)

args = parser.parse_args()

# Utiliser le répertoire spécifié par l'utilisateur ou la valeur par défaut
repertoire = args.img
nb_pixels = args.pix
csv_path = args.csv
verbose_images = args.verb
tag = args.tag

print(f"Le répertoire d'images utilisé est : {repertoire}")

# =========================================================
# Sélection des pixels
# =========================================================
if csv_path:
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=int)
    x_coords = data[:,0]
    y_coords = data[:,1]
    nb_pixels = len(x_coords)
else:
    np.random.seed(42)
    unique_indices = np.random.choice(total_pixels, size=nb_pixels, replace=False)
    x_coords = unique_indices % img_width_test
    y_coords = unique_indices // img_width_test

print(f"{len(x_coords)} pixels sélectionnés.")

print("\n--- Importation des Images ---")
pattern = re.compile(r"(\d+)\s+tl([\d.-]+)\s+pl([\d.-]+)\s+tv([\d.-]+)\s+pv([\d.-]+)\.jpg") # Pour les images issues du Bonn Dataset
#pattern = re.compile(r"img_tv0_pv0_tl(\d+)_pl(\d+)\.jpg") # Pour les images avec spécularité
images_info = []

# Fonction pour calculer les coordonées stéréographiques
def stereo(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Projection sur x y
    px = x / (1 + z + 1e-8)
    py = y / (1 + z + 1e-8)
    return px, py

# Only process if real images were found
if os.path.exists(repertoire) and os.listdir(repertoire) and not images_info:
    for nom_fichier in os.listdir(repertoire):
        if nom_fichier.endswith(".jpg"): # Type de fichier image
            match = pattern.match(nom_fichier)
            if match:
                i, tl, pl, tv, pv = match.groups() # Pour les images issues du Bonn Dataset
                #tl, pl = match.groups() # Pour les images avec spécularité
                tl, pl = int(tl), int(pl)
                px, py = stereo(np.radians(tl), np.radians(pl))

                chemin_image = os.path.join(repertoire, nom_fichier)

                original_image = Image.open(chemin_image)
                w, h = original_image.size

                left = (w - img_width_test) / 2
                top = (h - img_height_test) / 2
                right = (w + img_width_test) / 2
                bottom = (h + img_height_test) / 2
                image_test = original_image.crop((left, top, right, bottom))
                pixels_test = np.array(image_test, dtype=np.float64) / 255.0

                pixels_selected = pixels_test[y_coords, x_coords, :]

                images_info.append({
                    "tl": tl,
                    "pl": pl,
                    "px": px,
                    "py": py,
                    "pixels_test": pixels_test,
                    "pixels": pixels_selected
                })
            else:
                print(f"Nom de fichier ignoré : {nom_fichier}")

if images_info: # On vérifie que les images ont été importées
    print(f"Forme de pixels_test (première image): {images_info[0]['pixels_test'].shape} \nForme de pixels (première image): {images_info[0]['pixels'].shape}")
else:
    print("Aucune information d'image à afficher")


print("\n--- Tri des Images ---")
images_info.sort(key=lambda d: (d["tl"], d["pl"]))

num_angular_observations_per_abrdf = len(images_info)
if num_angular_observations_per_abrdf == 0:
    raise ValueError("Aucune image trouvée. Impossible de continuer.")
print(f"Nombre d'observations angulaires par ABRDF : {num_angular_observations_per_abrdf}")

if verbose_images:
    # Les 5 premières images, une fois triées
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(min(5, len(images_info))):
        axes[i].imshow(images_info[i]['pixels_test'])
        axes[i].axis('off')
    axes[1].set_title("Visualisation des 5 premières images, une fois triées")
    plt.show()

    # Visualisation des pixels sélectionnés (en utilisant la première image pour exemple)
    full_image_selected_pixels = np.zeros((img_height_test, img_width_test, 3), dtype=np.float64)
    # Assurez-vous que pixels_selected est défini, par exemple en prenant la première observation
    if images_info:
        full_image_selected_pixels[y_coords, x_coords] = images_info[0]['pixels']
        plt.figure(figsize=(6,6))
        plt.imshow(full_image_selected_pixels)
        plt.axis('off')
        plt.title("Visualisation des pixels sélectionnés")
        plt.show()


# =========================================================
# Préparation des données
# =========================================================
all_encoder_inputs = []
all_decoder_light_dirs = []
all_target_rgb = []

abrdfs_rgb_only = []
for pixel_idx in range(len(y_coords)):
    current_texel_rgb_observations = []
    for capture_info in images_info:
        r, g, b = capture_info["pixels"][pixel_idx]
        current_texel_rgb_observations.append([r, g, b])
    abrdfs_rgb_only.append(np.array(current_texel_rgb_observations, dtype=np.float64))
abrdfs_rgb_only_np = np.stack(abrdfs_rgb_only, axis=0) # Shape: (nb_pixels, num_angular_observations, 3)
print(f"Shape des ABRDFs (RGB seulement, par pixel): {abrdfs_rgb_only_np.shape}")

for pixel_idx in range(nb_pixels):
    current_pixel_abrdf = abrdfs_rgb_only_np[pixel_idx]

    for obs_idx, capture_info in enumerate(images_info):
        all_encoder_inputs.append(current_pixel_abrdf)

        px_l, py_l = capture_info["px"], capture_info["py"]
        all_decoder_light_dirs.append([px_l, py_l])

        r, g, b = capture_info["pixels"][pixel_idx]
        all_target_rgb.append([r, g, b])

final_encoder_input_np = np.stack(all_encoder_inputs, axis=0)
final_decoder_light_dirs_np = np.array(all_decoder_light_dirs, dtype=np.float64)
final_target_rgb_np = np.array(all_target_rgb, dtype=np.float64)

print(f"Shape de l'input final de l'encodeur (ABRDFs répétées): {final_encoder_input_np.shape}")
print(f"Shape des directions de lumière du décodeur finales: {final_decoder_light_dirs_np.shape}")
print(f"Shape du RGB cible final: {final_target_rgb_np.shape}")


# ===================================================
# Définition du Modèle Keras
# ===================================================

# Encoder
encoder_input = keras.Input(shape=(num_angular_observations_per_abrdf, num_channels_rgb), name='encoder_input')
x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(encoder_input)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Flatten()(x)
latent_vector = layers.Dense(latent_dim, name='latent_output')(x)

encoder = Model(inputs=encoder_input, outputs=latent_vector, name='encoder')
if verbose_images: encoder.summary()

# Decoder
latent_input = keras.Input(shape=(latent_dim,), name='latent_input')
light_dir_input = keras.Input(shape=(2,), name='light_dir_input') # px, py

decoder_input_combined = layers.concatenate([latent_input, light_dir_input], axis=-1)

x = layers.Dense(106, activation='relu', name='fc1')(decoder_input_combined)
x = layers.Dense(106, activation='relu', name='fc2')(x)
x = layers.Dense(106, activation='relu', name='fc3')(x)
x = layers.Dense(106, activation='relu', name='fc4')(x)
output_rgb = layers.Dense(3, activation='linear', name='output_rgb')(x)

decoder = Model(inputs=[latent_input, light_dir_input], outputs=output_rgb, name='decoder')
if verbose_images: decoder.summary()

# Full Model (Encoder + Decoder)
full_model_encoder_input = keras.Input(shape=(num_angular_observations_per_abrdf, num_channels_rgb), name='full_model_encoder_input')
full_model_decoder_light_dir_input = keras.Input(shape=(2,), name='full_model_decoder_light_dir_input')

# On donne le encoder_input à l'encodeur
encoded_latent_vector = encoder(full_model_encoder_input)

# On donne latent_vector et light_dir au décodeur
reconstructed_rgb = decoder([encoded_latent_vector, full_model_decoder_light_dir_input])

full_model = Model(inputs=[full_model_encoder_input, full_model_decoder_light_dir_input], outputs=reconstructed_rgb, name='full_model')
if verbose_images: full_model.summary()


# Compilation du modèle
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
full_model.compile(optimizer=optimizer, loss='mse')

# Préparation des données pour Keras
# Les données sont déjà en NumPy arrays, Keras les accepte directement
# Effectuer la division train/validation
X_train_encoder, X_val_encoder, \
X_train_decoder_light_dirs, X_val_decoder_light_dirs, \
y_train_rgb, y_val_rgb = train_test_split(
    final_encoder_input_np, final_decoder_light_dirs_np, final_target_rgb_np,
    test_size=validation_split, random_state=42
)

print(f"\nTaille du jeu d'entraînement (paires pixel-observation) : {len(y_train_rgb)}")
print(f"Taille du jeu de validation (paires pixel-observation) : {len(y_val_rgb)}")

# Entraînement du modèle
print("\n--- Initialisation et Entraînement du Modèle Keras ---")
history = full_model.fit(
    x=[X_train_encoder, X_train_decoder_light_dirs],
    y=y_train_rgb,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=([X_val_encoder, X_val_decoder_light_dirs], y_val_rgb),
    verbose=1 # Affiche la barre de progression par défaut
)

print("\nEntraînement terminé.")


# =========================================================
# Exportation des Poids
# =========================================================
output_data_dir = "data_for_exploration_keras"
os.makedirs(output_data_dir, exist_ok=True)

output_weights_name = os.path.join(output_data_dir, f'weights_model_{tag}.csv')
output_map_name = os.path.join(output_data_dir, f'latent_map_{tag}.csv')
output_tsne_filename = os.path.join(output_data_dir, f'tsne_results_{tag}.csv')

print("\n--- Exportation de poids ---")
all_weights_data = []

# Parcourir les couches du décodeur pour extraire les poids et biais
for layer in decoder.layers:
    if isinstance(layer, (layers.Dense)): # Filtrer les couches Dense
        weights = layer.get_weights() # Renvoie une liste [weights, bias] si présents

        # Traiter les poids
        if len(weights) > 0:
            weight_matrix = weights[0]
            if weight_matrix.ndim == 2:
                print(f"  - Couche: {layer.name} (poids), Forme: {weight_matrix.shape}")
                for row_idx, weight_row in enumerate(weight_matrix):
                    for col_idx, weight_val in enumerate(weight_row):
                        all_weights_data.append({
                            'layer': layer.name,
                            'type': 'weight',
                            'row_index': row_idx,
                            'col_index': col_idx,
                            'value': weight_val
                        })
            
            # Traiter les biais
            if len(weights) > 1:
                bias_vector = weights[1]
                if bias_vector.ndim == 1:
                    print(f"  - Couche: {layer.name} (biais), Forme: {bias_vector.shape}")
                    for idx, bias_val in enumerate(bias_vector):
                        all_weights_data.append({
                            'layer': layer.name,
                            'type': 'bias',
                            'row_index': 0, # Les biais sont un vecteur
                            'col_index': idx,
                            'value': bias_val
                        })

if all_weights_data:
    df_weights = pd.DataFrame(all_weights_data)
    df_weights.to_csv(output_weights_name, index=False)
    print(f"\nLes poids du modèle Keras ont été exportés avec succès dans '{output_weights_name}'")
else:
    print("Aucun poids de couche Dense/Conv1D trouvé dans le modèle Keras à exporter.")


# =========================================================
# Génération de la carte latente
# =========================================================
print("\n--- Génération de la carte latente ---")
# On utilise l'image_test complète de chaque observation pour générer la carte latente
# On utilise un tableau NumPy vide pour stocker les ABRDF de tous les pixels de l'image complète
all_pixels_abrdfs_for_encoder = np.zeros((total_pixels, num_angular_observations_per_abrdf, num_channels_rgb), dtype=np.float64)

for obs_idx, capture_info in enumerate(images_info):
    # 'pixels_test' contient l'image entière pour cette observation angulaire
    # Shape: (img_height_test, img_width_test, num_channels_rgb)
    
    # Aplatir l'image pour accéder aux pixels dans l'ordre de 0 à total_pixels-1
    flattened_image_pixels = capture_info["pixels_test"].reshape(-1, num_channels_rgb) # (total_pixels, 3)
    
    # Assigner ces observations à l'index correct (obs_idx) pour tous les pixels
    all_pixels_abrdfs_for_encoder[:, obs_idx, :] = flattened_image_pixels

print(f"Forme de l'entrée de l'encodeur pour la carte latente complète (NumPy) : {all_pixels_abrdfs_for_encoder.shape}")

# Prédiction avec l'encodeur pour obtenir les vecteurs latents
# Diviser les données en batches pour éviter de saturer la mémoire
latent_batch_size = 512
all_latent_vectors = []

# Utiliser tf.data.Dataset pour l'inférence par batch
dataset_for_latent_map = tf.data.Dataset.from_tensor_slices(all_pixels_abrdfs_for_encoder).batch(latent_batch_size)

for batch_encoder_input in tqdm(dataset_for_latent_map, desc="Processing latent map batches"):
    batch_latent_vectors = encoder.predict(batch_encoder_input, verbose=0)
    all_latent_vectors.append(batch_latent_vectors)

# Concaténer tous les vecteurs latents des batches
all_latent_vectors_np = np.concatenate(all_latent_vectors, axis=0) # Shape: (nb_pixels_full_image, latent_dim)

# Reshaper en carte 2D (height, width, latent_dim)
latent_map = all_latent_vectors_np.reshape((img_height_test, img_width_test, latent_dim))
print(f"Forme de la carte latente : {latent_map.shape}")

# Aplatir la carte latente pour l'exportation CSV
flattened_latent_map = latent_map.reshape(-1, latent_map.shape[-1])
df_latent_map = pd.DataFrame(flattened_latent_map)

# Enregistrer la carte latente
df_latent_map.to_csv(output_map_name, index=False)
print(f"Forme des données de la carte latente exportées : {flattened_latent_map.shape}")
print(f"La carte latente a été exportée vers '{output_map_name}'")


# =========================================================
# Génération et enregistrement de la t-SNE
# =========================================================
print("\n--- Calcul de t-SNE ---")
H, W, _ = latent_map.shape # Récupérer H et W de la carte latente
num_pixels_total = H * W

latent_vectors_flat = latent_map.reshape(-1, latent_dim)

print(f"Forme des données pour t-SNE : {latent_vectors_flat.shape}")

start_time = time.time()
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, learning_rate=200, random_state=42)
latent_2d = tsne.fit_transform(latent_vectors_flat)
end_time = time.time()
print(f"t-SNE calculé en {end_time - start_time:.2f} secondes.")
print(f"Forme des données après t-SNE : {latent_2d.shape}")

if verbose_images:
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], s=10, alpha=0.7)
    plt.title("Visualisation t-SNE des vecteurs latents des pixels")
    plt.xlabel("Dimension 1 de t-SNE")
    plt.ylabel("Dimension 2 de t-SNE")
    plt.grid(True)
    plt.show()

# Générer les coordonnées (X, Y) de l'image originale pour chaque pixel
image_x_coords, image_y_coords = np.meshgrid(np.arange(W), np.arange(H))
image_x_coords_flat = image_x_coords.flatten()
image_y_coords_flat = image_y_coords.flatten()

# Créer un DataFrame pour stocker toutes les informations t-SNE
tsne_data = pd.DataFrame({
    'tsne_x': latent_2d[:, 0],
    'tsne_y': latent_2d[:, 1],
    'image_x': image_x_coords_flat,
    'image_y': image_y_coords_flat
})

# Enregistrer le DataFrame dans un fichier CSV
tsne_data.to_csv(output_tsne_filename, index=False)
print(f"\nRésultats t-SNE et coordonnées d'image enregistrés dans '{output_tsne_filename}'")
print(f"Forme des données t-SNE exportées : {tsne_data.shape}")