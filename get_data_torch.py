import numpy as np
import os
import re
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import time

# Importations Pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.manifold import TSNE



# =========================================================
# Configuration des paramètres
# =========================================================

img_width_test, img_height_test = 256, 256
total_pixels = img_height_test * img_width_test
epochs = 10
batch_size = 256 # Adjusted to match TensorFlow's effective batch_size if it trains on (nb_pixels * samples) lines per epoch
learning_rate = 0.005
validation_split = 0.2
latent_dim = 8
num_channels_rgb = 3 # RGB only for encoder input

# =========================================================
# Configuration des arguments de ligne de commande
# =========================================================
parser = argparse.ArgumentParser(description="Entraîne un modèle pour la décomposition de l'ABRDF et génère une carte latente.")
parser.add_argument(
    '-img',
    type=str,
    default=None, # Valeur par défaut si non spécifié par l'utilisateur
    help="Chemin du répertoire contenant les images d'entraînement (e.g., ../MANYFILES/tv000_pv000)"
)

parser.add_argument(
    '-pix',
    type=int,
    default = 2500,
    help="Nombre de pixels à sélectionner aléatoirement (e.g., 2500)" 
    # A préciser seulement si on ne les choisit pas manuellement avec un fichier .csv
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
    help="1 si on veut les images, 0 sinon"
)

parser.add_argument(
    '-tag',
    default = 0,
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
# encoder_input: (nb_pixels * samples, samples, 3) --> Chaque ABRDF pour un pixel, répétée 'samples' fois
# decoder_input (light_dirs): (nb_pixels * samples, 2) --> Une direction de lumière par ligne
# target_rgb: (nb_pixels * samples, 3) --> Une couleur RGB cible par ligne

all_encoder_inputs = [] # Stocke les ABRDFs complètes (RGB) pour CHAQUE pixel, pour CHAQUE observation de lumière
all_decoder_light_dirs = [] # Stocke la direction de lumière pour CHAQUE observation (px, py)
all_target_rgb = [] # Stocke la couleur RGB pour CHAQUE observation (R, G, B)

# Obtenir toutes les ABRDFs complètes (num_pixels, num_angular_observations, 3)
abrdfs_rgb_only = []
for pixel_idx in range(len(y_coords)):
    current_texel_rgb_observations = []
    for capture_info in images_info:
        r, g, b = capture_info["pixels"][pixel_idx]
        current_texel_rgb_observations.append([r, g, b])
    abrdfs_rgb_only.append(np.array(current_texel_rgb_observations, dtype=np.float64))
abrdfs_rgb_only_np = np.stack(abrdfs_rgb_only, axis=0) # Shape: (nb_pixels, num_angular_observations, 3)
print(f"Shape of ABRDFs (RGB only, per pixel): {abrdfs_rgb_only_np.shape}")


# Construire les inputs
# L'itération sera sur chaque pixel, puis sur chaque observation angulaire pour ce pixel
for pixel_idx in range(nb_pixels):
    current_pixel_abrdf = abrdfs_rgb_only_np[pixel_idx] # (num_angular_observations, 3)

    for obs_idx, capture_info in enumerate(images_info):
        # L'entrée de l'encodeur est l'ABRDF complète du pixel (num_angular_observations, 3)
        # Elle est répétée 'num_angular_observations' fois pour chaque pixel.
        all_encoder_inputs.append(current_pixel_abrdf)

        # La direction de lumière est celle de l'observation actuelle
        px_l, py_l = capture_info["px"], capture_info["py"]
        all_decoder_light_dirs.append([px_l, py_l])

        # La cible RGB est la couleur de ce pixel pour cette observation
        r, g, b = capture_info["pixels"][pixel_idx]
        all_target_rgb.append([r, g, b])

# Convertir en NumPy arrays
final_encoder_input_np = np.stack(all_encoder_inputs, axis=0)
final_decoder_light_dirs_np = np.array(all_decoder_light_dirs, dtype=np.float64)
final_target_rgb_np = np.array(all_target_rgb, dtype=np.float64)

print(f"Shape of final encoder input (repeated ABRDFs): {final_encoder_input_np.shape}")
print(f"Shape of final decoder light directions: {final_decoder_light_dirs_np.shape}")
print(f"Shape of final target RGB: {final_target_rgb_np.shape}")


# ===================================================
# Création du PyTorch Dataset et DataLoader
# ===================================================

class CustomABRFDatasetTensorFlowLike(torch.utils.data.Dataset):
    def __init__(self, encoder_input, decoder_light_dirs, target_rgb):
        self.encoder_input = encoder_input # (nb_pixels * num_obs, num_obs, 3)
        self.decoder_light_dirs = decoder_light_dirs # (nb_pixels * num_obs, 2)
        self.target_rgb = target_rgb # (nb_pixels * num_obs, 3)

    def __len__(self):
        return self.encoder_input.shape[0] # Nombre total d'échantillons individuels (pixel_obs)

    def __getitem__(self, idx):
        return self.encoder_input[idx], self.decoder_light_dirs[idx], self.target_rgb[idx]

# Convertir les tableaux NumPy en tenseurs PyTorch
encoder_input_tensor = torch.from_numpy(final_encoder_input_np).float()
decoder_light_dirs_tensor = torch.from_numpy(final_decoder_light_dirs_np).float()
target_rgb_tensor = torch.from_numpy(final_target_rgb_np).float()

dataset_tf_like = CustomABRFDatasetTensorFlowLike(encoder_input_tensor, decoder_light_dirs_tensor, target_rgb_tensor)

# On sépare le datasent en train/validation
train_size = int(0.8 * len(dataset_tf_like))
val_size = len(dataset_tf_like) - train_size
train_dataset, val_dataset = random_split(dataset_tf_like, [train_size, val_size])

# Création du DataLoaders
# Chaque élément du loader sera un tuple ou un tensor:
#   encoder_input_batch: (batch_size, num_angular_observations, 3)
#   decoder_light_dirs_batch: (batch_size, 2)
#   target_rgb_batch: (batch_size, 3)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"\nTaille du jeu d'entraînement (paires pixel-observation): {len(train_dataset)}")
print(f"Taille du jeu de validation (paires pixel-observation): {len(val_dataset)}")


# ===================================================
# Définition du Modèle PyTorch
# ===================================================

class Encoder(nn.Module):
    def __init__(self, latent_dim, num_angular_observations, num_features_per_observation):
        super(Encoder, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=num_features_per_observation, out_channels=128, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        
        # On calcule la taille de la sortie Conv1D + Pooling avant d'applatir
        flattened_size = 128 * (num_angular_observations // 2)
        self.dense_latent = nn.Linear(flattened_size, latent_dim)

    def forward(self, x):
        # x : (batch_size, num_angular_observations, num_features_per_observation)
        # Et on permute (batch_size, num_features_per_observation, num_angular_observations) pour la Conv1D
        x = x.permute(0, 2, 1) 
        x = torch.relu(self.bn1(self.conv1d_1(x)))
        x = self.pool1(x)
        x = self.flatten(x)
        return self.dense_latent(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, angle_dim=2):
        super(Decoder, self).__init__()
        input_dim = latent_dim + angle_dim
        self.fc1 = nn.Linear(input_dim, 106)
        self.fc2 = nn.Linear(106, 106)
        self.fc3 = nn.Linear(106, 106)
        self.fc4 = nn.Linear(106, 106)
        self.output_rgb = nn.Linear(106, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.output_rgb(x)

class FullModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(FullModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input_batch, decoder_light_dirs_batch):
        # encoder_input_batch: (batch_size, num_angular_observations, 3) -> Chaque ligne est une ABRDF complète
        # decoder_light_dirs_batch: (batch_size, 2) -> Chaque ligne est une direction de lumière unique pour l'input du décodeur

        # On prend le vecteur latent pour chaque ABRDF du batch
        latent_vectors = self.encoder(encoder_input_batch) # Shape: (batch_size, latent_dim)

        # On le concatène avec la direction spécifique pour chaque échantillon du batch
        # latent_vectors et decoder_light_dirs_batch sont déjà de la même taille
        decoder_input = torch.cat((latent_vectors, decoder_light_dirs_batch), dim=-1) # Shape: (batch_size, latent_dim + 2)

        reconstructed_rgb = self.decoder(decoder_input) # Shape: (batch_size, 3)
        
        return reconstructed_rgb, latent_vectors

# Entraînement du modèle
print("\n--- Initialisation et Entraînement du Modèle PyTorch ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

encoder = Encoder(latent_dim, num_angular_observations_per_abrdf, num_channels_rgb).to(device)
decoder = Decoder(latent_dim).to(device)
model = FullModel(encoder, decoder).to(device)

# On prend la même fonction de perte et epsilon que l'optimisateur adam par défaut dans Keras
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)

# Initialisation des poids pour qu'ils soient plus proches des valeurs par défaut de Keras (Glorot uniforme pour Dense/Linéaire)
# Keras Conv1D utilise Glorot uniforme pour le kernel, et zéro pour les biais
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight) # Glorot uniform
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias) # Zeros bias
model.apply(init_weights)
print("Poids initialisée avec Xavier Uniform et Bias à Zéro.")

torch.manual_seed(42) # On spécifie la seed pour limiter les paramètres aléatoires
np.random.seed(42)

# Training loop
history = {'loss': [], 'val_loss': []}

print("\nDébut de l'entraînement...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)", leave=False)
    for i, (encoder_input_batch, decoder_light_dirs_batch, ground_truth_rgb_batch) in enumerate(train_loader_tqdm):
        encoder_input_batch = encoder_input_batch.to(device)
        decoder_light_dirs_batch = decoder_light_dirs_batch.to(device)
        ground_truth_rgb_batch = ground_truth_rgb_batch.to(device)

        optimizer.zero_grad()

        reconstructed_rgb, _ = model(encoder_input_batch, decoder_light_dirs_batch)
        
        loss = criterion(reconstructed_rgb, ground_truth_rgb_batch)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * encoder_input_batch.size(0) # Accumule la loss pour chaque batch
        
        train_loader_tqdm.set_postfix(loss=f"{loss.item():.7f}") # Pour l'affichage

    epoch_loss = running_loss / len(train_loader.dataset)
    history['loss'].append(epoch_loss)

    # Validation
    model.eval()
    val_running_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Val)", leave=False)
    with torch.no_grad():
        for val_encoder_input_batch, val_decoder_light_dirs_batch, val_ground_truth_rgb_batch in val_loader_tqdm:
            val_encoder_input_batch = val_encoder_input_batch.to(device)
            val_decoder_light_dirs_batch = val_decoder_light_dirs_batch.to(device)
            val_ground_truth_rgb_batch = val_ground_truth_rgb_batch.to(device)

            val_reconstructed_rgb, _ = model(val_encoder_input_batch, val_decoder_light_dirs_batch)
            
            val_loss = criterion(val_reconstructed_rgb, val_ground_truth_rgb_batch)
            val_running_loss += val_loss.item() * val_encoder_input_batch.size(0)
            
            val_loader_tqdm.set_postfix(val_loss=f"{val_loss.item():.7f}")
    
    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    history['val_loss'].append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.7f}, Val Loss: {epoch_val_loss:.7f}")

print("\nEntraînement terminé")

# =========================================================
# Exportation des Poids
# =========================================================
output_data_dir = "data_for_exploration"
os.makedirs(output_data_dir, exist_ok=True) # S'assurer que le répertoire existe

output_weights_name = os.path.join(output_data_dir, f'weights_model_{tag}.csv')
output_map_name = os.path.join(output_data_dir, f'latent_map_{tag}.csv')
output_tsne_filename = os.path.join(output_data_dir, f'tsne_results_{tag}.csv')

print("\n--- Exportation de poids ---")
all_weights_data = []
state_dict = decoder.state_dict()

for name, param in state_dict.items():
    # On s'intéresse aux poids et biais des couches linéaires
    if 'weight' in name and param.dim() == 2: # Poids d'une couche linéaire (matrice 2D)
        layer_name = name.replace('.weight', '')
        weights = param.cpu().numpy() # Convertir en NumPy array
        print(f"  - Couche: {layer_name} (poids), Forme: {weights.shape}")

        for row_idx, weight_row in enumerate(weights):
            for col_idx, weight_val in enumerate(weight_row):
                all_weights_data.append({
                    'layer': layer_name,
                    'type': 'weight',
                    'row_index': row_idx,
                    'col_index': col_idx,
                    'value': weight_val
                })

    elif 'bias' in name and param.dim() == 1: # Biais
        layer_name = name.replace('.bias', '')
        biases = param.cpu().numpy() # Convertir en NumPy array
        print(f"  - Couche: {layer_name} (biais), Forme: {biases.shape}")

        for idx, bias_val in enumerate(biases):
            all_weights_data.append({
                'layer': layer_name,
                'type': 'bias',
                'row_index': 0, # Les biais sont un vecteur, on utilise 0 pour la ligne
                'col_index': idx,
                'value': bias_val
            })

# On créex un DataFrame pandas pour l'exporter en CSV
if all_weights_data:
    df = pd.DataFrame(all_weights_data)
    df.to_csv(output_weights_name, index=False)
    print(f"\nLes poids du modèle PyTorch ont été exportés avec succès dans '{output_weights_name}'")
else:
    print("Aucun poids de couche linéaire (weight/bias) trouvé dans le modèle PyTorch à exporter.")


# =========================================================
# Génération de la carte latente
# =========================================================
print("\n-- Génération de la carte latente ---")
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

print(f"Forme de l'entrée de l'encodeur pour la carte latente complète (NumPy): {all_pixels_abrdfs_for_encoder.shape}")



# On met l'encodeur en mode évaluation et sur le bon appareil
encoder.eval()
encoder.to(device)

# Convertir le NumPy array en Tensor PyTorch et le déplacer vers le bon appareil
# Shape: (nb_pixels_full_image, num_angular_observations_per_abrdf, num_channels_rgb)
encoder_input_for_all_pixels_tensor = torch.from_numpy(all_pixels_abrdfs_for_encoder).float().to(device) 

print(f"Forme de l'entrée de l'encodeur (tensor) : {encoder_input_for_all_pixels_tensor.shape}")

# Prédiction avec l'encodeur pour obtenir les vecteurs latents
# Diviser les données en batches pour éviter de saturer la mémoire
latent_batch_size = 512
latent_dataset = TensorDataset(encoder_input_for_all_pixels_tensor)
latent_loader = DataLoader(latent_dataset, batch_size=latent_batch_size, shuffle=False)

all_latent_vectors = []

# C'est ici que l'on commence à générer la carte latente
with torch.no_grad(): # Désactiver le calcul de gradient pour l'inférence
    for batch_idx, (batch_encoder_input,) in enumerate(tqdm(latent_loader, desc="Processing latent map batches")):
        batch_encoder_input = batch_encoder_input.to(device)
        
        # L'encodeur prendra (batch_size, num_angular_observations, 3)
        # et produira (batch_size, latent_dim)
        batch_latent_vectors = encoder(batch_encoder_input)
        
        all_latent_vectors.append(batch_latent_vectors.cpu()) # sur CPU pour éviter les problèmes de mémoire GPU

    # Concaténer tous les vecteurs latents des batches
    all_latent_vectors_tensor = torch.cat(all_latent_vectors, dim=0) # Shape: (nb_pixels_full_image, latent_dim)
    
    # Reshaper en carte 2D (height, width, latent_dim)
    latent_map = all_latent_vectors_tensor.numpy().reshape((img_height_test, img_width_test, latent_dim))
    print(f"Forme de la carte latente: {latent_map.shape}")


# Aplatir la carte latente pour l'exportation CSV
flattened_latent_map = latent_map.reshape(-1, latent_map.shape[-1])
df = pd.DataFrame(flattened_latent_map)

# Enregistrer la carte latente
df.to_csv(output_map_name, index=False)
print(f"Forme des données exportées : {flattened_latent_map.shape}")
print(f"La carte latente a été exportée vers '{output_map_name}'")

# =========================================================
# Génération de la clusterisation t-SNE
# =========================================================
print("--- Calcul de t-SNE ---")
H, W, latent_dim = latent_map.shape
num_pixels = H * W

# Aplatir la carte latente pour obtenir tous les vecteurs latents de pixels
latent_vectors_flat = latent_map.reshape(-1, latent_dim) # transforme (H, W, latent_dim) en (H*W, latent_dim)

print(f"\nForme des données pour t-SNE: {latent_vectors_flat.shape}")

# - n_components: la dimension de l'espace de sortie (2 pour la visualisation 2D)
# - perplexity: mesure l'équilibre entre l'attention portée aux voisins locaux et globaux
# - max_iter: nombre maximum d'itérations
# - learning_rate: taux d'apprentissage de l'optimisation
# - random_state: pour la reproductibilité des résultats

start_time = time.time()
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, learning_rate=200, random_state=42)

# Effectuer la réduction de dimensionnalité
# Le résultat 'latent_2d' aura la forme (num_pixels, 2)
latent_2d = tsne.fit_transform(latent_vectors_flat)

end_time = time.time()
print(f"t-SNE calculé en {end_time - start_time:.2f} secondes.")
print(f"Forme des données après t-SNE: {latent_2d.shape}")

if verbose_images:
    # Visualisation des résultats t-SNE
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], s=10, alpha=0.7)
    plt.title("Visualisation t-SNE des vecteurs latents des pixels")
    plt.xlabel("Dimension 1 de t-SNE")
    plt.ylabel("Dimension 2 de t-SNE")
    plt.grid(True)
    plt.show()

# Générer les coordonnées (X, Y) de l'image originale pour chaque pixel
# L'ordre doit correspondre à l'aplatissement de latent_map.reshape(-1, latent_dim)
# (0,0), (0,1), ..., (0, W-1), (1,0), (1,1), ...
image_x_coords, image_y_coords = np.meshgrid(np.arange(W), np.arange(H))
image_x_coords_flat = image_x_coords.flatten()
image_y_coords_flat = image_y_coords.flatten()

# Créer un DataFrame pour stocker toutes les informations
tsne_data = pd.DataFrame({
    'tsne_x': latent_2d[:, 0],
    'tsne_y': latent_2d[:, 1],
    'image_x': image_x_coords_flat,
    'image_y': image_y_coords_flat
})

# Enregistrer le DataFrame dans un fichier CSV
tsne_data.to_csv(output_tsne_filename, index=False)
print(f"\nRésultats t-SNE et coordonnées d'image enregistrés dans '{output_tsne_filename}'")
print(f"Forme des données exportées : {tsne_data.shape}")