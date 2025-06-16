from keras.layers import Conv1D, Input, Flatten, Dense, MaxPooling1D, BatchNormalization, Concatenate
from keras.optimizers import Adam
from keras.models import Model
import tensorflow as tf
from PIL import Image
import pandas as pd
import copy
import numpy as np
import keras
import os
import re

# Fonction pour calculer les coordonées stéréographiques

def stereo(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Projection sur x y
    px = x / (1 + z + 1e-8)
    py = y / (1 + z + 1e-8)
    return px, py

# Dossier contenant les images
#repertoire = "../Render/amb_01_diff_0_spec_1"
repertoire = "../Render/captures_bleu"
img_width, img_height = 128, 128

# Expression pour extraire les infos du nom de fichier
#pattern = re.compile(r"img_tv(\d+)_pv(\d+)_tl(\d+)_pl(\d+)\.png") #all
pattern = re.compile(r"img_tv0_pv0_tl(\d+)_pl(\d+)\.jpg") #les images avec tv = 45 (la latitude moyenne)

# Stockage des résultats
images_info = []

for nom_fichier in os.listdir(repertoire):
    if nom_fichier.endswith(".jpg"):
        match = pattern.match(nom_fichier)
        if match:
            tl, pl = match.groups()
            tl, pl = int(tl), int(pl)
            px, py = stereo(np.radians(tl), np.radians(pl)) # Calcul des coordonnées stéréo
            chemin_image = os.path.join(repertoire, nom_fichier)
            image = Image.open(chemin_image)
            w, h = image.size
            left = (w - img_width) / 2
            top = (h - img_height) / 2
            right = (w + img_width) / 2
            bottom = (h + img_height) / 2
            #left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            image = Image.open(chemin_image).crop((left, top, right, bottom))
            pixels = np.array(image, dtype=np.float32) / 255.0


            images_info.append({
                "tl": tl,
                "pl": pl,
                "px": px,
                "py": py,
                "image": image,
                "pixels": pixels,
            })

images_info.sort(key=lambda d: (d["tl"], d["pl"]))

###
#--- Création du modèle
###

# =================
# Encoder
# =================

latent_dim = 8 # Dimension du bottleneck
num_channels = 3 # RGB
samples = len(images_info) # Nombre de configuration angulaires
matrix_H = 9
matrix_W = 9

input_shape = (samples, num_channels)

# Definition
i = Input(shape=input_shape, name='encoder_input')

# Bloc 1
x = Conv1D(filters = 64, kernel_size=3, padding='same', activation='relu')(i)
x = BatchNormalization()(x) #weight decay
x = MaxPooling1D(pool_size=2)(x)

# Bloc 2
x = Conv1D(filters = 128, kernel_size=3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

# Bloc 3
x = Conv1D(filters = 256, kernel_size=3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)

# Bloc 4
x = Conv1D(filters = 512, kernel_size=3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=2)(x)


# Projection
x = Flatten()(x)
latent_output = Dense(latent_dim, name='latent_output')(x)

# Modèle
encoder = Model(inputs=i, outputs=latent_output, name="Encoder")
encoder.summary()


# =================
# Decoder
# =================

# Paramètres
angle_dim = 2  # 2 pour lumière
input_dim = latent_dim + angle_dim  # 8 + 2 = 10

# Entrée
decoder_input = Input(shape=(input_dim,), name='decoder_input')

# 4 couches Fully Connected avec ReLU
x = Dense(106, activation='relu')(decoder_input)
x = Dense(106, activation='relu')(x)
x = Dense(106, activation='relu')(x)
x = Dense(106, activation='relu')(x)

# Sortie RGB (3 valeurs)
decoder_output = Dense(3, activation='linear', name='rgb_output')(x)

# Modèle
decoder_model = Model(inputs=decoder_input, outputs=decoder_output, name="Decoder")
decoder_model.summary()


# =================
# Modele
# =================
view_light_input = Input(shape=(angle_dim,), name='view_light_input')
latent_vector = encoder(i)
decoder_input = Concatenate()([latent_vector, view_light_input])
model_outputs = decoder_model(decoder_input)

model = Model(inputs=[i, view_light_input], outputs=model_outputs, name='Modele')
model.summary()


# --- L2 Loss ---
def custom_reconstruction_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# on prend toutes les images
image_stack = np.stack([info["pixels"] for info in images_info], axis=0)
# image_stack.shape = (N_images, height, width, 3)

# on fait de même avec les angles
tl_stack = np.stack([info["tl"] for info in images_info], axis=0)
pl_stack = np.stack([info["pl"] for info in images_info], axis=0)

# et pour les positions stéréo
px_stack = np.stack([info["px"] for info in images_info], axis=0)
py_stack = np.stack([info["py"] for info in images_info], axis=0)

# Préparation des données

# On prend toutes les ABRDF par texel
num_pixels = img_height*img_width
image_stack_flat = image_stack.reshape((samples, num_pixels, num_channels))

texel_observations = np.zeros((num_pixels, samples, num_channels))
texel_observations = np.transpose(image_stack_flat, (1, 0, 2)) # Pour avoir les informations par pixel, on inverse juste les axes
print("Nouvelle forme:", texel_observations.shape)

light_dirs = np.stack([px_stack, py_stack], axis=1) # Forme (81, 2)
rep_light_dirs = np.tile(light_dirs, (num_pixels, 1, 1))
target_rgb_train = texel_observations  # (900, 81, 3)

texel_observations.shape, rep_light_dirs.shape, target_rgb_train.shape

reshaped_input = np.repeat(texel_observations, samples, axis=0)
# Nouvelle forme: (num_pixels * samples, 81, 3)

reshaped_light_dirs = rep_light_dirs.reshape(-1, 2)
# Nouvelle forme: (num_pixels * samples, 2)

reshaped_target_rgb = target_rgb_train.reshape(-1, 3)
# Nouvelle forme: (num_pixels * samples, 3)

reshaped_input.shape, reshaped_light_dirs.shape, reshaped_target_rgb.shape


### Entraînement du modoèle

custom_optimizer = Adam(learning_rate=0.001) # Adam est une version améliorée de la descente de gradient stochastique, par défaut on avait 0.001
model.compile(custom_optimizer, loss=lambda y_true, y_pred: custom_reconstruction_loss(y_true, y_pred))

history = model.fit(
    [reshaped_input, reshaped_light_dirs], # Keras gère le batch_size automatiquement
    reshaped_target_rgb,
    epochs=30,
    batch_size=5, # Keras va prendre 5 pixels à la fois et les passer à l'encodeur/décodeur
    validation_split=0.2
)

enco_save_path = os.path.join('modeles/', 'my_enco_bleu.keras')
encoder.save(enco_save_path)

deco_save_path = os.path.join('modeles/', 'my_deco_bleu.keras')
decoder_model.save(deco_save_path)