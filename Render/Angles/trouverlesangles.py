import re
import os

# Dossier contenant les images
repertoire = "../Bonn_Dataset/tv000_pv000"

# Expression pour extraire les infos du nom de fichier
pattern = re.compile(r"(\d+)\s+tl([\d.-]+)\s+pl([\d.-]+)\s+tv([\d.-]+)\s+pv([\d.-]+)\.jpg")
#pattern = re.compile(r"tv(\d+)_pv(\d+)")

# Stockage des résultats
images_info = []


for nom_fichier in os.listdir(repertoire):
    if nom_fichier.endswith(".jpg"):
        match = pattern.search(nom_fichier)
        if match:
            i, tl, pl, tv, pv = match.groups()
            print(f"tl = {tl}, pl = {pl}")
            images_info.append((int(tl), int(pl)))
        else:
            print(f"Nom de fichier ignoré : {nom_fichier}")

# Trier les angles de caméra
images_info.sort(key=lambda x: (x[0], x[1]))  # tri par tv, puis pv

# Export dans un fichier texte
with open("infos_angles_lum.txt", "w") as f:
    for tl, pl in images_info:
        f.write(f"angles_lum tl={tl} pl={pl}\n")

print(f"{len(images_info)} fichiers traités et exportés.")