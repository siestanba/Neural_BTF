import tkinter as tk
from tkinter import simpledialog, filedialog
from PIL import Image, ImageTk
import pandas as pd
import sys
import os
import re
import csv


# Constantes globales
SCALE_FACTOR = 3 # l'image affichée pour la sélection des pixels sera 3 fois plus grande
sampling_step = 1 # si = 1 on sélectionne tous les pixels dans le rectangle

# Variables globales
selected_pixels_set = set() # Stocke les coordonnées de pixels uniques (x, y) en tuples
start_x_canvas = None # Clic souris X
start_y_canvas = None # Clic souris Y
current_rect_id = None # To store the ID of the temporary transparent rectangle

original_image_width = None
original_image_height = None
display_image_width = None
display_image_height = None
total_pixels_in_original_image = 0 # Pour ne pas dépasser le nombre de pixels de l'image originale


def add_pixels_in_rectangle(x1_canvas, y1_canvas, x2_canvas, y2_canvas):
    global selected_pixels_set

    # Convertion des coordonnées de l'image affichée à l'image originale
    x1_orig = int(x1_canvas / SCALE_FACTOR)
    y1_orig = int(y1_canvas / SCALE_FACTOR)
    x2_orig = int(x2_canvas / SCALE_FACTOR)
    y2_orig = int(y2_canvas / SCALE_FACTOR)

    # Vérifier que le pixel est dans l'image
    min_x_orig = max(0, min(x1_orig, x2_orig))
    max_x_orig = min(original_image_width - 1, max(x1_orig, x2_orig))
    min_y_orig = max(0, min(y1_orig, y2_orig))
    max_y_orig = min(original_image_height - 1, max(y1_orig, y2_orig))

    pixels_added_this_selection = 0

    # Itérations sur les coordonées de l'image originale
    for y_orig in range(int(min_y_orig), int(max_y_orig) + 1, sampling_step):
        for x_orig in range(int(min_x_orig), int(max_x_orig) + 1, sampling_step):
            if len(selected_pixels_set) >= total_pixels_in_original_image:
                print(f"Nombre maximal de pixels ({total_pixels_in_original_image}) atteint.")
                return

            pixel_coord_orig = (x_orig, y_orig)
            if pixel_coord_orig not in selected_pixels_set:
                selected_pixels_set.add(pixel_coord_orig)
                pixels_added_this_selection += 1
                
                # On colorie les pixels sélectionnés
                display_x = x_orig * SCALE_FACTOR
                display_y = y_orig * SCALE_FACTOR
                canvas.create_rectangle(display_x, display_y, 
                                        display_x + SCALE_FACTOR, display_y + SCALE_FACTOR,
                                        fill="red", outline="red", tags="selected_pixel")
    
    if pixels_added_this_selection > 0:
        print(f"Vous avec ajouté {pixels_added_this_selection} nouveaux pixels uniques. Sélection totale: {len(selected_pixels_set)}")


# Events
def on_mouse_press(event):
    global start_x_canvas, start_y_canvas, current_rect_id
    start_x_canvas = event.x
    start_y_canvas = event.y
    if current_rect_id:
        canvas.delete(current_rect_id)
    current_rect_id = None


def on_mouse_drag(event):
    global current_rect_id
    if start_x_canvas is None or start_y_canvas is None:
        return

    if current_rect_id:
        canvas.delete(current_rect_id)

    current_rect_id = canvas.create_rectangle(start_x_canvas, start_y_canvas, event.x, event.y,
                                               outline="red", fill="red",
                                               stipple="gray25", tags="temp_rect")


def on_mouse_release(event):
    global start_x_canvas, start_y_canvas, current_rect_id
    
    if current_rect_id:
        canvas.delete(current_rect_id)
        current_rect_id = None

    if start_x_canvas is not None and start_y_canvas is not None:
        x1_canvas, y1_canvas = start_x_canvas, start_y_canvas
        x2_canvas, y2_canvas = event.x, event.y

        add_pixels_in_rectangle(x1_canvas, y1_canvas, x2_canvas, y2_canvas)
        
        start_x_canvas = None
        start_y_canvas = None
    
    print(f"Selection terminée. Nombre total de pixels uniques sélectionnés: {len(selected_pixels_set)}.")


def clear_selection():
    global selected_pixels_set
    selected_pixels_set.clear()
    canvas.delete("selected_pixel")
    canvas.delete("temp_rect") 
    print("Selection effacée.")

def change_sampling_step():
    global sampling_step
    new_step = simpledialog.askinteger("Pas de l'échantillonnage", 
                                      f"Enter un pas  pour l'échantillonnage (actuellement: {sampling_step}).\n"
                                      f"Une taille de 1 sélectionne chaque pixel de l'image ORIGINALE. Une taille de N sélectionne chaque Nième pixel..",
                                      parent=root, minvalue=1, maxvalue=50)
    if new_step is not None:
        sampling_step = new_step
        print(f"Nouveau pas d'échantillonnage: {sampling_step}.")

# Sauvegarder les coordonnées au format csv
def save_selection_to_csv(output_filename):
    if not selected_pixels_set:
        print("Aucun pixel n'a été sélectionné.")
        return

    try:
        # Créer un DataFrame avec les coordonnées des pixels sélectionnés
        # Les pixels sont déjà stockés sous forme de (x, y)
        df_selected_pixels = pd.DataFrame(list(selected_pixels_set), columns=['x', 'y'])

        # Enregistrer le DataFrame dans le fichier CSV
        df_selected_pixels.to_csv(output_filename, index=False)
        print(f"{len(selected_pixels_set)} pixels ont été sauvegardés dans {output_filename}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")

# =========================================================
# Importation des images
# =========================================================

def get_images(chemin):
    pattern = re.compile(r"(\d+)\s+tl([\d.-]+)\s+pl([\d.-]+)\s+tv([\d.-]+)\s+pv([\d.-]+)\.jpg")

    if not os.path.exists(chemin):
        print(f"Le dossier spécifié n'existe pas : {chemin}")

    for nom_fichier in os.listdir(chemin):
        if nom_fichier.endswith(".jpg"):
            match = pattern.match(nom_fichier)
            if match:
                chemin_image = os.path.join(chemin, nom_fichier)
                original_full_image = Image.open(chemin_image)
                
                orig_w = original_full_image.width
                orig_h = original_full_image.height
                
                disp_w = orig_w * SCALE_FACTOR
                disp_h = orig_h * SCALE_FACTOR

                img_for_display = original_full_image.resize(
                    (disp_w, disp_h), Image.Resampling.NEAREST
                )
                
                total_pix = orig_w * orig_h
                print(f"Dimensions de l'image originale: {orig_w}x{orig_h}. Nombre Max de pixels: {total_pixels_in_original_image}")
                print(f"Dimensions de l'image affichée (facteur {SCALE_FACTOR}x): {disp_w}x{disp_h}")
                return img_for_display, orig_w, orig_h, disp_w, disp_h, total_pix # Return defaults if no image is found
            else:
                print(f"Nom de fichier ignoré : {nom_fichier}")

# GUI Setup
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Utilisation: python GUI.py <chemin_vers_les_images> <nom_du_fichier_de_sortie.csv>")
        print("Exemple: python programme.py /Users/YourUser/Documents/MesSuperImages mes_pixels_selectionnes.csv")
        sys.exit(1)

    path_to_image = sys.argv[1]
    output_csv_filename = sys.argv[2] # Récupération du nom du fichier de sortie

    root = tk.Tk()
    root.title(f"Pixel Selector (Original: {original_image_width}x{original_image_height}, Affichage: {display_image_width}x{display_image_height}) - Max : {total_pixels_in_original_image} pixels")

    image_for_display, original_image_width, original_image_height, display_image_width, display_image_height, total_pixels_in_original_image = get_images(path_to_image)
    tk_image = ImageTk.PhotoImage(image_for_display)

    canvas = tk.Canvas(root, width=display_image_width, height=display_image_height, bg="lightgray")
    canvas.pack()

    canvas.create_image(0, 0, anchor=tk.NW, image=tk_image)

    canvas.bind("<Button-1>", on_mouse_press)
    canvas.bind("<B1-Motion>", on_mouse_drag)
    canvas.bind("<ButtonRelease-1>", on_mouse_release)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)

    clear_button = tk.Button(button_frame, text="Effacer", command=clear_selection)
    clear_button.pack(side=tk.LEFT, padx=5)

    change_sampling_button = tk.Button(button_frame, text="Pas de l'échantillonnage", command=change_sampling_step)
    change_sampling_button.pack(side=tk.LEFT, padx=5)

    save_button = tk.Button(button_frame, text="Sauvegarder", command=lambda: save_selection_to_csv(output_csv_filename))
    save_button.pack(side=tk.LEFT, padx=5)

    root.mainloop()

    final_x_coords = [p[0] for p in selected_pixels_set]
    final_y_coords = [p[1] for p in selected_pixels_set]

    print(f"\nCoordonnées uniques en X (IMAGE ORIGINALE): {final_x_coords}")
    print(f"\nCoordonnées uniques en Y (IMAGE ORIGINALE): {final_y_coords}")
    print(f"Nombre total de pixels sélectionnés (IMAGE ORIGINALE): {len(selected_pixels_set)}")
