from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import *
import math
import re
import os


# Création manuelle de l'application
base = ShowBase()

# Désactiver le contrôle souris de la caméra
base.disableMouse()

# Charger un modèle de base fourni avec Panda3D
#cube = base.loader.loadModel("models/box")

cube = base.loader.loadModel("textures/sol")

cube.reparentTo(base.render)
cube.setPos(0, 0, 0)

# Positionner la caméra
base.camera.setPos(0, -10, 15) # positions sous la forme (x, y, z) donc 20 unités en arrière, et 5 unités vers le haut
# base.camera.setPos(0, 0, 2) # a activer si on prend des photos
base.camera.lookAt(0, 0, 0)

# Create Ambient Light 
# Lumière diffuse sans direction ni ombres
ambientLight = AmbientLight('ambientLight')
ambientLight.setColor((0.3, 0.3, 0.3, 1)) # (R, G, B, A/transparance)
ambientLightNP = render.attachNewNode(ambientLight)
render.setLight(ambientLightNP)


# Source de lumiere ponctuelle (Point light)
# Lumière dans toutes les directions avec atténuations sur la distance
lumierePoint = PointLight('LumierePoint')
lumierePoint.setColor(Vec4(0.8, 0.8, 0.8, 1)) #Vec4 permet de le mettre sous forme d'un vecteur taille 4 (pour opérations)
lumierePointNP = render.attachNewNode(lumierePoint) # Permet d'ajouter le NodePath a la scène (pour pouvoir l'utiliser)
lumierePointNP.setPos(3, 0, 0)
# eclairer tout le graphe de scene
render.setLight(lumierePointNP) # Active la lumière pour affecter les autres objets (cube.setLight(...) n'affecterait que le cube )


# Charger une sphère pour visualiser la position de la lumière
sphere = loader.loadModel("models/smiley")  # ou "models/ball" si smiley indisponible
sphere.reparentTo(lumierePointNP) # On colle le smiley à la lumière pour visualiser sa position
sphere.setScale(0.3)

# Fonciton pour visualiser le positionnement des lumières
def visualiser_positions_lumiere(fichier="infos_angles_lum.txt", r=3):
    sphere.hide()
    with open(fichier, "r") as f:
        #print(f"Vous avez importé {len(f.readlines())} angles de lumière") # N'affiche pas les lumières si on print
        for line in f:
            match = re.search(r"tl=(\d+)\s+pl=(\d+)", line)
            if match:
                tl, pl = map(int, match.groups())
                tl = math.radians(tl)
                pl = math.radians(pl)

                x = r * math.cos(pl) * math.sin(tl)
                y = r * math.sin(pl) * math.sin(tl)
                z = r * math.cos(tl)

                marker = loader.loadModel("models/smiley")
                marker.reparentTo(render)
                marker.setScale(0.1)
                marker.setPos(x, y, z)
                marker.setColor(1, 0, 0, 1)
            else:
                print(f"Ligne ignorée : {line.strip()}")


# callback
def animer2(task):
    lum_r = 3
    photo = 1

    if not hasattr(task, 'pairs'):
        # Charger les angles caméras
        with open("Angles/infos_angles_cam.txt", "r") as f:
            camera_angles = [
                tuple(map(int, re.search(r"tv=(\d+)\s+pv=(\d+)", l).groups()))
                for l in f if "tv=" in l
            ]

        # Charger les angles lumières
        with open("Angles/infos_angles_lum.txt", "r") as f:
            light_angles = [
                tuple(map(int, re.search(r"tl=(\d+)\s+pl=(\d+)", l).groups()))
                for l in f if "tl=" in l
            ]

        # Créer toutes les paires (caméra, lumière)
        task.pairs = [(cam, light) for cam in camera_angles for light in light_angles]
        print(f"Cette configuration contient {len(light_angles)} angles de lumière et {len(camera_angles)} angles de caméra, soit {len(task.pairs)} images au total")
        task.index = 0
        task.last_update_time = 0

    if task.index >= len(task.pairs):
        return Task.done

    if task.time - task.last_update_time >= 0.3:
        task.last_update_time = task.time

        (tv, pv), (tl, pl) = task.pairs[task.index]

        # Caméra
        tv_rad = math.radians(tv)
        pv_rad = math.radians(pv)
        cam_r = 2
        cam_x = cam_r * math.cos(pv_rad) * math.sin(tv_rad)
        cam_y = cam_r * math.sin(pv_rad) * math.sin(tv_rad)
        cam_z = cam_r * math.cos(tv_rad)
        base.camera.setPos(cam_x, cam_y, cam_z)
        base.camera.lookAt(0, 0, 0)

        # Lumière
        tl_rad = math.radians(tl)
        pl_rad = math.radians(pl)
        x = lum_r * math.cos(pl_rad) * math.sin(tl_rad)
        y = lum_r * math.sin(pl_rad) * math.sin(tl_rad)
        z = lum_r * math.cos(tl_rad)
        lumierePointNP.setPos(x, y, z)

        print(f"Caméra: tv={tv}, pv={pv} | Lumière: tl={tl}, pl={pl}")

        if photo:
            sphere.hide()
            os.makedirs("captures", exist_ok=True)
            filename = f"captures/img_tv{tv}_pv{pv}_tl{tl}_pl{pl}.png"
            base.win.saveScreenshot(Filename(filename))

        task.index += 1

    return Task.cont


base.setFrameRateMeter(True) # On est constant autour de 60 frames/s
#visualiser_positions_lumiere()
taskMgr.add(animer2, "animer")

# Lancer la boucle principale
base.run()