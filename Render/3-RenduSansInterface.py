from direct.showbase.ShowBase import ShowBase
from panda3d.core import *
import math
import re
import os

# Initialiser Panda sans boucle principale
loadPrcFileData("", "window-type offscreen")
base = ShowBase()

# Désactiver le contrôle souris
base.disableMouse()

# Charger le modèle
cube = base.loader.loadModel("textures/bleu")
cube.reparentTo(base.render)
cube.setPos(0, 0, 0)

# Lumière ambiante
ambientLight = AmbientLight('ambientLight')
ambientLight.setColor((0.3, 0.3, 0.3, 1))
ambientLightNP = base.render.attachNewNode(ambientLight)
base.render.setLight(ambientLightNP)

# Lumière ponctuelle
lumierePoint = PointLight('LumierePoint')
lumierePoint.setColor(Vec4(0.8, 0.8, 0.8, 1))
lumierePointNP = base.render.attachNewNode(lumierePoint)
base.render.setLight(lumierePointNP)

# Charger les angles caméra
with open("Angles/infos_angles_cam.txt", "r") as f:
    camera_angles = [
        tuple(map(int, re.search(r"tv=(\d+)\s+pv=(\d+)", l).groups()))
        for l in f if "tv=" in l
    ]

# Charger les angles lumière
with open("Angles/infos_angles_lum.txt", "r") as f:
    light_angles = [
        tuple(map(int, re.search(r"tl=(\d+)\s+pl=(\d+)", l).groups()))
        for l in f if "tl=" in l
    ]

pairs = [(cam, light) for cam in camera_angles for light in light_angles]
print(f"Cette configuration contient {len(light_angles)} angles de lumière et {len(camera_angles)} angles de caméra, soit {len(pairs)} images au total")

lum_r = 3
cam_r = 1.5
os.makedirs("captures", exist_ok=True)

for idx, ((tv, pv), (tl, pl)) in enumerate(pairs):
    # Position caméra
    tv_rad = math.radians(tv)
    pv_rad = math.radians(pv)
    cam_x = cam_r * math.cos(pv_rad) * math.sin(tv_rad)
    cam_y = cam_r * math.sin(pv_rad) * math.sin(tv_rad)
    cam_z = cam_r * math.cos(tv_rad)
    base.camera.setPos(cam_x, cam_y, cam_z)
    base.camera.lookAt(0, 0, 0)

    # Position lumière
    tl_rad = math.radians(tl)
    pl_rad = math.radians(pl)
    x = lum_r * math.cos(pl_rad) * math.sin(tl_rad)
    y = lum_r * math.sin(pl_rad) * math.sin(tl_rad)
    z = lum_r * math.cos(tl_rad)
    lumierePointNP.setPos(x, y, z)

    # Enregistrer l'image
    filename = f"captures/img_tv{tv}_pv{pv}_tl{tl}_pl{pl}.jpg"
    base.graphicsEngine.renderFrame()  # Nécessaire pour forcer le rendu
    base.win.saveScreenshot(Filename(filename))
    print(f"[{idx+1}/{len(pairs)}] Sauvegarde: {filename}")

print("Terminé !")