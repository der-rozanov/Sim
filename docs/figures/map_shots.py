# -*- coding: utf-8 -*-
"""
Иллюстрации карт для docs/USER_GUIDE.md: вид сверху и 3D-кадр с высоты.

    python docs/figures/map_shots.py              # все карты
    python docs/figures/map_shots.py alps marsh   # выбранные

Результат: docs/img/maptop_<карта>.jpg (600 × 600) и docs/img/map3d_<карта>.jpg.
Каждая карта — отдельный процесс (окно Ursina одно на процесс).
"""

import os
import sys
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG = os.path.join(ROOT, "docs", "img")

# точка камеры (X, Y, Z) и куда смотрит, м (X — восток, Y — вверх, Z — север)
VIEWS = {
    "default": (2500, 900, -2500, -800, 0, 2500),
    "valley":  (-2600, 700, -3800, -800, 100, 1500),
    "alps":    (0, 3000, -7500, 300, 500, 2000),
    "marsh":   (-1200, 90, -500, -3200, 0, 1500),
}


def shoot(name):
    sys.path.insert(0, ROOT)
    from PIL import Image
    from panda3d.core import Filename
    from ursina import camera, scene, Vec3, application
    from viz.viewer3d import make_app

    app, world = make_app("map shots", name, size=(1600, 900))
    world.top_image(1200).resize((600, 600), Image.LANCZOS).save(
        os.path.join(IMG, f"maptop_{name}.jpg"), quality=88)
    scene.fog_density = (0.0, 30000.0)
    x, y, z, lx, ly, lz = VIEWS[name]
    world.follow((x, y, z))
    camera.position = Vec3(x, y, z)
    camera.look_at(Vec3(lx, ly, lz))
    png = os.path.join(IMG, f"_map3d_{name}.png")
    frames = [0]

    def update():                                    # несколько кадров — сцена прогрузилась
        frames[0] += 1
        if frames[0] == 20:
            application.base.win.saveScreenshot(Filename.fromOsSpecific(png))
        if frames[0] == 22:
            Image.open(png).convert("RGB").save(os.path.join(IMG, f"map3d_{name}.jpg"), quality=85)
            os.remove(png)
            os._exit(0)

    import __main__
    __main__.update = update
    app.run()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--one"]:                   # дочерний процесс: одна карта
        shoot(sys.argv[2])
    else:
        for name in sys.argv[1:] or VIEWS:
            subprocess.run([sys.executable, __file__, "--one", name], cwd=ROOT)
            print(f"{name}: docs/img/maptop_{name}.jpg, docs/img/map3d_{name}.jpg")
