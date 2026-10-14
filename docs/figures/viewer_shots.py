# -*- coding: utf-8 -*-
"""
Снимки окна 2D-просмотра логов (viz/viewer.py) для документации.

    python docs/figures/viewer_shots.py

Прогоняет сценарии без окон (MPLBACKEND=Agg), берёт свежий .flightlog каждого и
сохраняет кадр анимации viewer в docs/img/viewer_<имя>.png.
"""
import glob
import os
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")

from flight_logger import load_log
from viz.viewer import FlightLogViewer

# имя снимка: (сценарий, шаблон имени лога, доля прогона для кадра анимации)
SHOTS = {
    "s4": ("scenarios/s4_stall.py", "*Управляемый_срыв*", 0.40),
    "s8": ("scenarios/s8_coordinated_turn.py", "*Координированный_разворот*", 0.45),
}


def newest(pattern):
    files = glob.glob(os.path.join(ROOT, "results", pattern + ".flightlog"))
    return max(files, key=os.path.getmtime)


def main():
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    for name, (script, pattern, frac) in SHOTS.items():
        subprocess.run([sys.executable, os.path.join(ROOT, script)], cwd=ROOT, env=env,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        v = FlightLogViewer(load_log(newest(pattern)), anim_speed=2.0)
        fig = v.build(animated=True)
        v.goto(frac * v.t[-1])
        out = os.path.join(ROOT, "docs", "img", f"viewer_{name}.png")
        fig.savefig(out, dpi=90)
        print("сохранено", out)


if __name__ == "__main__":
    main()
