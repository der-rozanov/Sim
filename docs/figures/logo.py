"""Логотип: α одним штрихом, хвост — траектория набора, на конце — стреловидное крыло.

python docs/figures/logo.py  →  docs/img/logo.png (README), viz/icon.png, viz/icon.ico (окна)
"""
import io
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TILE, EDGE, INK, WING = "#FFFFFF", "#888888", "#111111", "#E5262A"

# узловая кубика x = t² − 1, y = t(t² − 1): петля слева, два хвоста вправо — буква α
t = np.linspace(-1.3, 1.42, 600)
x, y = t**2 - 1, 1.05 * t * (t**2 - 1)
# ширина штриха: толще в петле, сужение к крылу
w = 0.085 + 0.06 * np.exp(-((t + 0.15) / 0.9) ** 2)
w[t > 1.0] *= np.interp(t[t > 1.0], [1.0, 1.42], [1.0, 0.75])
dx, dy = np.gradient(x), np.gradient(y)
n = np.hypot(dx, dy)
nx, ny = -dy / n, dx / n
stroke = np.r_[np.c_[x + w * nx, y + w * ny], np.c_[x - w * nx, y - w * ny][::-1]]

# стреловидное крыло (вид сверху) по касательной к концу хвоста; (вдоль, поперёк)
u = np.array([dx[-1], dy[-1]]) / n[-1]
half = [(0.30, 0), (0.13, 0.045), (-0.13, 0.43), (-0.23, 0.43), (-0.03, 0.075), (-0.15, 0)]
c0 = np.array([x[-1], y[-1]]) + 0.06 * u
wing = np.array([c0 + f * u + s * np.array([-u[1], u[0]])
                 for f, s in half + [(f, -s) for f, s in half[-2:0:-1]]])

# вписать знак в квадрат ±0.87
pts = np.vstack([stroke, wing])
c = (pts.max(0) + pts.min(0)) / 2
k = 1.75 / (pts.max(0) - pts.min(0)).max()
stroke, wing = (stroke - c) * k, (wing - c) * k


def render(px=512) -> Image.Image:
    fig = plt.figure(figsize=(4, 4), dpi=px / 4)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off"); ax.set_aspect("equal")
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05)
    ax.add_patch(FancyBboxPatch((-1, -1), 2, 2, boxstyle="round,pad=0,rounding_size=0.36",
                                fc=TILE, ec=EDGE, lw=1.5))
    ax.add_patch(Polygon(stroke, fc=INK, ec=INK, lw=1))
    ax.add_patch(Polygon(wing, fc=WING, ec=TILE, lw=3, joinstyle="round"))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


if __name__ == "__main__":
    img = render()
    img.save(os.path.join(ROOT, "docs", "img", "logo.png"))
    img.resize((256, 256), Image.LANCZOS).save(os.path.join(ROOT, "viz", "icon.png"))
    img.save(os.path.join(ROOT, "viz", "icon.ico"),
             sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("docs/img/logo.png, viz/icon.png, viz/icon.ico")
