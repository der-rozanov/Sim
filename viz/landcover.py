# -*- coding: utf-8 -*-
"""
Карта типов поверхности по спутниковому снимку (для «мультяшной» карты).

Каждый пиксель снимка (уменьшенного до n × n) описывается 5 признаками:
  r, g, b / V   — цветность (V — средняя яркость RGB + 1);
  ln V / 1.5    — яркость;
  ln(1 + σ) / 3 — текстура: СКО яркости в окне 7 × 7 на удвоенном разрешении
                  (лес «рябой», луг и вода гладкие).
Класс — ближайший эталон из PROTOTYPES (евклидово расстояние). Эталоны — центры
кластеров k-means (K = 10) по снимку Каинок 10 × 10 км; классы им присвоены
вручную по виду на снимке. Затем — мажоритарное сглаживание (убирает «соль»).

Это раскраска для тренажёра, а не тематическая карта: дымка/облака уходят
в луг, тёмная трава иногда — в воду (воду дальше уточняет OSM).
Результат кэшируется рядом со снимком: landcover.png (значение пикселя — класс).
"""

import os

import numpy as np
from PIL import Image, ImageFilter

WATER, FOREST, SCRUB, FIELD, MEADOW, GRASS = range(6)
CLASSES = ("water", "forest", "scrub", "field", "meadow", "grass")

# (r/V, g/V, b/V, lnV/1.5, ln(1+σ)/3) → класс
PROTOTYPES = [
    ((0.873, 1.205, 0.864, 2.633, 0.298), GRASS),    # тёмная гладкая трава (и русло в дымке)
    ((0.979, 1.179, 0.799, 2.840, 0.621), MEADOW),
    ((0.837, 1.325, 0.783, 2.672, 0.583), GRASS),
    ((1.090, 1.040, 0.839, 3.045, 0.466), FIELD),    # бурая пашня
    ((0.703, 1.496, 0.722, 2.436, 0.832), FOREST),   # тёмный «рябой» лес
    ((0.852, 1.304, 0.789, 2.674, 0.844), SCRUB),
    ((0.996, 1.136, 0.827, 2.877, 0.944), MEADOW),   # луг с кустами
    ((0.624, 1.263, 0.970, 2.075, 0.380), WATER),    # старицы, озёра
    ((1.004, 1.162, 0.794, 2.878, 0.400), MEADOW),
    ((1.074, 1.064, 0.830, 3.033, 0.727), FIELD),    # пашня с бороздами
]
DARK_SMOOTH = (0, 7)                                 # эталоны «похоже на воду» (для проверки OSM)


def _box(x, r):
    """Скользящее среднее (2r+1)×(2r+1) через кумулятивные суммы."""
    p = np.pad(x, ((r + 1, r), (r + 1, r)), mode="edge").cumsum(0).cumsum(1)
    k = 2 * r + 1
    return (p[k:, k:] - p[:-k, k:] - p[k:, :-k] + p[:-k, :-k]) / (k * k)


def features(ortho, n):
    """Снимок (PIL) → признаки (n, n, 5)."""
    a = np.asarray(ortho.resize((n, n), Image.BOX).filter(ImageFilter.GaussianBlur(1.5)), float)
    lum = np.asarray(ortho.convert("L").resize((2 * n, 2 * n), Image.BOX), float)
    m1, m2 = _box(lum, 3), _box(lum * lum, 3)
    sd = np.sqrt(np.maximum(m2 - m1 * m1, 0)).astype(np.float32)
    sd = np.asarray(Image.fromarray(sd, "F").resize((n, n), Image.BOX))
    V = a.mean(2) + 1
    return np.stack([a[..., 0] / V, a[..., 1] / V, a[..., 2] / V,
                     np.log(V) / 1.5, np.log1p(sd) / 3], -1)


def classify(ortho, n=2048):
    """
    Снимок → (класс (n, n) uint8, номер эталона (n, n) uint8), со сглаживанием.
    """
    f = features(ortho, n).reshape(-1, 5)
    P = np.array([p for p, _ in PROTOTYPES])
    proto = np.concatenate([np.argmin(((f[i:i + 400000, None] - P[None]) ** 2).sum(-1), 1)
                            for i in range(0, len(f), 400000)]).reshape(n, n).astype(np.uint8)
    cls = np.array([c for _, c in PROTOTYPES], np.uint8)[proto]
    img = Image.fromarray(cls, "L")
    for k in (5, 5, 7):
        img = img.filter(ImageFilter.ModeFilter(k))
    return np.asarray(img).copy(), proto


def load(cache_dir, n=2048):
    """Классы и маска «тёмное гладкое» из кэша (или посчитать и сохранить)."""
    path = os.path.join(cache_dir, "landcover.png")
    if os.path.exists(path):
        a = np.asarray(Image.open(path))
        return a[..., 0].copy(), a[..., 1] > 0
    print("[landcover] классификация снимка…")
    cls, proto = classify(Image.open(os.path.join(cache_dir, "ortho.jpg")), n)
    wet = Image.fromarray((np.isin(proto, DARK_SMOOTH) * 255).astype(np.uint8)).filter(ImageFilter.ModeFilter(7))
    wet = np.asarray(wet)
    Image.fromarray(np.stack([cls, wet, wet], -1)).save(path)
    return cls, wet > 0
