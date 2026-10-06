# -*- coding: utf-8 -*-
"""
Подготовка реальной карты для 3D-тренажёра: спутниковый снимок + рельеф.

Источники (скачиваются один раз в кэш viz/map_cache/<имя>/, в git не попадают):
  снимок  — Esri World Imagery (тайлы Web Mercator, ~0.7 м/пикс на уровне 17).
            © Esri, Maxar, Earthstar Geographics и сообщество ГИС. Условия Esri
            не разрешают распространять тайлы — поэтому только локальный кэш.
  рельеф  — AWS Terrain Tiles (Mapzen terrarium, открытые данные; SRTM ~30 м).
            Высота: h = R·256 + G + B/256 − 32768, м.

Результат — квадрат size × size м с центром (lat0, lon0), «север вверх»:
  ortho.jpg  — снимок N×N пикс (строка 0 — север, столбец 0 — запад);
  dem.npy    — высоты на той же сетке (M×M), м над уровнем моря;
  meta.json  — центр, размер, источники.

Локальные координаты — касательная плоскость в центре (сфера R = 6 371 км):
  N = R·(lat − lat0),  E = R·cos(lat0)·(lon − lon0)   [рад → м].
Ошибка на 2 км от центра ~ 1e−4 относительно — для тренажёра пренебрежимо.
"""

import os
import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

R_EARTH = 6371008.8
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_cache")
IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TERRAIN = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
UA = {"User-Agent": "UAV-sim-diploma/1.0 (local cache)"}


def _merc_px(lat, lon, z):
    """Глобальные пиксельные координаты Web Mercator (тайл 256)."""
    n = 256.0 * 2 ** z
    x = (np.asarray(lon) + 180.0) / 360.0 * n
    la = np.radians(lat)
    y = (1.0 - np.log(np.tan(la) + 1.0 / np.cos(la)) / np.pi) / 2.0 * n
    return x, y


def _fetch(url, retries=5):
    """Скачать тайл; при сетевом сбое (таймаут, обрыв) — повтор с паузой 2, 4, 8… с."""
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries - 1:
                raise
            print(f"[map] повтор {attempt + 1}/{retries - 1}: {url} ({e})")
            time.sleep(2 ** (attempt + 1))


def _mosaic(url_tpl, z, x0, x1, y0, y1):
    """Сшить тайлы [x0..x1]×[y0..y1] в один массив (H, W, 3) uint8."""
    keys = [(tx, ty) for ty in range(y0, y1 + 1) for tx in range(x0, x1 + 1)]
    with ThreadPoolExecutor(8) as ex:
        imgs = list(ex.map(lambda k: _fetch(url_tpl.format(z=z, x=k[0], y=k[1])), keys))
    out = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256, 3), np.uint8)
    for (tx, ty), im in zip(keys, imgs):
        out[(ty - y0) * 256:(ty - y0 + 1) * 256, (tx - x0) * 256:(tx - x0 + 1) * 256] = np.asarray(im)
    return out


def _sample(mosaic, px, py):
    """Билинейная выборка mosaic (H, W, C) в точках px, py (float, пиксели мозаики)."""
    H, W = mosaic.shape[:2]
    px = np.clip(px, 0, W - 1.001); py = np.clip(py, 0, H - 1.001)
    x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
    fx, fy = (px - x0)[..., None], (py - y0)[..., None]
    m = mosaic.astype(np.float32)
    top = m[y0, x0] * (1 - fx) + m[y0, x0 + 1] * fx
    bot = m[y0 + 1, x0] * (1 - fx) + m[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bot * fy


def _local_grid(lat0, lon0, size, n):
    """lat, lon центров пикселей сетки n×n (строка 0 — север)."""
    s = (np.arange(n) + 0.5) / n * size - size / 2
    E, N = np.meshgrid(s, -s)                         # N убывает вниз по строкам
    lat = lat0 + np.degrees(N / R_EARTH)
    lon = lon0 + np.degrees(E / (R_EARTH * np.cos(np.radians(lat0))))
    return lat, lon


def _resample(url_tpl, z, lat0, lon0, size, n, margin=1):
    lat, lon = _local_grid(lat0, lon0, size, n)
    gx, gy = _merc_px(lat, lon, z)
    x0, x1 = int(gx.min() // 256) - margin, int(gx.max() // 256) + margin
    y0, y1 = int(gy.min() // 256) - margin, int(gy.max() // 256) + margin
    mos = _mosaic(url_tpl, z, x0, x1, y0, y1)
    return _sample(mos, gx - x0 * 256, gy - y0 * 256), (x1 - x0 + 1) * (y1 - y0 + 1)


def clean_dem(dem, win=5, tol=25.0):
    """
    Убрать выбросы рельефа: в тайлах terrarium дыры данных — значения ~ −1500 м
    (на швах тайлов попадают в интерполяцию). Точка, отличающаяся от медианы
    окна win×win больше чем на tol м, заменяется медианой.
    """
    p = win // 2
    padded = np.pad(dem, p, mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(padded, (win, win)), axis=(-1, -2))
    bad = np.abs(dem - med) > tol
    out = dem.copy()
    out[bad] = med[bad]
    return out, int(bad.sum())


def prepare(name, lat0, lon0, size=4000.0, ortho_px=4096, img_zoom=17,
            dem_px=401, dem_zoom=14, force=False):
    """Скачать и подготовить карту (если ещё нет в кэше). Возвращает папку кэша."""
    d = os.path.join(CACHE, name)
    if not force and all(os.path.exists(os.path.join(d, f))
                         for f in ("ortho.jpg", "dem.npy", "meta.json")):
        return d
    os.makedirs(d, exist_ok=True)
    print(f"[map] {name}: загрузка снимка (уровень {img_zoom}) и рельефа (уровень {dem_zoom})…")
    rgb, n_img = _resample(IMAGERY, img_zoom, lat0, lon0, size, ortho_px)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(os.path.join(d, "ortho.jpg"), quality=92)
    t, n_dem = _resample(TERRAIN, dem_zoom, lat0, lon0, size, dem_px)
    dem = t[..., 0] * 256.0 + t[..., 1] + t[..., 2] / 256.0 - 32768.0
    dem, n_bad = clean_dem(dem)
    if n_bad:
        print(f"[map] рельеф: исправлено выбросов — {n_bad}")
    np.save(os.path.join(d, "dem.npy"), dem.astype(np.float32))
    meta = dict(name=name, lat0=lat0, lon0=lon0, size=size, ortho_px=ortho_px,
                dem_px=dem_px, img_zoom=img_zoom, dem_zoom=dem_zoom,
                tiles=dict(imagery=n_img, terrain=n_dem),
                sources=dict(imagery="Esri World Imagery (© Esri, Maxar, Earthstar Geographics)",
                             terrain="AWS Terrain Tiles / Mapzen terrarium (SRTM)"))
    json.dump(meta, open(os.path.join(d, "meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[map] готово: {n_img} тайлов снимка, {n_dem} тайлов рельефа → {d}")
    return d


if __name__ == "__main__":
    prepare("kainki", 55.644685, 48.507016)   # центр — стык полос «Т»
