# -*- coding: utf-8 -*-
"""
Данные OpenStreetMap для «мультяшной» карты 3D-тренажёра.

Источник: Overpass API (© участники OpenStreetMap, лицензия ODbL — данные
можно хранить и распространять с указанием авторства). Выгрузка кэшируется в
viz/map_cache/<имя>/osm.json.

Что берётся (квадрат size × size м с центром lat0, lon0):
  площади: вода (natural=water, water=*, waterway=riverbank), лес (natural=wood,
           landuse=forest), кустарник, болото, жилая застройка, пашня, луг;
  линии:   реки и ручьи (waterway), дороги (highway);
  здания:  контуры building=*.
Мультиполигоны (река с островами) собираются из кусков контура (stitch_rings).

Координаты — те же, что в mapdata.py: касательная плоскость в центре,
E (восток) и N (север), м.
"""

import os
import json
import urllib.request
import urllib.parse

import numpy as np

from viz.mapdata import CACHE, R_EARTH

OVERPASS = "https://overpass-api.de/api/interpreter"
UA = {"User-Agent": "UAV-sim-diploma/1.0 (local cache)"}

# дороги: ширина на карте, м, и цвет (тип грунт/асфальт)
ROAD_W = {"motorway": 12, "trunk": 10, "primary": 9, "secondary": 8, "tertiary": 7,
          "unclassified": 6, "residential": 5, "service": 4, "living_street": 4,
          "track": 3.5, "path": 1.5, "footway": 1.5}
RIVER_W = {"river": 25, "canal": 8, "stream": 3, "ditch": 1.5, "drain": 1.5}


def fetch(name, lat0, lon0, size=4000.0, force=False):
    """Скачать выгрузку Overpass в кэш (один раз). Возвращает dict JSON."""
    d = os.path.join(CACHE, name)
    path = os.path.join(d, "osm.json")
    if os.path.exists(path) and not force:
        return json.load(open(path, encoding="utf-8"))
    os.makedirs(d, exist_ok=True)
    h = size / 2
    dlat = np.degrees(h / R_EARTH)
    dlon = np.degrees(h / (R_EARTH * np.cos(np.radians(lat0))))
    bb = f"{lat0 - dlat},{lon0 - dlon},{lat0 + dlat},{lon0 + dlon}"
    q = (f"[out:json][timeout:120];("
         f"way({bb})[natural];relation({bb})[natural];way({bb})[landuse];relation({bb})[landuse];"
         f"way({bb})[waterway];way({bb})[water];relation({bb})[water];"
         f"way({bb})[highway];way({bb})[building];);out geom;")
    print(f"[osm] {name}: загрузка OpenStreetMap…")
    req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(), headers=UA)
    raw = urllib.request.urlopen(req, timeout=180).read()
    open(path, "wb").write(raw)
    data = json.loads(raw)
    print(f"[osm] готово: {len(data['elements'])} объектов → {path}")
    return data


def _local(geom, lat0, lon0):
    """[{lat, lon}, …] → массив (n, 2) [E, N], м."""
    la = np.array([g["lat"] for g in geom]); lo = np.array([g["lon"] for g in geom])
    N = R_EARTH * np.radians(la - lat0)
    E = R_EARTH * np.cos(np.radians(lat0)) * np.radians(lo - lon0)
    return np.column_stack([E, N])


def stitch_rings(parts, tol=0.5):
    """Собрать замкнутые контуры из кусков ломаных (члены мультиполигона)."""
    parts = [p for p in parts if len(p) >= 2]
    rings = []
    while parts:
        cur = parts.pop(0)
        changed = True
        while changed and np.linalg.norm(cur[0] - cur[-1]) > tol:
            changed = False
            for i, p in enumerate(parts):
                if np.linalg.norm(cur[-1] - p[0]) < tol:
                    cur = np.vstack([cur, p[1:]])
                elif np.linalg.norm(cur[-1] - p[-1]) < tol:
                    cur = np.vstack([cur, p[::-1][1:]])
                elif np.linalg.norm(cur[0] - p[-1]) < tol:
                    cur = np.vstack([p[:-1], cur])
                elif np.linalg.norm(cur[0] - p[0]) < tol:
                    cur = np.vstack([p[::-1][:-1], cur])
                else:
                    continue
                parts.pop(i)
                changed = True
                break
        if len(cur) >= 3:
            rings.append(cur)                 # незамкнутые остатки тоже рисуем (обрезка bbox)
    return rings


def classify(tags):
    """Класс площади по тегам OSM (или None)."""
    n, lu, w = tags.get("natural"), tags.get("landuse"), tags.get("water")
    if n == "water" or w or tags.get("waterway") == "riverbank" or lu in ("reservoir", "basin"):
        return "water"
    if n == "wood" or lu == "forest":
        return "wood"
    if n == "scrub":
        return "scrub"
    if n == "wetland":
        return "wetland"
    if lu in ("residential", "allotments", "garages", "industrial", "farmyard"):
        return "residential"
    if lu in ("farmland", "orchard"):
        return "farmland"
    if lu in ("meadow", "grass", "village_green") or n in ("grassland", "heath"):
        return "meadow"
    return None


def parse(data, lat0, lon0):
    """
    Выгрузка Overpass → геометрия в местных координатах:
      areas[класс] = [(внешние контуры, внутренние контуры), …]
      lines = [(точки (n, 2), ширина м, "water" | "road" | "track")]
      buildings = [контур (n, 2), …]
    """
    areas, lines, buildings = {}, [], []
    for el in data["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "way" and "geometry" in el:
            pts = _local(el["geometry"], lat0, lon0)
            closed = len(pts) > 3 and np.linalg.norm(pts[0] - pts[-1]) < 0.5
            if "building" in tags and closed:
                buildings.append(pts)
                continue
            cls = classify(tags) if closed else None
            if cls:
                areas.setdefault(cls, []).append(([pts], []))
            ww, hw = tags.get("waterway"), tags.get("highway")
            if ww in RIVER_W and not closed:
                lines.append((pts, RIVER_W[ww], "water"))
            elif hw in ROAD_W:
                lines.append((pts, ROAD_W[hw], "track" if hw in ("track", "path", "footway") else "road"))
        elif el["type"] == "relation":
            cls = classify(tags)
            if not cls:
                continue
            outer = [_local(m["geometry"], lat0, lon0) for m in el["members"]
                     if m.get("role") == "outer" and m.get("geometry")]
            inner = [_local(m["geometry"], lat0, lon0) for m in el["members"]
                     if m.get("role") == "inner" and m.get("geometry")]
            areas.setdefault(cls, []).append((stitch_rings(outer), stitch_rings(inner)))
    return areas, lines, buildings


def load(name, lat0, lon0, size=4000.0):
    return parse(fetch(name, lat0, lon0, size), lat0, lon0)
