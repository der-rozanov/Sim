"""
Модель ветра wind(h, t, params) -> (Vwx, Vwh).

Возвращает компоненты ветра в земной СК:
  Vwx > 0 — попутный горизонтальный ветер (в направлении полёта), м/с
  Vwh > 0 — восходящий вертикальный ветер, м/с

Три составляющие (складываются):
  1. Постоянный горизонтальный ветер
  2. Сдвиг ветра по высоте (основной демонстрационный сценарий)
  3. Одиночный порыв (импульс)
"""

import numpy as np
from config import WindParams


def wind(h: float, t: float, params: WindParams) -> tuple:
    """
    Суммарный ветер на высоте h в момент времени t.

    Возвращает: (Vwx, Vwh) в м/с
    """
    Vwx = _constant(params)
    Vwx += _shear(h, params)
    Vwx += _gust(t, params)
    return Vwx, 0.0   # вертикальный ветер пока нулевой (задел на будущее)


def _constant(params: WindParams) -> float:
    """Постоянный фоновый ветер."""
    return params.Vw_const


def _shear(h: float, params: WindParams) -> float:
    """
    Линейный сдвиг ветра в слое [h_shear_lo, h_shear_hi].
    Ниже слоя — нулевой прирост, выше — полный перепад dV_shear.
    """
    lo = params.h_shear_lo
    hi = params.h_shear_hi
    dV = params.dV_shear

    if hi <= lo or dV == 0.0:
        return 0.0

    h_clipped = np.clip(h, lo, hi)
    return dV * (h_clipped - lo) / (hi - lo)


def _gust(t: float, params: WindParams) -> float:
    """
    Прямоугольный порыв: амплитуда gust_amp, начало gust_t0, длительность gust_dur.
    """
    t0  = params.gust_t0
    dur = params.gust_dur
    amp = params.gust_amp

    if amp == 0.0:
        return 0.0

    if t0 <= t <= t0 + dur:
        return amp
    return 0.0
