"""
Модель ветра wind(h, t, params) -> (Vwx, Vwh, Vwy).

Возвращает компоненты ветра в земной СК (скорость воздушной массы):
  Vwx > 0 — ветер на север (при psi=0 — попутный), м/с
  Vwh > 0 — восходящий вертикальный ветер, м/с
  Vwy > 0 — ветер на восток (при psi=0 — боковой, слева направо), м/с
Боковая составляющая — последняя, чтобы пара (Vwx, Vwh) продольного кода
не сдвинулась.

Составляющие (складываются):
  1. Постоянный ветер (северный и восточный)
  2. Сдвиг ветра по высоте (северная составляющая, основной демонстрационный сценарий)
  3. Одиночный порыв (импульс) по всем трём осям
"""

import numpy as np
from .config import WindParams


def wind(h: float, t: float, params: WindParams) -> tuple:
    """
    Суммарный ветер на высоте h в момент времени t.

    Возвращает: (Vwx, Vwh, Vwy) в м/с
    """
    Vwx = _constant(params)
    Vwx += _shear(h, params)
    Vwx += _gust(t, params)
    Vwh  = _gust_vwh(t, params)
    Vwy  = params.Vw_cross + _gust_vwy(t, params)
    return Vwx, Vwh, Vwy


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


def _gust_vwh(t: float, params: WindParams) -> float:
    """Вертикальный порыв: та же временна́я маска, что и горизонтальный."""
    if params.gust_vwh == 0.0:
        return 0.0
    t0, dur = params.gust_t0, params.gust_dur
    return params.gust_vwh if t0 <= t <= t0 + dur else 0.0


def _gust_vwy(t: float, params: WindParams) -> float:
    """Боковой порыв: та же временна́я маска, что и горизонтальный."""
    if params.gust_vwy == 0.0:
        return 0.0
    t0, dur = params.gust_t0, params.gust_dur
    return params.gust_vwy if t0 <= t <= t0 + dur else 0.0


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
