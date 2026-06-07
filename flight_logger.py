# -*- coding: utf-8 -*-
"""
Логгер полётных данных.

Сохраняет результаты симуляции в файл .flightlog (формат npz под капотом):
  numpy-массивы по каналам + JSON-метаданные.

Использование:
    from flight_logger import FlightLogger

    logger = FlightLogger(
        scenario="С4: Срыв",
        description="Тримовый полёт → тангаж 20°/газ=0 → вывод −5°",
        aircraft=aircraft,
        wind_params=wind_params,
        cfg=cfg,
        sp=sp,
        trim=(alpha_trim, de_trim, thr_trim),
        events=[
            {"t": 8.0,  "label": "срыв",  "color": "red"},
            {"t": 25.0, "label": "вывод", "color": "dodgerblue"},
        ],
    )

    log = run(controls_fn, aircraft, wind_params, cfg, state0=s0)

    logger.save(
        log,
        path="results/s4.flightlog",
        # Опциональные каналы — NaN, если не переданы
        alpha_probe=alpha_probe_buf,   # list или array, рад
        alpha_est=alpha_est_buf,       # list или array, рад
        h_ref=h_ref_buf,               # list или array, м
        theta_ref=theta_ref_buf,       # list или array, рад
        # Парный прогон (второй агент, те же индексы t)
        paired={
            "label": "без зонда",
            "alpha_probe": ...,
            "alpha_est":   ...,
            "delta_e":     ...,
            "throttle":    ...,
            "h":           ...,
            "Va":          ...,
        },
    )

Загрузка:
    from flight_logger import load_log
    data = load_log("results/s4.flightlog")
    # data["t"], data["Va"], data["alpha_true"], data["meta"] (dict)
"""

import json
import os
import re
import sys
import datetime
import numpy as np

from sim.config import AircraftParams, WindParams, SimConfig, SensorParams
from runner import Log
from sim.state import THETA, Q, H, X, U, W

# Папка для логов по умолчанию — results/ рядом с flight_logger.py
_DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _sanitize_scenario(name: str) -> str:
    """
    Привести имя сценария к виду, пригодному для имени файла.
    'С8: Контроль высоты + Va + энергия' → 'Контроль_высоты_Va_энергия'
    """
    # Убрать префикс вида «С8: » или «S8: »
    name = re.sub(r'^[СсSs]\d+\s*[:\-]\s*', '', name).strip()
    # Разделители → подчёркивание
    name = re.sub(r'[/\\|,;°]+', '_', name)
    # Пробелы и + → подчёркивание
    name = re.sub(r'[\s+]+', '_', name)
    # Убрать недопустимые символы для имени файла
    name = re.sub(r'[<>?*"\']+', '', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name or "scenario"


def _auto_path(scenario: str, results_dir: str) -> str:
    """Сгенерировать путь вида results/Sim_{сценарий}_{ЧЧ-ММ-СС}_{ГГГГ-ММ-ДД}.flightlog."""
    os.makedirs(results_dir, exist_ok=True)
    now   = datetime.datetime.now()
    stem  = _sanitize_scenario(scenario)
    fname = f"Sim_{stem}_{now.strftime('%H-%M-%S')}_{now.strftime('%Y-%m-%d')}.flightlog"
    return os.path.join(results_dir, fname)


# ---------------------------------------------------------------------------
# Уровни индикатора УА
# ---------------------------------------------------------------------------
ALERT_NORM = 0
ALERT_WARN = 1
ALERT_CRIT = 2
ALERT_STALL = 3


def _compute_alert(alpha_rad: np.ndarray, aircraft: AircraftParams) -> np.ndarray:
    """Вычислить уровень тревоги (0–3) для каждого шага."""
    a = np.abs(alpha_rad)
    alert = np.zeros(len(a), dtype=np.int8)
    alert[a >= aircraft.alpha_warning] = ALERT_WARN
    alert[a >= aircraft.alpha_crit]    = ALERT_CRIT
    alert[a >= aircraft.alpha_stall]   = ALERT_STALL
    return alert


def _pad_or_slice(buf, n: int, fill=np.nan) -> np.ndarray:
    """Привести список/массив к длине n (обрезать или дополнить NaN)."""
    if buf is None:
        return np.full(n, fill)
    arr = np.asarray(buf, dtype=float)
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - len(arr), fill)])


# ---------------------------------------------------------------------------
# FlightLogger
# ---------------------------------------------------------------------------

class FlightLogger:
    """Собирает метаданные сценария и сохраняет лог в .flightlog."""

    def __init__(
        self,
        scenario: str,
        description: str = "",
        aircraft: AircraftParams = None,
        wind_params: WindParams = None,
        cfg: SimConfig = None,
        sp: SensorParams = None,
        trim: tuple = None,          # (alpha_trim_rad, de_trim_rad, thr_trim)
        events: list = None,         # [{"t": float, "label": str, "color": str}, ...]
        results_dir: str = None,     # папка для автосохранения (None = results/ рядом с модулем)
    ):
        self._scenario    = scenario
        self._description = description
        self._aircraft    = aircraft or AircraftParams()
        self._wind        = wind_params or WindParams()
        self._cfg         = cfg or SimConfig()
        self._sp          = sp or SensorParams()
        self._trim        = trim
        self._events      = events or []
        self._results_dir = results_dir or _DEFAULT_RESULTS_DIR

    # ------------------------------------------------------------------
    def save(
        self,
        log: Log,
        path: str = None,
        *,
        alpha_probe=None,
        alpha_est=None,
        h_ref=None,
        theta_ref=None,
        paired: dict = None,
    ) -> str:
        """
        Сохранить лог в файл.

        Parameters
        ----------
        log          : Log из runner.run()
        path         : путь к файлу; если None — генерируется автоматически:
                       results/Sim_{сценарий}_{ЧЧ-ММ-СС}_{ГГГГ-ММ-ДД}.flightlog
        alpha_probe  : list/array, измерения зонда, рад  (опционально)
        alpha_est    : list/array, косвенная оценка УА, рад (опционально)
        h_ref        : list/array, уставка высоты, м (опционально)
        theta_ref    : list/array, уставка тангажа, рад (опционально)
        paired       : dict с данными второго агента парного прогона (опционально)

        Returns
        -------
        str : итоговый путь к сохранённому файлу
        """
        if path is None:
            path = _auto_path(self._scenario, self._results_dir)
        n = len(log.t)
        ac = self._aircraft
        dt = log.t[1] - log.t[0] if n > 1 else self._cfg.dt

        # ---- Производные каналы ----------------------------------------
        alpha_true = log.alpha                         # рад
        gamma      = log.state[:, THETA] - alpha_true  # угол траектории по воздуху

        P_thrust = log.controls[:, 1] * log.Va        # нормированная мощность [о.е.*м/с]
        E_thrust  = np.cumsum(P_thrust) * dt           # нарастающий итог

        alert = _compute_alert(alpha_true, ac)

        # ---- Опциональные каналы ---------------------------------------
        ap_arr  = _pad_or_slice(alpha_probe, n)
        ae_arr  = _pad_or_slice(alpha_est,   n)
        hr_arr  = _pad_or_slice(h_ref,       n)
        thr_arr = _pad_or_slice(theta_ref,   n)

        # ---- Метаданные (JSON) -----------------------------------------
        trim_alpha, trim_de, trim_thr = (
            self._trim if self._trim is not None else (None, None, None)
        )

        meta = {
            "scenario":    self._scenario,
            "description": self._description,
            "saved_at":    datetime.datetime.now().isoformat(timespec="seconds"),
            "aircraft": {
                "mass": ac.mass, "Jy": ac.Jy,
                "S": ac.S, "b": ac.b, "c": ac.c,
                "rho": ac.rho, "g": ac.g,
                "CL0": ac.CL0, "CLa": ac.CLa, "CLde": ac.CLde,
                "CDp": ac.CDp, "e_oswald": ac.e_oswald,
                "Cm0": ac.Cm0, "Cma": ac.Cma, "Cmq": ac.Cmq, "Cmde": ac.Cmde,
                "alpha_stall_deg":   float(np.degrees(ac.alpha_stall)),
                "alpha_crit_deg":    float(np.degrees(ac.alpha_crit)),
                "alpha_warning_deg": float(np.degrees(ac.alpha_warning)),
                "k_motor": ac.k_motor, "T_max": ac.T_max,
            },
            "wind": {
                "Vw_const":   self._wind.Vw_const,
                "h_shear_lo": self._wind.h_shear_lo,
                "h_shear_hi": self._wind.h_shear_hi,
                "dV_shear":   self._wind.dV_shear,
                "gust_amp":   self._wind.gust_amp,
                "gust_t0":    self._wind.gust_t0,
                "gust_dur":   self._wind.gust_dur,
            },
            "sim": {
                "dt": self._cfg.dt, "t_end": self._cfg.t_end,
                "Va0": self._cfg.Va0, "h0": self._cfg.h0,
                "theta0": self._cfg.theta0,
            },
            "sensors": {
                "gyro_noise": self._sp.gyro_noise,
                "baro_noise": self._sp.baro_noise,
                "airspeed_noise": self._sp.airspeed_noise,
                "probe_noise": self._sp.probe_noise,
                "probe_bias":  self._sp.probe_bias,
                "gps_vel_noise": self._sp.gps_vel_noise,
            },
            "trim": {
                "alpha_deg": float(np.degrees(trim_alpha)) if trim_alpha is not None else None,
                "de_deg":    float(np.degrees(trim_de))    if trim_de    is not None else None,
                "throttle":  float(trim_thr)               if trim_thr   is not None else None,
            },
            "events": self._events,
            "alert_levels": {
                "NORM": 0, "WARN": 1, "CRIT": 2, "STALL": 3,
                "thresholds_deg": {
                    "warning": float(np.degrees(ac.alpha_warning)),
                    "crit":    float(np.degrees(ac.alpha_crit)),
                    "stall":   float(np.degrees(ac.alpha_stall)),
                },
            },
            "channels": {
                "t":          "время, с",
                "h":          "высота, м",
                "x":          "горизонтальная дальность, м",
                "u":          "продольная скорость (связ.), м/с",
                "w":          "нормальная скорость (связ.), м/с",
                "theta":      "угол тангажа, рад",
                "q":          "угловая скорость тангажа, рад/с",
                "Va":         "воздушная скорость, м/с",
                "alpha_true": "истинный УА, рад",
                "gamma":      "угол траектории (по воздуху), рад",
                "wind_x":     "горизонтальный ветер, м/с",
                "wind_h":     "вертикальный ветер, м/с",
                "delta_e":    "руль высоты, рад",
                "throttle":   "тяга, о.е. [0-1]",
                "E_kin":      "кинетическая энергия, Дж",
                "E_pot":      "потенциальная энергия, Дж",
                "E_mech":     "полная механическая энергия, Дж",
                "P_thrust":   "нормированная мощность двигателя, о.е.*м/с",
                "E_thrust":   "накопленная энергия двигателя, о.е.*м",
                "alert":      "индикатор УА: 0=НОРМ 1=ПРЕД 2=КРИТ 3=СРЫВ",
                "alpha_probe":"измерение зонда, рад (NaN если нет зонда)",
                "alpha_est":  "косвенная оценка УА (ИНС+GPS), рад (NaN если нет)",
                "h_ref":      "уставка высоты, м (NaN если нет контура высоты)",
                "theta_ref":  "уставка тангажа, рад (NaN если нет)",
            },
            "has_probe":    not np.all(np.isnan(ap_arr)),
            "has_est":      not np.all(np.isnan(ae_arr)),
            "has_h_ref":    not np.all(np.isnan(hr_arr)),
            "has_theta_ref":not np.all(np.isnan(thr_arr)),
            "has_paired":   paired is not None,
        }

        # ---- Сборка массивов -------------------------------------------
        arrays = dict(
            meta_json   = np.array([json.dumps(meta, ensure_ascii=False)]),
            t           = log.t,
            h           = log.state[:, H],
            x           = log.state[:, X],
            u           = log.state[:, U],
            w           = log.state[:, W],
            theta       = log.state[:, THETA],
            q           = log.state[:, Q],
            Va          = log.Va,
            alpha_true  = alpha_true,
            gamma       = gamma,
            wind_x      = log.wind_vec[:, 0],
            wind_h      = log.wind_vec[:, 1],
            delta_e     = log.controls[:, 0],
            throttle    = log.controls[:, 1],
            E_kin       = log.E_kin,
            E_pot       = log.E_pot,
            E_mech      = log.E_total,
            P_thrust    = P_thrust,
            E_thrust    = E_thrust,
            alert       = alert,
            alpha_probe = ap_arr,
            alpha_est   = ae_arr,
            h_ref       = hr_arr,
            theta_ref   = thr_arr,
        )

        # ---- Парный прогон ---------------------------------------------
        if paired is not None:
            for key, val in paired.items():
                if key == "label":
                    continue
                arr = _pad_or_slice(val, n) if not isinstance(val, np.ndarray) else val[:n]
                arrays[f"paired_{key}"] = arr
            meta["paired_label"] = paired.get("label", "парный прогон")

        # np.savez_compressed добавляет .npz если расширение не .npz —
        # сохраняем в tmp-путь и переименовываем в запрошенный path.
        tmp_path = path if path.endswith(".npz") else path + ".npz"
        np.savez_compressed(tmp_path, **arrays)
        if tmp_path != path:
            if os.path.exists(path):
                os.remove(path)
            os.rename(tmp_path, path)
        print(f"[FlightLogger] Сохранено: {path}  ({n} шагов, {log.t[-1]:.1f} с)")
        return path


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------

def load_log(path: str) -> dict:
    """
    Загрузить .flightlog файл.

    Возвращает словарь:
        data["t"], data["Va"], data["alpha_true"], ...   — numpy-массивы
        data["meta"]                                     — dict метаданных

    Пример:
        data = load_log("results/s8.flightlog")
        plt.plot(data["t"], np.degrees(data["alpha_true"]))
    """
    if not os.path.exists(path) and os.path.exists(path + ".npz"):
        path = path + ".npz"
    raw = np.load(path, allow_pickle=False)
    result = {}
    for key in raw.files:
        if key == "meta_json":
            result["meta"] = json.loads(str(raw["meta_json"][0]))
        else:
            result[key] = raw[key]
    return result


# ---------------------------------------------------------------------------
# Краткая сводка по загруженному логу
# ---------------------------------------------------------------------------

def print_log_info(data: dict) -> None:
    """Распечатать краткое содержимое лога."""
    m = data.get("meta", {})
    print(f"Сценарий : {m.get('scenario', '?')}")
    print(f"Описание : {m.get('description', '')}")
    print(f"Сохранён : {m.get('saved_at', '?')}")
    t = data["t"]
    print(f"Длительность : {t[-1]:.2f} с  ({len(t)} шагов, dt={t[1]-t[0]:.4f} с)")
    print(f"Каналы : {[k for k in data if k != 'meta']}")
    trim = m.get("trim", {})
    print(f"Трим : alpha={trim.get('alpha_deg', '?')} °  "
          f"de={trim.get('de_deg', '?')} °  thr={trim.get('throttle', '?')}")
    print(f"Ветер : Vw_const={m.get('wind', {}).get('Vw_const', 0)} м/с")
    events = m.get("events", [])
    if events:
        print(f"События :")
        for ev in events:
            print(f"  t={ev['t']:.1f} с — {ev['label']}")
    flags = ["has_probe", "has_est", "has_h_ref", "has_theta_ref", "has_paired"]
    active = [f for f in flags if m.get(f)]
    if active:
        print(f"Доп. каналы : {active}")


# ---------------------------------------------------------------------------
# CLI: python flight_logger.py <file.flightlog> [--stats]
# ---------------------------------------------------------------------------

def _cli_inspect(path: str, show_stats: bool = False) -> None:
    """Текстовый осмотр лог-файла без GUI."""
    data = load_log(path)
    m    = data["meta"]
    t    = data["t"]
    n    = len(t)
    dt   = t[1] - t[0] if n > 1 else 0.0

    sep = "─" * 60
    print(sep)
    print(f"  Файл     : {os.path.abspath(path)}")
    print(f"  Размер   : {os.path.getsize(path) / 1024:.1f} КБ")
    print(sep)
    print(f"  Сценарий : {m.get('scenario', '?')}")
    print(f"  Описание : {m.get('description', '')}")
    print(f"  Сохранён : {m.get('saved_at', '?')}")
    print(sep)
    print(f"  Время    : 0 … {t[-1]:.2f} с  ({n} шагов, dt={dt:.4f} с)")

    wind = m.get("wind", {})
    print(f"  Ветер    : Vwx={wind.get('Vw_const', 0):+.1f} м/с  "
          f"сдвиг={wind.get('dV_shear', 0):+.1f} м/с  "
          f"порыв={wind.get('gust_amp', 0):.1f} м/с")

    trim = m.get("trim", {})
    print(f"  Трим     : alpha={trim.get('alpha_deg', '?'):.3f}°  "
          f"de={trim.get('de_deg', '?'):.3f}°  "
          f"thr={trim.get('throttle', '?'):.4f}")

    events = m.get("events", [])
    if events:
        print(f"  События  : " +
              "  |  ".join(f"t={e['t']:.1f}с {e['label']}" for e in events))

    flags = {
        "has_probe":     "зонд УА",
        "has_est":       "ИНС+GPS оценка",
        "has_h_ref":     "уставка высоты",
        "has_theta_ref": "уставка тангажа",
        "has_paired":    "парный прогон",
    }
    active = [lbl for f, lbl in flags.items() if m.get(f)]
    print(f"  Доп. данные: {', '.join(active) if active else 'нет'}")

    channels = [k for k in data if k != "meta"]
    print(f"  Каналов  : {len(channels)}")
    for k in channels:
        arr = data[k]
        has_nan = bool(np.any(np.isnan(arr))) if arr.dtype.kind == "f" else False
        nan_str = "  (содержит NaN)" if has_nan else ""
        print(f"    {k:<16s}  shape={arr.shape}  dtype={arr.dtype}{nan_str}")

    if show_stats:
        print(sep)
        print("  Статистика по каналам:")
        skip = {"alert"}
        for k in channels:
            arr = data[k]
            if arr.dtype.kind not in ("f", "i") or k in skip:
                continue
            valid = arr[~np.isnan(arr)] if arr.dtype.kind == "f" else arr
            if len(valid) == 0:
                continue
            print(f"    {k:<16s}  min={valid.min():+.4f}  max={valid.max():+.4f}"
                  f"  mean={valid.mean():+.4f}  std={valid.std():.4f}")

    al_levels = {0: "НОРМ", 1: "ПРЕД", 2: "КРИТ", 3: "СРЫВ"}
    alert = data["alert"]
    dt_s = dt
    print(sep)
    print("  Индикатор УА:")
    for lv, lbl in al_levels.items():
        secs = float(np.sum(alert == lv)) * dt_s
        print(f"    {lbl:6s}  {secs:6.2f} с")
    print(sep)


if __name__ == "__main__":
    import argparse
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Текстовый осмотр .flightlog файла")
    parser.add_argument("file", help="Путь к .flightlog файлу")
    parser.add_argument("--stats", action="store_true",
                        help="Показать min/max/mean/std по каждому каналу")
    args = parser.parse_args()
    _cli_inspect(args.file, show_stats=args.stats)
