# -*- coding: utf-8 -*-
"""
GameScenario3D — пилотирование ЛА в реальном времени в 3D (Ursina).

Физика — та же 6DOF-модель (step_rk4, dt = 0.01 с, реальное время);
сцена, силуэт, камеры — viz/viewer3d.View3D. По выходу полёт сохраняется
в results/*.flightlog — его можно пересмотреть в python viz/viewer3d.py.

РУЧНОЙ режим (рули отклоняются, пока клавиша нажата, и плавно возвращаются
к балансировке — как ручка RC-пульта):
  W / S  — руль высоты (W — нос вниз, S — нос вверх)
  A / D  — элероны (крен влево / вправо)
  Q / E  — руль направления (нос влево / вправо)
  Shift  — большие расходы: с W/S/A/D/Q/E рули ±10° (без Shift: δe ±2°, δa, δr ±3°)
  ↑ / ↓  — триммер руля высоты ±0.3° (↑ — нос вниз)
  X / Z  — тяга ±5 %

САУ (P — вкл/выкл; при включении захватывает текущие θ, Va, путевой угол χ):
  W / S  — уставка тангажа θ_ref ∓1°   (в режиме H — уставка высоты ∓10 м)
  A / D  — уставка путевого угла χ_ref ∓10°
  X / Z  — уставка скорости Va_ref ±1 м/с
  H      — удержание высоты (захват текущей)
  N      — навигация по маршруту вкл/выкл (маршрут — в окне станции; уставки χ_ref
           и h_ref даёт WaypointNavigator, control/navigation.py). Ручные уставки
           W/S/A/D навигацию выключают.
САУ — control/sau.py (тот же класс, что в s8–s10): χ → φ_ref → δa, β → δr,
h → θ_ref → δe, Va → δt.

Окно наземной станции (viz/station.py, отдельный процесс): PFD, карта с маршрутом
(точки ставятся мышью, FLY — САУ ведёт по точкам), рули, α/β, TECS, журнал событий.
Измерения в САУ — истинные (без шума).

Общие: Пробел — пауза, R — сброс, 1–4 — камера, M — масштаб силуэта, F1 — уровень HUD,
F2 — профилировщик кадра (viz/perf.py; итог печатается при выходе), Esc — выход.
Внизу справа: FPS; «sim ×» — скорость модели относительно реального времени (1.00 — успевает,
меньше — шагов физики на кадр не хватает, полёт замедлен); «lag» — накопленное отставание, с. HUD — viz/hud3d.py (подписи по-английски).

Земля твёрдая (sim/ground.py, РЕШ-25): поверхность карты — World.surface(), как
нарисована, с ВПП и покрытиями. ЛА садится на шасси, катится (Q/E — носовое
колесо), взлетает. Авария: касание конструкцией (винт, крыло, хвост), касание
колесом быстрее gear_v_crash (поломка шасси), касание воды.

Запуск:
    python scenarios/GameScenario3D.py
    python scenarios/GameScenario3D.py --wind-n -5 --wind-e 3   # ветер, м/с
    python scenarios/GameScenario3D.py --ap --cam 3              # старт с включённой САУ
    python scenarios/GameScenario3D.py --no-map                  # без окна станции
    python scenarios/GameScenario3D.py --ap --bench 60           # замер FPS: 60 с с САУ и выход
"""

import sys
import os
import argparse
import queue
import multiprocessing as mp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.turbulence import TURB_LEVELS
from sim.config import WindParams, SimConfig, SensorParams, AIRCRAFT_TYPES, aircraft_by_name
from sim.integrators import step_rk4
from sim.dynamics import specific_force
from sim.actuators import actuator_init, actuator_step
from sim.ground import contacts, near_ground
from sim.wind import make_wind_fn
from sim.state import (air_data, earth_velocity, total_energy, full_controls, canonical_euler,
                       wind_body, U, V, W, THETA, X, H, PHI, PSI, Y, N_STATES, N_CONTROLS)
from runner import Log, compute_trim, trim_state
from flight_logger import FlightLogger
from control.controllers import PitchControlParams, wrap_angle
from control.navigation import WaypointNavigator, NavParams
from control.sau import SAU, SAUParams, TruthSensors, RealisticSensors, Mode
from control.estimator import StateEstimator
from sim.aero import coef_CL
from viz import station
from viz.viewer3d import (View3D, make_app, MAPS, ned_to_u, body_axes, air_velocity_axis,
                          screenshot_and_quit)
from viz.hud3d import HudData
from ursina import Entity, application, held_keys, time as utime

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
# Двойные расходы (dual rate, как на RC-пульте): без Shift — малые отклонения для точного
# пилотирования, с Shift — большие для энергичного маневра (выравнивание, вывод из сваливания).
# Отклик FPV в разомкнутом контуре (Va = 16 м/с, руль удерживается 1 с, штиль):
#   δe 2° → q ≈ 21°/с, θ ≈ 18°;   δe 10° → q ≈ 106°/с, α ≈ 8.5° (порог предупреждения 9°)
#   δa 3° → p ≈ 29°/с, φ ≈ 26°;   δa 10° → p ≈ 100°/с
#   δr 3° → r ≈ 13°/с;            δr 10° → r ≈ 44°/с
DE_DEFL   = np.radians(2.0)     # отклонение руля высоты от триммера при нажатой W/S
DA_DEFL   = np.radians(3.0)     # элероны при A/D
DR_DEFL   = np.radians(3.0)     # руль направления при Q/E
HI_DEFL   = np.radians(10.0)    # все рули при нажатом Shift (большие расходы)
SURF_RATE = np.radians(20.0)    # скорость отклонения «ручки» с клавиатуры (команда; привод — sim/actuators.py)
TRIM_STEP = np.radians(0.3)
# Клавиши ручного пилотирования: при потере фокуса 3D-окном Ursina не получает
# «клавиша отпущена» — held_keys остаются 1 и рули «заклинивает» (полёт 2026-10-08)
PILOT_KEYS = ("w", "s", "a", "d", "q", "e", "shift", "left shift", "right shift")
THR_STEP  = 0.05

THETA_STEP, THETA_LIM = np.radians(1.0), np.radians(20.0)
CHI_STEP  = np.radians(10.0)
VA_STEP, VA_MIN, VA_MAX = 1.0, 10.0, 30.0   # FPV: сваливание 7.5, максимум ≈31.5 м/с
H_STEP, H_MIN, H_MAX = 10.0, 20.0, 1000.0
KH        = PitchControlParams().KH   # рад/м, расчёт control/tuning.py

MANUAL_REF = ("маршрут: ручная уставка", "NAV: manual ref")   # событие: ручная уставка выключила маршрут
MAX_STEPS_PER_FRAME = 10        # при просадке кадров — замедление, а не «рывок»
RT_TAU    = 1.0                 # с — постоянная сглаживания скорости модели (строка «sim ×…»)
RT_WARMUP = 2.0                 # с — отставание не считается с запуска окна (загрузка)
TRAIL_MAX = 3000                # точек следа (0.1 с) — последние 5 мин


def _wrap180(a):
    """Угол в градусах → [−180, 180) (крен копится при бочках)."""
    return (a + 180.0) % 360.0 - 180.0


class Game(Entity):

    HELP = ("W/S δe  A/D δa  Q/E δr  ↑/↓ trim  X/Z thr  P AP  H alt  N nav  "
            "T turb  Space pause  R reset  1-4 cam  Esc quit")

    def __init__(self, aircraft, wind_params, cfg, terrain, cam=1, ap=False, shot=None,
                 q_station=None, bench=None, sensors="truth", est="ekf"):
        """q_station — (q_cmd, q_tel) окна станции (viz/station.py) или None;
        bench — выход через bench с (замер FPS, итог профилировщика — в консоль);
        sensors — датчики САУ: "truth" (истина) | "real" (RealisticSensors);
        est — оценка состояния при sensors="real": "ekf" (StateEstimator) | "none"."""
        self.sensors_kind, self.est_kind = sensors, est
        self.q_station, self.bench, self._t_wall = q_station, bench, 0.0
        self.nav = WaypointNavigator(NavParams())
        self.route, self.loop = [], False        # маршрут из окна станции: [(N, E, h), ...]
        super().__init__()
        self.ac, self.wp, self.cfg = aircraft, wind_params, cfg
        self.wind_call = make_wind_fn(wind_params)
        self.alpha_trim, self.de_trim, self.thr_trim = compute_trim(aircraft, cfg.Va0)
        self.s0 = trim_state(aircraft, cfg)
        self.s0[PSI] = terrain.start_psi          # старт вдоль ВПП карты
        # старт в балансировке относительно воздуха: земная скорость = воздушная + ветер
        # (без скачка скольжения в первую секунду при --wind-n / --wind-e)
        self.s0[[U, V, W]] += wind_body(self.s0, self.wind_call(cfg.h0, 0.0))
        self.ter = terrain
        self.ground = lambda N, E: terrain.surface(E, N)    # NED → Ursina: X — восток, Z — север
        self.view = View3D(aircraft.b, aircraft.c, terrain, self.HELP, cam=cam,
                           aircraft_name=aircraft.name)

        self.sau = self._make_sau()

        self.rec = []                  # (t, state, controls, Va, alpha, beta, wind, cmd)
        self.events = []
        self.shot, self._frames = shot, 0
        self.reset()
        if ap:
            self._toggle_ap()

    def _make_sau(self):
        """САУ (датчики — _make_sensors); стенд Bench3D подменяет её своей (защита по α)."""
        params = SAUParams(pitch=PitchControlParams(Va_ref=self.cfg.Va0),
                           h_min=H_MIN, h_max=H_MAX)
        return SAU(self.ac, (self.alpha_trim, self.de_trim, self.thr_trim),
                   self._make_sensors(), params, nav=self.nav, estimator=self._make_estimator())

    def _make_sensors(self):
        """Датчики САУ: истинные значения или реалистичный набор (--sensors real)."""
        self.sensor_params = SensorParams()
        if self.sensors_kind == "truth":
            return TruthSensors()
        return RealisticSensors(self.sensor_params, np.random.default_rng(0),
                                accel_fn=self._specific_force)

    def _make_estimator(self):
        """Оценка состояния: EKF (control/estimator.py) — только на реалистичных датчиках."""
        if self.sensors_kind == "real" and self.est_kind == "ekf":
            return StateEstimator(self.sensor_params)
        return None

    def _specific_force(self, t, s):
        """Удельная сила для акселерометра: те же рули, ветер и земля, что у физики."""
        gf = self.ground if near_ground(s, self.ac, self.ground) else None
        return specific_force(s, self.controls, t, self.ac, self.wind_call, gf)

    @property
    def nav_on(self):
        return self.ap and self.sau.mode is Mode.ROUTE

    @property
    def h_hold(self):
        """Высоту держит САУ: режим ALTITUDE или ROUTE."""
        return self.ap and self.sau.mode is not Mode.PITCH

    # --- состояние игры ---------------------------------------------------
    def reset(self):
        if self.rec:                       # каждый полёт — свой файл (время в логе монотонно)
            self.save()
            self.rec, self.events = [], []
        self.state = self.s0.copy()
        self.t, self.acc = 0.0, 0.0
        self.wind_call = make_wind_fn(self.wp)  # ряд порывов — заново с t = 0
        self.t_real, self.rt_ratio = 0.0, 1.0   # реальное время полёта (без пауз), с; скорость модели
        self.paused, self.crashed = False, False
        self.on_ground, self.crash_msg, self.crash_en = False, "", ""
        self.ap = False
        self.de_trim_man = self.de_trim
        self.thr = self.thr_trim
        self.surf = np.array([self.de_trim, 0.0, 0.0])        # δe, δa, δr (ручной)
        self.controls = actuator_init([self.de_trim, self.thr_trim, 0.0, 0.0], self.ac)
        self.cmd = self.controls.copy()                       # команда до приводов рулей
        self.sau.set_mode(Mode.PITCH)
        self.sau.engage(theta=self.alpha_trim, h=self.cfg.h0, Va=self.cfg.Va0, chi=0.0)
        self.trail = []

    def _cycle_turb(self):
        """Турбулентность: выкл → слабая → умеренная → выкл; включается с текущего t."""
        lv = TURB_LEVELS[(TURB_LEVELS.index(self.wp.turb_level) + 1) % len(TURB_LEVELS)]
        self.wp.turb_level = lv
        self.wind_call = make_wind_fn(self.wp, t0=self.t)
        ru = {"off": "выкл", "light": "слабая", "moderate": "умеренная"}[lv]
        self._event(f"турбулентность: {ru}", "gray", en=f"turbulence: {lv}")

    def _event(self, label, color="orange", en=None):
        """label — в лог (по-русски); en — подпись для окна станции (по-английски)."""
        ev = {"t": round(self.t, 2), "label": label, "color": color}
        if en:
            ev["label_en"] = en
        self.events.append(ev)

    def _toggle_ap(self):
        if self.ap:
            self._nav_off()
            self.ap = False
            self.de_trim_man = self.controls[0]          # без рывка при выключении
            self.surf = np.array([self.controls[0], 0.0, 0.0])
            self.thr = self.controls[1]
            self._event("САУ выкл", "gray", "AP OFF")
            return
        self.ap = True
        s = self.state
        Va, _, _ = air_data(s, self.wind_call(s[H], self.t))
        Vx, Vy, _ = earth_velocity(s)
        self.sau.set_mode(Mode.PITCH)
        self.sau.engage(theta=float(np.clip(s[THETA], -THETA_LIM, THETA_LIM)), h=float(s[H]),
                        Va=float(np.clip(Va, VA_MIN, VA_MAX)), chi=float(np.arctan2(Vy, Vx)))
        self._event("САУ вкл", "dodgerblue", "AP ON")

    # --- навигация по маршруту ----------------------------------------------
    def _nav_go(self):
        """Лететь по маршруту с первой точки (САУ включается, если выключена)."""
        if not self.route or self.crashed:
            return
        if not self.ap:
            self._toggle_ap()
        s = self.state
        self.sau.fly_route(self.route, start=(s[X], s[Y], s[H]))
        self._event("маршрут: старт", "dodgerblue", "NAV: start")

    def _nav_off(self, why="маршрут: стоп", why_en="NAV: stop"):
        """Навигация выкл: САУ держит текущие путевой угол и высоту."""
        if not self.nav_on:
            return
        Vx, Vy, _ = earth_velocity(self.state)
        self.sau.refs.chi = float(np.arctan2(Vy, Vx))
        self.sau.refs.h = float(np.clip(self.state[H], H_MIN, H_MAX))
        self.sau.set_mode(Mode.ALTITUDE)
        self._event(why, "gray", why_en)

    def _flight_mode(self):
        """Режим полёта для окна станции: MAN / PITCH / ALT / NAV."""
        return "NAV" if self.nav_on else "ALT" if self.h_hold else "PITCH" if self.ap else "MAN"

    def _set_vert(self, to_alt):
        """Вертикальный канал САУ: удержание текущей высоты или тангажа; маршрут — стоп."""
        self._nav_off(*MANUAL_REF)
        self.sau.refs.h = float(self.state[H])
        self.sau.set_mode(Mode.ALTITUDE if to_alt else Mode.PITCH)
        self._event("удержание h" if to_alt else "θ_ref", "dodgerblue", "ALT hold" if to_alt else "PITCH hold")

    def _set_flight_mode(self, fm):
        """Режим полёта из окна станции — то же, что клавиши P (MAN), H (PITCH ⇄ ALT), N (NAV)."""
        if fm == "MAN":
            if self.ap:
                self._toggle_ap()
        elif fm == "NAV":
            self._nav_go()
        elif fm in ("PITCH", "ALT"):
            if not self.ap:
                self._toggle_ap()                           # САУ включается в режиме PITCH
            if fm != self._flight_mode():
                self._set_vert(fm == "ALT")

    def _poll_station(self):
        if self.q_station is None:
            return
        while True:
            try:
                kind, val = self.q_station[0].get_nowait()
            except queue.Empty:
                break
            if kind == "route":
                self.route, self.loop = [tuple(w) for w in val["wps"]], bool(val["loop"])
                self.nav.params.loop = self.loop
                pts = ned_to_u(*(np.array(self.route) * [1, 1, -1]).T) if self.route else []
                self.view.set_route(pts, self.loop)
                if self.nav_on and not self.route:
                    self._nav_off()
                elif self.nav_on:                  # правка в полёте — к точке с тем же номером
                    self.nav.update_route(self.route)
            elif kind == "go":
                self._nav_go()
            elif kind == "stop":
                self._nav_off()
            elif kind == "mode":
                self._set_flight_mode(val)

    def _fma(self):
        """Табло режимов станции: 5 колонок (строка 1, строка 2, цвет 1, цвет 2).
        Цвета: g — активный режим, c — уставка, w — нейтрально, a — предупреждение, d — пусто."""
        deg, refs, sau = np.degrees, self.sau.refs, self.sau
        if not self.ap:
            return [("MAN THR", f"{100 * self.controls[1]:.0f} %", "w", "d"),
                    ("MAN", f"TRIM {deg(self.de_trim_man):+.1f}°", "w", "d"),
                    ("MAN", self._route_tag(), "w", "d"),
                    self._fma_prot(), ("MAN", "", "w", "d")]
        thr = ("TECS" if sau.params.lon == "tecs" else "SPEED", f"SPD {refs.Va:.1f}", "g", "c")
        ver = (("ALT", f"{refs.h:.0f} m", "g", "c") if self.h_hold
               else ("PITCH", f"{deg(refs.theta):+.1f}°", "g", "c"))
        if self.nav_on:
            lat = ("NAV ORBIT" if self.nav.mode == "orbit" else f"NAV {self.nav.idx + 1}/{len(self.route)}",
                   f"TRK {deg(refs.chi) % 360:03.0f}°", "g", "c")
        else:
            lat = ("TRK", f"{deg(refs.chi) % 360:03.0f}°", "g", "c")
        return [thr, ver, lat, self._fma_prot(), ("AP", "" if self.nav_on else self._route_tag(), "w", "d")]

    def _route_tag(self):
        """Маршрут загружен, но не активен (N / FLY — лететь): серым, чтобы не путать с режимом."""
        return f"ROUTE {len(self.route)} WP" if self.route else ""

    def _fma_prot(self):
        """Колонка защиты по α: в игре защиты нет — только предупреждение по УА."""
        a = self._alert()
        return ("AOA", ["", "WARN", "CRIT", "STALL"][a], "a", "a" if a == 1 else "r") if a else ("", "", "d", "d")

    def _alert(self):
        alpha = air_data(self.state, self.wind_call(self.state[H], self.t))[1]
        return int(sum(alpha >= x for x in (self.ac.alpha_warning, self.ac.alpha_crit, self.ac.alpha_stall)))

    def _station_telemetry(self):
        if self.q_station is None:
            return
        s, nav, c, cmd, deg = self.state, self.nav, self.controls, self.cmd, np.degrees
        w = self.wind_call(s[H], self.t)
        Va, alpha, beta = air_data(s, w)
        Vx, Vy, Vh = earth_velocity(s)
        tgt = nav.target if self.nav_on else None
        refs, lon = self.sau.refs, self.sau.lon
        tecs = None
        if self.ap and self.sau.params.lon == "tecs":       # производные энергии × Va — в м/с набора
            Vs = max(Va, 1.0)
            tecs = (lon.E * Vs, lon.E_ref * Vs, lon.B * Vs, lon.B_ref * Vs)
        fm = self._flight_mode()
        mode = "AUTO · " + fm if self.ap else "MANUAL"
        msg = {"fm": fm,"t": self.t, "n": s[X], "e": s[Y], "h": s[H], "Vh": Vh, "Va": Va, "Vg": float(np.hypot(Vx, Vy)),
               "psi": deg(s[PSI]) % 360, "chi": deg(np.arctan2(Vy, Vx)) % 360,
               "phi": _wrap180(deg(s[PHI])), "theta": deg(s[THETA]), "alpha": deg(alpha), "beta": deg(beta),
               "de": deg(c[0]), "da": deg(c[2]), "dr": deg(c[3]), "thr": c[1],
               "de_cmd": deg(cmd[0]), "da_cmd": deg(cmd[2]), "dr_cmd": deg(cmd[3]), "thr_cmd": cmd[1],
               "wind_n": w[0], "wind_e": w[2], "alert": self._alert(),
               "ap": self.ap, "mode": mode, "fma": self._fma(), "tecs": tecs,
               "Va_ref": refs.Va if self.ap else None, "h_ref": refs.h if self.h_hold else None,
               "chi_ref": deg(refs.chi) % 360 if self.ap else None,
               "status": ("CRASH" if self.crashed else "PAUSED" if self.paused
                          else "ON GROUND" if self.on_ground else "FLYING"),
               "events": [(e["t"], e.get("label_en", e["label"]), e["color"]) for e in self.events[-14:]],
               "nav": self.nav_on, "idx": nav.idx, "mode_nav": nav.mode,
               "circle": (tuple(nav.circle[:3])
                          if self.nav_on and nav.mode in ("fillet", "orbit") else None),
               "dist": float(np.hypot(tgt[0] - s[X], tgt[1] - s[Y])) if tgt is not None else 0.0}
        try:
            self.q_station[1].put_nowait(msg)
        except queue.Full:
            pass

    # --- ввод ---------------------------------------------------------------
    def input(self, key):
        if self.view.handle_key(key):
            return
        base = key.replace(" hold", "")
        if key == "escape":
            application.quit()
        elif key == "space":
            self.paused = not self.paused
        elif key == "r":
            self.reset()
        elif key == "p":
            self._toggle_ap()
        elif key == "t":
            self._cycle_turb()
        elif key == "n":
            if self.nav_on:
                self._nav_off()
            else:
                self._nav_go()
        elif key == "h" and self.ap:                        # высота ⇄ тангаж (маршрут — стоп, далее высота)
            self._set_vert(self.sau.mode is not Mode.ALTITUDE)   # до _nav_off: он сам ставит ALTITUDE
        elif base in ("x", "z"):
            sgn = 1 if base == "x" else -1
            if self.ap:
                refs = self.sau.refs
                refs.Va = float(np.clip(refs.Va + sgn * VA_STEP, VA_MIN, VA_MAX))
            else:
                self.thr = float(np.clip(self.thr + sgn * THR_STEP, 0.0, 1.0))
        elif self.ap and base in ("w", "s"):
            self._nav_off(*MANUAL_REF)
            sgn, refs = (1 if base == "s" else -1), self.sau.refs     # S — вверх
            if self.h_hold:
                refs.h = float(np.clip(refs.h + sgn * H_STEP, H_MIN, H_MAX))
            else:
                refs.theta = float(np.clip(refs.theta + sgn * THETA_STEP,
                                           -THETA_LIM, THETA_LIM))
        elif self.ap and base in ("a", "d"):
            self._nav_off(*MANUAL_REF)
            sgn = 1 if base == "d" else -1                  # D — вправо
            self.sau.refs.chi = wrap_angle(self.sau.refs.chi + sgn * CHI_STEP)
        elif not self.ap and base in ("up arrow", "down arrow"):
            sgn = 1 if base == "up arrow" else -1           # ↑ — нос вниз (δe > 0)
            self.de_trim_man = float(np.clip(self.de_trim_man + sgn * TRIM_STEP,
                                             self.ac.delta_e_min, self.ac.delta_e_max))

    # --- управление на шаг ----------------------------------------------------
    def _manual(self, dt):
        hi = held_keys["shift"] or held_keys["left shift"] or held_keys["right shift"]  # большие расходы
        de, da, dr = (HI_DEFL,) * 3 if hi else (DE_DEFL, DA_DEFL, DR_DEFL)
        cmd = np.array([
            self.de_trim_man + de * (held_keys["w"] - held_keys["s"]),
            da * (held_keys["d"] - held_keys["a"]),
            dr * (held_keys["q"] - held_keys["e"]),
        ])
        ac = self.ac                                    # рули не дальше упоров
        cmd = np.clip(cmd, [ac.delta_e_min, -ac.delta_a_max, -ac.delta_r_max],
                      [ac.delta_e_max, ac.delta_a_max, ac.delta_r_max])
        self.surf += np.clip(cmd - self.surf, -SURF_RATE * dt, SURF_RATE * dt)
        return np.array([self.surf[0], self.thr, self.surf[1], self.surf[2]])

    def _sau(self, s, Va, alpha, beta, dt):
        return self.sau.step(self.t, s, Va, alpha, beta, dt)

    # --- кадр -------------------------------------------------------------------
    def _focus_check(self):
        """3D-окно без фокуса → отпустить клавиши пилотирования (рули к балансировке)."""
        win = application.base.win
        self.focus = win is None or win.getProperties().getForeground()
        if not self.focus:
            for k in PILOT_KEYS:
                held_keys[k] = 0

    def update(self):
        perf = self.view.perf
        with perf("station"):
            self._focus_check()
            self._poll_station()
        dt, n = self.cfg.dt, 0
        if not (self.paused or self.crashed):
            self.acc += utime.dt
            n = int(self.acc / dt)
            if n > MAX_STEPS_PER_FRAME:
                n, self.acc = MAX_STEPS_PER_FRAME, 0.0
            else:
                self.acc -= n * dt
            t0 = self.t
            with perf("physics"):
                for _ in range(n):
                    self._step()
                    if self.crashed:
                        break
            # отставание от реального времени: время модели теряется, когда шагов на кадр
            # не хватает (MAX_STEPS_PER_FRAME) — модель идёт медленнее часов
            self.t_real += utime.dt
            if self._t_wall < RT_WARMUP:           # начало — загрузка (шейдеры, окна-спутники), не считать
                self.t_real = self.t
            elif utime.dt > 0:
                k = min(1.0, utime.dt / RT_TAU)
                self.rt_ratio += k * ((self.t - t0) / utime.dt - self.rt_ratio)
        perf.count("steps/frame", n)

        s = self.state
        pos = ned_to_u(s[X], s[Y], -s[H])
        x_b, z_b = body_axes(s[PHI], s[THETA], s[PSI])
        nose, up = ned_to_u(*x_b), ned_to_u(*(-z_b))
        c = self.controls
        self.view.render(pos, nose, up, c[0], c[2], c[3], c[1],
                         np.array(self.trail), utime.dt)
        self._hud()
        with perf("station"):
            self._station_telemetry()
        perf.frame(utime.dt)
        self.view.set_rate(utime.dt, lag=max(0.0, self.t_real - self.t), ratio=self.rt_ratio)

        self._frames += 1
        if self.shot and self._frames == 300:
            screenshot_and_quit(self.shot)
        self._t_wall += utime.dt
        if self.bench and self._t_wall >= self.bench:
            application.quit()

    def _step(self):
        s, dt = self.state, self.cfg.dt
        w_vec = self.wind_call(s[H], self.t)
        Va, alpha, beta = air_data(s, w_vec)
        if not self.ap:                    # датчики и оценка состояния работают всегда
            self.sau.observe(self.t, s, Va, alpha, beta, dt)
        self.cmd = full_controls(self._sau(s, Va, alpha, beta, dt) if self.ap
                                 else self._manual(dt))
        self.controls = actuator_step(self.controls, self.cmd, dt, self.ac)  # рули — с приводом
        self.rec.append((self.t, s.copy(), self.controls.copy(), Va, alpha, beta,
                         np.asarray(w_vec, float), self.cmd.copy()))
        if round(self.t / dt) % 10 == 0:
            self.trail.append(ned_to_u(s[X], s[Y], -s[H]))
            del self.trail[:-TRAIL_MAX]
        # земля — в шаг только у поверхности (вызов поверхности дорогой)
        gf = self.ground if near_ground(s, self.ac, self.ground) else None
        # После петель/бочек углы Эйлера копятся — приводим к канонич. виду (та же ориентация)
        self.state = canonical_euler(step_rk4(s, self.controls, dt, self.t, self.ac,
                                              self.wind_call, gf))
        self.t += dt
        if gf is not None:
            self._ground_contact()
        else:
            self.on_ground = False

    def _ground_contact(self):
        """Касание земли: посадка / авария (конструкция, шасси, вода)."""
        c = contacts(self.state, self.ac, self.ground)
        on = c["depth"] > 0
        if not on.any():
            self.on_ground = False
            return
        v_sink = max(0.0, float(-c["vn"][on].min()))         # скорость к земле, м/с
        wet = self.ter.is_water(c["pos"][on, 1], c["pos"][on, 0]).any()
        if (on & ~c["wheel"]).any():
            self._crash("УДАР О ЗЕМЛЮ", "CRASH: AIRFRAME HIT")
        elif wet:
            self._crash("ПРИВОДНЕНИЕ", "CRASH: WATER")
        elif v_sink > self.ac.gear_v_crash:
            self._crash(f"ПОЛОМКА ШАССИ ({v_sink:.1f} м/с)", f"CRASH: GEAR ({v_sink:.1f} m/s)")
        elif not self.on_ground:
            self._event(f"касание {v_sink:.1f} м/с", "green", f"touchdown {v_sink:.1f} m/s")
        self.on_ground = True

    def _crash(self, msg, msg_en):
        """msg — событие в логе (по-русски, как остальные), msg_en — для HUD."""
        self.crashed, self.crash_msg, self.crash_en = True, msg, msg_en
        self._event(msg.lower(), "red", msg_en)

    def _modes(self) -> str:
        """Строка режимов HUD: только активные (как табло режимов на PFD)."""
        deg, refs = np.degrees, self.sau.refs
        if not self.ap:
            return f"MAN   TRIM δe {deg(self.de_trim_man):+.1f}°"
        m = ["AP", "TECS" if self.sau.params.lon == "tecs" else "SPD/ALT"]
        m.append(f"ALT {refs.h:.0f}" if self.h_hold else f"PITCH {deg(refs.theta):+.1f}°")
        m.append(f"SPD {refs.Va:.1f}")
        if self.nav_on:
            m.append("NAV ORBIT" if self.nav.mode == "orbit"
                     else f"NAV {self.nav.idx + 1}/{len(self.route)}")
        else:
            m.append(f"TRK {deg(refs.chi) % 360:03.0f}")
        return "   ".join(m)

    def _hud_data(self) -> HudData:
        s, c = self.state, self.controls
        w_vec = self.wind_call(s[H], self.t)
        Va, alpha, beta = air_data(s, w_vec)
        Vx, Vy, Vh = earth_velocity(s)
        deg, refs = np.degrees, self.sau.refs
        al = [self.ac.alpha_warning, self.ac.alpha_crit, self.ac.alpha_stall]
        v = air_velocity_axis(s[PHI], s[THETA], s[PSI], alpha, beta)
        x_b, z_b = body_axes(s[PHI], s[THETA], s[PSI])
        return HudData(
            t=self.t, Va=Va, h=s[H], Vh=Vh, alpha=deg(alpha), beta=deg(beta),
            phi=_wrap180(deg(s[PHI])), psi=deg(s[PSI]) % 360, chi=deg(np.arctan2(Vy, Vx)) % 360,
            thr=c[1], wind_n=w_vec[0], wind_e=w_vec[2],
            alert=int(sum(alpha >= x for x in al)), modes=self._modes(),
            status="PAUSED" if self.paused else ("ON GROUND" if self.on_ground else ""),
            event=f"{self.crash_en} — R to reset" if self.crashed else "",
            Va_ref=refs.Va if self.ap else None, h_ref=refs.h if self.h_hold else None,
            chi_ref=deg(refs.chi) % 360 if self.ap else None,
            nose=ned_to_u(*x_b), vel=ned_to_u(*v))

    def _hud(self):
        with self.view.perf("hud data"):
            d = self._hud_data()
        self.view.set_hud(d)

    # --- сохранение -------------------------------------------------------------
    def save(self):
        if len(self.rec) < 200:                       # < 2 с — не сохранять
            return
        t, st, ct, Va, al, be, wv, cc = (np.array(x) for x in zip(*self.rec))
        E = np.array([total_energy(s, self.ac) for s in st])
        log = Log(t=t, state=st, controls=ct, Va=Va, alpha=al,
                  E_kin=E[:, 0], E_pot=E[:, 1], E_total=E[:, 2], wind_vec=wv, beta=be,
                  controls_cmd=cc)
        FlightLogger(scenario="Игра 3D", description="ручной полёт / САУ в реальном времени",
                     aircraft=self.ac, wind_params=self.wp, cfg=self.cfg,
                     trim=(self.alpha_trim, self.de_trim, self.thr_trim),
                     events=self.events).save(log)


def station_info(ac):
    """Постоянные данные ЛА для окна станции: пределы рулей, пороги α, скорости полос ленты
    скорости — горизонтальный полёт с n = 1: Va = √(2mg / (ρ S CL(α))) при alpha_warning и alpha_crit."""
    V = lambda a: float(np.sqrt(2 * ac.mass * ac.g / (ac.rho * ac.S * coef_CL(a, 0.0, 0.0, 16.0, ac))))
    deg = lambda x: float(np.degrees(x))
    return {"name": ac.name, "de_min": deg(ac.delta_e_min), "de_max": deg(ac.delta_e_max),
            "da_max": deg(ac.delta_a_max), "dr_max": deg(ac.delta_r_max),
            "alpha_warn": deg(ac.alpha_warning), "alpha_crit": deg(ac.alpha_crit),
            "V_stall": V(ac.alpha_crit), "V_warn": V(ac.alpha_warning)}


def start_station(terrain, aircraft, h_new, geometry, px=720):
    """Окно станции (viz/station.py) в отдельном процессе; возвращает (q_cmd, q_tel).
    px — сторона карты в окне, пикс (фон — в station.ZOOM_MAX раз крупнее, для масштаба)."""
    q_cmd, q_tel = mp.Queue(maxsize=100), mp.Queue(maxsize=200)
    mp.Process(target=station.run, daemon=True,
               args=(q_cmd, q_tel, terrain.save_top_image(px * station.ZOOM_MAX),
                     (terrain.map_x, terrain.map_z), station_info(aircraft), geometry, h_new)).start()
    return q_cmd, q_tel


def main():
    ap = argparse.ArgumentParser(description="Пилотирование ЛА в 3D (Ursina)")
    ap.add_argument("--wind-n", type=float, default=0.0, help="ветер на север, м/с")
    ap.add_argument("--wind-e", type=float, default=0.0, help="ветер на восток, м/с")
    ap.add_argument("--turb", default="off", choices=TURB_LEVELS,
                    help="турбулентность Драйдена (табл. 4.1 B&M); в полёте — клавиша T")
    ap.add_argument("--turb-alt", default="low", choices=["low", "medium"],
                    help="параметры турбулентности: для 50 м (low) или 600 м (medium)")
    ap.add_argument("--sensors", default="real", choices=["truth", "real"],
                    help="датчики САУ: истина или реалистичный набор (частоты, задержки, дрейф)")
    ap.add_argument("--est", default="ekf", choices=["ekf", "none"],
                    help="оценка состояния при --sensors real: EKF или «сырые» измерения")
    ap.add_argument("--h0", type=float, default=100.0, help="начальная высота, м")
    ap.add_argument("--cam", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--ap", action="store_true", help="старт с включённой САУ")
    ap.add_argument("--shot", default=None, help="скриншот через ~300 кадров и выход")
    ap.add_argument("--map", default="default", choices=list(MAPS), help="карта мира")
    ap.add_argument("--no-map", action="store_true", help="без окна станции (PFD + карта)")
    ap.add_argument("--aircraft", default="fpv", choices=list(AIRCRAFT_TYPES), help="тип ЛА")
    ap.add_argument("--bench", type=float, default=None, help="замер FPS: выход через N с")
    a = ap.parse_args()

    aircraft = aircraft_by_name(a.aircraft)
    wind_params = WindParams(Vw_const=a.wind_n, Vw_cross=a.wind_e,
                             turb_level=a.turb, turb_alt=a.turb_alt)
    cfg = SimConfig(Va0=16.0, h0=a.h0, theta0=0.0, dt=0.01, t_end=1e9)

    app, terrain = make_app(f"3D: пилотирование — {MAPS[a.map].title}", a.map)
    q_station = None if (a.no_map or a.shot) else start_station(terrain, aircraft, a.h0, "-0+0")
    game = Game(aircraft, wind_params, cfg, terrain, cam=a.cam, ap=a.ap, shot=a.shot,
                q_station=q_station, bench=a.bench, sensors=a.sensors, est=a.est)
    try:
        app.run()
    finally:
        print(game.view.perf.summary())
        game.save()


if __name__ == "__main__":
    main()
