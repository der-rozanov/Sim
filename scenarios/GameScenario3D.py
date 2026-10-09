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
  ↑ / ↓  — триммер руля высоты ±0.3° (↑ — нос вниз)
  X / Z  — тяга ±5 %

САУ (P — вкл/выкл; при включении захватывает текущие θ, Va, путевой угол χ):
  W / S  — уставка тангажа θ_ref ∓1°   (в режиме H — уставка высоты ∓10 м)
  A / D  — уставка путевого угла χ_ref ∓10°
  X / Z  — уставка скорости Va_ref ±1 м/с
  H      — удержание высоты (захват текущей)
  N      — навигация по маршруту вкл/выкл (маршрут — в окне карты; уставки χ_ref
           и h_ref даёт WaypointNavigator, control/navigation.py). Ручные уставки
           W/S/A/D навигацию выключают.
САУ — control/sau.py (тот же класс, что в s12–s14): χ → φ_ref → δa, β → δr,
h → θ_ref → δe, Va → δt.

Окно карты (viz/map_panel.py, отдельный процесс): вид сверху, ЛА и след; точки
маршрута ставятся мышью, «Лететь по маршруту» — САУ ведёт по точкам.
Измерения в САУ — истинные (без шума), как в исходном GameScenario.

Общие: Пробел — пауза, R — сброс, 1–4 — камера, M — масштаб силуэта, Esc — выход.

Земля твёрдая (sim/ground.py, РЕШ-25): поверхность карты — World.surface(), как
нарисована, с ВПП и покрытиями. ЛА садится на шасси, катится (Q/E — носовое
колесо), взлетает. Авария: касание конструкцией (винт, крыло, хвост), касание
колесом быстрее gear_v_crash (поломка шасси), касание воды.

Запуск:
    python scenarios/GameScenario3D.py
    python scenarios/GameScenario3D.py --wind-n -5 --wind-e 3   # ветер, м/с
    python scenarios/GameScenario3D.py --ap --cam 3              # старт с включённой САУ
    python scenarios/GameScenario3D.py --no-map                  # без окна карты
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

from sim.config import WindParams, SimConfig, AIRCRAFT_TYPES, aircraft_by_name
from sim.integrators import step_rk4
from sim.actuators import actuator_init, actuator_step
from sim.ground import contacts, near_ground
from sim.wind import wind as _wind
from sim.state import (air_data, earth_velocity, total_energy, full_controls, canonical_euler,
                       U, W, THETA, X, H, PHI, PSI, Y, N_STATES, N_CONTROLS)
from runner import Log, compute_trim, trim_state
from flight_logger import FlightLogger
from control.controllers import PitchControlParams, wrap_angle
from control.navigation import WaypointNavigator, NavParams
from control.sau import SAU, SAUParams, TruthSensors, Mode
from viz import map_panel
from viz.viewer3d import (View3D, make_app, MAPS, ned_to_u, body_axes,
                          screenshot_and_quit)
from ursina import Entity, application, held_keys, time as utime

# ------------------------------------------------------------------
# Параметры
# ------------------------------------------------------------------
# Отклонения подобраны по отклику модели (разомкнутый контур, Va = 30 м/с, 1 с удержания):
#   δa 3° → p ≈ 37°/с, φ ≈ 32°;  δe 2° → q ≈ 8°/с;  δr 3° → r ≈ 9°/с
DE_DEFL   = np.radians(2.0)     # отклонение руля высоты от триммера при нажатой W/S
DA_DEFL   = np.radians(3.0)     # элероны при A/D
DR_DEFL   = np.radians(3.0)     # руль направления при Q/E
SURF_RATE = np.radians(20.0)    # темп отклонения «ручки» с клавиатуры (команда; привод — sim/actuators.py)
TRIM_STEP = np.radians(0.3)
# Клавиши ручного пилотирования: при потере фокуса 3D-окном Ursina не получает
# «клавиша отпущена» — held_keys остаются 1 и рули «заклинивает» (полёт 2026-10-08)
PILOT_KEYS = ("w", "s", "a", "d", "q", "e")
THR_STEP  = 0.05

THETA_STEP, THETA_LIM = np.radians(1.0), np.radians(20.0)
CHI_STEP  = np.radians(10.0)
VA_STEP, VA_MIN, VA_MAX = 1.0, 10.0, 30.0   # FPV: сваливание 7.5, максимум ≈31.5 м/с
H_STEP, H_MIN, H_MAX = 10.0, 20.0, 1000.0
KH        = PitchControlParams().KH   # рад/м, расчёт control/tuning.py

MAX_STEPS_PER_FRAME = 10        # при просадке кадров — замедление, а не «рывок»
TRAIL_MAX = 3000                # точек следа (0.1 с) — последние 5 мин


def _wrap180(a):
    """Угол в градусах → [−180, 180) (крен копится при бочках)."""
    return (a + 180.0) % 360.0 - 180.0


class Game(Entity):

    HELP = ("W/S δe  A/D δa  Q/E δr  ↑/↓ трим  X/Z газ  P САУ  H высота  N маршрут  "
            "Пробел пауза  R сброс  1-4 камера  Esc выход")

    def __init__(self, aircraft, wind_params, cfg, terrain, cam=1, ap=False, shot=None,
                 q_map=None):
        """q_map — (q_cmd, q_tel) окна карты (viz/map_panel.py) или None."""
        self.q_map = q_map
        self.nav = WaypointNavigator(NavParams())
        self.route, self.loop = [], False        # маршрут из окна карты: [(N, E, h), ...]
        super().__init__()
        self.ac, self.wp, self.cfg = aircraft, wind_params, cfg
        self.wind_call = lambda h, t: _wind(h, t, wind_params)
        self.alpha_trim, self.de_trim, self.thr_trim = compute_trim(aircraft, cfg.Va0)
        self.s0 = trim_state(aircraft, cfg)
        self.s0[PSI] = terrain.start_psi          # старт вдоль ВПП карты
        self.ter = terrain
        self.ground = lambda N, E: terrain.surface(E, N)    # NED → Ursina: X — восток, Z — север
        self.view = View3D(aircraft.b, aircraft.c, terrain, self.HELP, cam=cam, hud_lines=13,
                           aircraft_name=aircraft.name)

        self.sau = self._make_sau()

        self.rec = []                  # (t, state, controls, Va, alpha, beta, wind, cmd)
        self.events = []
        self.shot, self._frames = shot, 0
        self.reset()
        if ap:
            self._toggle_ap()

    def _make_sau(self):
        """САУ на истинных измерениях; стенд Bench3D подменяет её своей (защита по α)."""
        params = SAUParams(pitch=PitchControlParams(Va_ref=self.cfg.Va0),
                           h_min=H_MIN, h_max=H_MAX)
        return SAU(self.ac, (self.alpha_trim, self.de_trim, self.thr_trim),
                   TruthSensors(), params, nav=self.nav)

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
        self.paused, self.crashed = False, False
        self.on_ground, self.crash_msg = False, ""
        self.ap = False
        self.de_trim_man = self.de_trim
        self.thr = self.thr_trim
        self.surf = np.array([self.de_trim, 0.0, 0.0])        # δe, δa, δr (ручной)
        self.controls = actuator_init([self.de_trim, self.thr_trim, 0.0, 0.0], self.ac)
        self.cmd = self.controls.copy()                       # команда до приводов рулей
        self.sau.set_mode(Mode.PITCH)
        self.sau.engage(theta=self.alpha_trim, h=self.cfg.h0, Va=self.cfg.Va0, chi=0.0)
        self.trail = []

    def _event(self, label, color="orange"):
        self.events.append({"t": round(self.t, 2), "label": label, "color": color})

    def _toggle_ap(self):
        if self.ap:
            self._nav_off()
            self.ap = False
            self.de_trim_man = self.controls[0]          # без рывка при выключении
            self.surf = np.array([self.controls[0], 0.0, 0.0])
            self.thr = self.controls[1]
            self._event("САУ выкл", "gray")
            return
        self.ap = True
        s = self.state
        Va, _, _ = air_data(s, self.wind_call(s[H], self.t))
        Vx, Vy, _ = earth_velocity(s)
        self.sau.set_mode(Mode.PITCH)
        self.sau.engage(theta=float(np.clip(s[THETA], -THETA_LIM, THETA_LIM)), h=float(s[H]),
                        Va=float(np.clip(Va, VA_MIN, VA_MAX)), chi=float(np.arctan2(Vy, Vx)))
        self._event("САУ вкл", "dodgerblue")

    # --- навигация по маршруту ----------------------------------------------
    def _nav_go(self):
        """Лететь по маршруту с первой точки (САУ включается, если выключена)."""
        if not self.route or self.crashed:
            return
        if not self.ap:
            self._toggle_ap()
        s = self.state
        self.sau.fly_route(self.route, start=(s[X], s[Y], s[H]))
        self._event("маршрут: старт", "dodgerblue")

    def _nav_off(self, why="маршрут: стоп"):
        """Навигация выкл: САУ держит текущие путевой угол и высоту."""
        if not self.nav_on:
            return
        Vx, Vy, _ = earth_velocity(self.state)
        self.sau.refs.chi = float(np.arctan2(Vy, Vx))
        self.sau.refs.h = float(np.clip(self.state[H], H_MIN, H_MAX))
        self.sau.set_mode(Mode.ALTITUDE)
        self._event(why, "gray")

    def _poll_map(self):
        if self.q_map is None:
            return
        while True:
            try:
                kind, val = self.q_map[0].get_nowait()
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

    def _map_telemetry(self):
        if self.q_map is None:
            return
        s, nav = self.state, self.nav
        Va = air_data(s, self.wind_call(s[H], self.t))[0]
        tgt = nav.target if self.nav_on else None
        msg = {"t": self.t, "n": s[X], "e": s[Y], "h": s[H], "psi": s[PSI], "Va": Va,
               "ap": self.ap, "nav": self.nav_on, "idx": nav.idx, "mode": nav.mode,
               "circle": (tuple(nav.circle[:3])
                          if self.nav_on and nav.mode in ("fillet", "orbit") else None),
               "dist": float(np.hypot(tgt[0] - s[X], tgt[1] - s[Y])) if tgt is not None else 0.0}
        try:
            self.q_map[1].put_nowait(msg)
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
        elif key == "n":
            if self.nav_on:
                self._nav_off()
            else:
                self._nav_go()
        elif key == "h" and self.ap:                        # высота ⇄ тангаж (маршрут — стоп)
            self._nav_off("маршрут: ручная уставка")
            to_alt = self.sau.mode is Mode.PITCH
            self.sau.refs.h = float(self.state[H])
            self.sau.set_mode(Mode.ALTITUDE if to_alt else Mode.PITCH)
            self._event("удержание h" if to_alt else "θ_ref", "dodgerblue")
        elif base in ("x", "z"):
            sgn = 1 if base == "x" else -1
            if self.ap:
                refs = self.sau.refs
                refs.Va = float(np.clip(refs.Va + sgn * VA_STEP, VA_MIN, VA_MAX))
            else:
                self.thr = float(np.clip(self.thr + sgn * THR_STEP, 0.0, 1.0))
        elif self.ap and base in ("w", "s"):
            self._nav_off("маршрут: ручная уставка")
            sgn, refs = (1 if base == "s" else -1), self.sau.refs     # S — вверх
            if self.h_hold:
                refs.h = float(np.clip(refs.h + sgn * H_STEP, H_MIN, H_MAX))
            else:
                refs.theta = float(np.clip(refs.theta + sgn * THETA_STEP,
                                           -THETA_LIM, THETA_LIM))
        elif self.ap and base in ("a", "d"):
            self._nav_off("маршрут: ручная уставка")
            sgn = 1 if base == "d" else -1                  # D — вправо
            self.sau.refs.chi = wrap_angle(self.sau.refs.chi + sgn * CHI_STEP)
        elif not self.ap and base in ("up arrow", "down arrow"):
            sgn = 1 if base == "up arrow" else -1           # ↑ — нос вниз (δe > 0)
            self.de_trim_man = float(np.clip(self.de_trim_man + sgn * TRIM_STEP,
                                             self.ac.delta_e_min, self.ac.delta_e_max))

    # --- управление на шаг ----------------------------------------------------
    def _manual(self, dt):
        cmd = np.array([
            self.de_trim_man + DE_DEFL * (held_keys["w"] - held_keys["s"]),
            DA_DEFL * (held_keys["d"] - held_keys["a"]),
            DR_DEFL * (held_keys["q"] - held_keys["e"]),
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
        self._focus_check()
        self._poll_map()
        dt = self.cfg.dt
        if not (self.paused or self.crashed):
            self.acc += utime.dt
            n = int(self.acc / dt)
            if n > MAX_STEPS_PER_FRAME:
                n, self.acc = MAX_STEPS_PER_FRAME, 0.0
            else:
                self.acc -= n * dt
            for _ in range(n):
                self._step()
                if self.crashed:
                    break

        s = self.state
        pos = ned_to_u(s[X], s[Y], -s[H])
        x_b, z_b = body_axes(s[PHI], s[THETA], s[PSI])
        nose, up = ned_to_u(*x_b), ned_to_u(*(-z_b))
        c = self.controls
        self.view.render(pos, nose, up, c[0], c[2], c[3], c[1],
                         np.array(self.trail), utime.dt)
        self._hud()
        self._map_telemetry()

        self._frames += 1
        if self.shot and self._frames == 300:
            screenshot_and_quit(self.shot)

    def _step(self):
        s, dt = self.state, self.cfg.dt
        w_vec = self.wind_call(s[H], self.t)
        Va, alpha, beta = air_data(s, w_vec)
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
            self._crash("УДАР О ЗЕМЛЮ")
        elif wet:
            self._crash("ПРИВОДНЕНИЕ")
        elif v_sink > self.ac.gear_v_crash:
            self._crash(f"ПОЛОМКА ШАССИ ({v_sink:.1f} м/с)")
        elif not self.on_ground:
            self._event(f"касание {v_sink:.1f} м/с", "green")
        self.on_ground = True

    def _crash(self, msg):
        self.crashed, self.crash_msg = True, msg
        self._event(msg.lower(), "red")

    def _hud(self):
        s, c = self.state, self.controls
        w_vec = self.wind_call(s[H], self.t)
        Va, alpha, beta = air_data(s, w_vec)
        Vx, Vy, _ = earth_velocity(s)
        deg = np.degrees
        refs = self.sau.refs
        if self.ap:
            ref = (f"h_ref {refs.h:6.0f} м" if self.h_hold
                   else f"θ_ref {deg(refs.theta):5.1f}°")
            nav = ""
            if self.nav_on:
                nav = ("  НАВ: кружение" if self.nav.mode == "orbit" else
                       f"  НАВ → точка {self.nav.idx + 1}/{len(self.route)}")
            mode = (f"САУ  {ref}  Va_ref {refs.Va:4.1f}\n"
                    f"     χ_ref {deg(refs.chi) % 360:5.1f}°{nav}")
        else:
            mode = f"РУЧНОЙ  трим δe {deg(self.de_trim_man):5.2f}°\n"
        state = "ПАУЗА" if self.paused else ("НА ЗЕМЛЕ" if self.on_ground else "")
        text = (
            f"{mode}\n"
            f"t   {self.t:7.2f} с   {state}\n"
            f"Va  {Va:7.2f} м/с" + (f"   уставка {refs.Va:4.1f}\n" if self.ap else "\n") +
            f"газ δt {c[1]:5.2f}  ({100 * c[1]:3.0f} %)\n"
            f"h   {s[H]:7.1f} м\n"
            f"α   {deg(alpha):7.2f}°    β  {deg(beta):6.2f}°\n"
            f"φ   {_wrap180(deg(s[PHI])):7.2f}°    θ  {deg(s[THETA]):6.2f}°\n"
            f"ψ   {deg(s[PSI]) % 360:7.1f}°    χ  {deg(np.arctan2(Vy, Vx)) % 360:6.1f}°\n"
            f"δe  {deg(c[0]):7.2f}°    δa {deg(c[2]):6.2f}°\n"
            f"δr  {deg(c[3]):7.2f}°\n"
            f"ветер С {w_vec[0]:5.1f}  В {w_vec[2]:5.1f}  верт {w_vec[1]:5.1f} м/с"
        )
        al = [self.ac.alpha_warning, self.ac.alpha_crit, self.ac.alpha_stall]
        alert = int(sum(alpha >= x for x in al))
        msg = f"{self.crash_msg} — R для сброса" if self.crashed else ""
        self.view.set_hud(text, alert, msg)

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


def start_map_window(terrain, h_new, geometry, px=720):
    """Окно карты (viz/map_panel.py) в отдельном процессе; возвращает (q_cmd, q_tel).
    px — ширина карты в окне, пикс (фон — в map_panel.ZOOM_MAX раз крупнее, для масштаба)."""
    q_cmd, q_tel = mp.Queue(maxsize=100), mp.Queue(maxsize=200)
    mp.Process(target=map_panel.run, daemon=True,
               args=(q_cmd, q_tel, terrain.save_top_image(px * map_panel.ZOOM_MAX), (terrain.map_x, terrain.map_z),
                     geometry, h_new)).start()
    return q_cmd, q_tel


def main():
    ap = argparse.ArgumentParser(description="Пилотирование ЛА в 3D (Ursina)")
    ap.add_argument("--wind-n", type=float, default=0.0, help="ветер на север, м/с")
    ap.add_argument("--wind-e", type=float, default=0.0, help="ветер на восток, м/с")
    ap.add_argument("--h0", type=float, default=100.0, help="начальная высота, м")
    ap.add_argument("--cam", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--ap", action="store_true", help="старт с включённой САУ")
    ap.add_argument("--shot", default=None, help="скриншот через ~300 кадров и выход")
    ap.add_argument("--map", default="default", choices=list(MAPS), help="карта мира")
    ap.add_argument("--no-map", action="store_true", help="без окна карты (маршрута)")
    ap.add_argument("--aircraft", default="fpv", choices=list(AIRCRAFT_TYPES), help="тип ЛА")
    a = ap.parse_args()

    aircraft = aircraft_by_name(a.aircraft)
    wind_params = WindParams(Vw_const=a.wind_n, Vw_cross=a.wind_e)
    cfg = SimConfig(Va0=16.0, h0=a.h0, theta0=0.0, dt=0.01, t_end=1e9)

    app, terrain = make_app(f"3D: пилотирование — {MAPS[a.map].title}", a.map)
    q_map = None if (a.no_map or a.shot) else start_map_window(terrain, a.h0, "-0+0")
    game = Game(aircraft, wind_params, cfg, terrain, cam=a.cam, ap=a.ap, shot=a.shot,
                q_map=q_map)
    try:
        app.run()
    finally:
        game.save()


if __name__ == "__main__":
    main()
