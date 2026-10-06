# -*- coding: utf-8 -*-
"""
Bench3D — испытательный стенд САУ для студентов: 3D-полёт в реальном времени
+ отдельное окно с блок-схемой САУ, где параметры крутятся ползунками на лету.

Сейчас на схеме — боковой канал по крену (LateralController):
    χ_ref → [ПИ курса] → [огр. φ_ref] → [ПИ крена − Kd·p] → [огр. δa] → ЛА
Продольный канал работает сам: удержание высоты и скорости (как в GameScenario3D).
Руль направления — штатный контур β (на схеме не показан).

Окно схемы — viz/bench_panel.py, отдельный процесс (tkinter); связь — очереди.
Расчёт САУ и физики — только здесь, в 3D-процессе.

3D-окно (клавиши как в GameScenario3D, САУ включена с удержанием высоты):
  A / D  — уставка курса χ_ref ∓10°    (кнопки ступенек — в окне схемы)
  W / S  — уставка высоты ∓10 м        X / Z — Va_ref ±1 м/с
  P      — САУ вкл/выкл (выкл — ручное управление, контур разомкнут)
  Пробел — пауза, R — сброс, 1–4 — камера, Esc — выход

Запуск:
    python scenarios/Bench3D.py
    python scenarios/Bench3D.py --wind-e 5 --map kainki
"""

import sys
import os
import argparse
import dataclasses
import queue
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from sim.config import AircraftParams, WindParams, SimConfig
from sim.state import H, P, PHI, earth_velocity
from control.controllers import LateralControlParams, wrap_angle
from viz.viewer3d import make_app, MAPS
from ursina import window, Vec2
from viz import bench_panel
from GameScenario3D import Game, _wrap180

WIN_3D = (900, 720)               # 3D-окно у левого края экрана, окно схемы — у правого


def param_spec(aircraft):
    """Параметры на схеме: (ключ, подпись, мин, макс, шаг, по умолчанию). Углы — в градусах."""
    d = LateralControlParams()
    return [
        ("chi_Kp", "Kp", 0.0, 10.0, 0.05, d.chi_Kp),
        ("chi_Ki", "Ki", 0.0, 5.0, 0.05, d.chi_Ki),
        ("phi_ref_max", "±φ_max, °", 5.0, 60.0, 1.0, round(np.degrees(d.phi_ref_max))),
        ("phi_Kp", "Kp", 0.0, 3.0, 0.01, d.phi_Kp),
        ("phi_Ki", "Ki", 0.0, 1.0, 0.01, d.phi_Ki),
        ("phi_Kd", "Kd", 0.0, 0.5, 0.005, d.phi_Kd),
        ("da_max", "±δa_max, °", 2.0, 25.0, 1.0, round(np.degrees(aircraft.delta_a_max))),
    ]


def apply_params(lat, v):
    """Параметры из окна схемы → LateralController (на лету, интегралы не сбрасываются)."""
    lat.pid_chi.Kp, lat.pid_chi.Ki = v["chi_Kp"], v["chi_Ki"]
    lat.params.phi_ref_max = np.radians(v["phi_ref_max"])
    lat.pid_phi.Kp, lat.pid_phi.Ki = v["phi_Kp"], v["phi_Ki"]
    lat.params.phi_Kd = v["phi_Kd"]
    lat.aircraft.delta_a_max = np.radians(v["da_max"])   # копия параметров ЛА, только для САУ


class Bench(Game):

    HELP = ("A/D χ_ref -/+10°  W/S h_ref  X/Z Va_ref  P САУ вкл/выкл  "
            "Пробел пауза  R сброс  1-4 камера  Esc выход")

    def __init__(self, aircraft, wind_params, cfg, terrain, q_par, q_tel, **kw):
        self.q_par, self.q_tel = q_par, q_tel
        super().__init__(aircraft, wind_params, cfg, terrain, **kw)
        # Ограничение δa на стенде меняет только САУ, не модель ЛА
        self.lat.aircraft = dataclasses.replace(aircraft)

    def reset(self):
        super().reset()
        self.chi_unwrap = None             # непрерывный χ для осциллографа
        self._toggle_ap()                  # стенд стартует с САУ
        self.h_hold, self.h_ref = True, float(self.state[H])

    def update(self):
        self._poll_panel()
        super().update()
        self._telemetry()

    def _poll_panel(self):
        while True:
            try:
                kind, val = self.q_par.get_nowait()
            except queue.Empty:
                return
            if kind == "par":
                apply_params(self.lat, val)
            elif kind == "chi" and self.ap:
                self.chi_ref = wrap_angle(self.chi_ref + np.radians(val))

    def _telemetry(self):
        s, deg = self.state, np.degrees
        Vx, Vy, _ = earth_velocity(s)
        chi = np.arctan2(Vy, Vx)
        if self.chi_unwrap is None:
            self.chi_unwrap = chi
        self.chi_unwrap += wrap_angle(chi - self.chi_unwrap)
        e_chi = wrap_angle(self.chi_ref - chi)
        phi = np.radians(_wrap180(deg(s[PHI])))
        msg = {
            "t": self.t, "ap": self.ap,
            "chi": deg(self.chi_unwrap), "chi_ref": deg(self.chi_unwrap + e_chi),
            "e_chi": deg(e_chi),
            "phi_ref": deg(self.lat.phi_ref), "phi": deg(phi),
            "e_phi": deg(self.lat.phi_ref - phi),
            "p": deg(s[P]), "kdp": deg(self.lat.params.phi_Kd * s[P]),
            "da": deg(self.controls[2]),
        }
        try:
            self.q_tel.put_nowait(msg)
        except queue.Full:                 # окно схемы не успевает / закрыто — не ждём
            pass


def main():
    ap = argparse.ArgumentParser(description="Стенд САУ: 3D + блок-схема с параметрами")
    ap.add_argument("--wind-n", type=float, default=0.0, help="ветер на север, м/с")
    ap.add_argument("--wind-e", type=float, default=0.0, help="ветер на восток, м/с")
    ap.add_argument("--h0", type=float, default=200.0, help="начальная высота, м")
    ap.add_argument("--cam", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--map", default="default", choices=list(MAPS), help="карта мира")
    a = ap.parse_args()

    aircraft = AircraftParams()
    wind_params = WindParams(Vw_const=a.wind_n, Vw_cross=a.wind_e)
    cfg = SimConfig(Va0=30.0, h0=a.h0, theta0=0.0, dt=0.01, t_end=1e9)

    q_par, q_tel = mp.Queue(maxsize=100), mp.Queue(maxsize=200)
    panel = mp.Process(target=bench_panel.run, daemon=True,
                       args=(q_par, q_tel, param_spec(aircraft), "-0+0"))
    panel.start()

    app, terrain = make_app(f"Стенд САУ — {MAPS[a.map].title}", a.map, size=WIN_3D)
    window.position = Vec2(0, 40)          # заголовок окна виден — можно перетащить
    game = Bench(aircraft, wind_params, cfg, terrain, q_par, q_tel, cam=a.cam)
    try:
        app.run()
    finally:
        game.save()


if __name__ == "__main__":
    main()
