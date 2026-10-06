# -*- coding: utf-8 -*-
"""
Bench3D — испытательный стенд САУ для студентов: 3D-полёт в реальном времени
+ отдельное окно с блок-схемами САУ, где параметры крутятся ползунками на лету.

Вкладки окна схемы (viz/bench_panel.py):
  Крен            χ_ref → [ПИ курса] → [огр. φ_ref] → [ПИ крена − Kd·p] → [огр. δa]
  Тангаж / высота h_ref → [П высоты KH] → θ_ref → [ПИД θ] → [огр. q_ref] → [ПИД q]
                  → [× −(Va₀/Va)²] → [огр. δe]
  Скорость        Va_ref → [ПИД Va] + δt_трим → [огр. 0…1]
  Рыскание (β)    β = 0 → [ПИ β] → [× −1] → [огр. δr]
Регуляторы — штатные PitchController, SpeedController, LateralController
(как в GameScenario3D); по умолчанию параметры штатные — летает.

Окно схемы — отдельный процесс (tkinter); связь — очереди.
Расчёт САУ и физики — только здесь, в 3D-процессе.

3D-окно (клавиши как в GameScenario3D, САУ включена с удержанием высоты):
  A / D  — уставка курса χ_ref ∓10°    (кнопки ступенек и толчков — в окне схемы)
  W / S  — уставка высоты ∓10 м        X / Z — Va_ref ±1 м/с
  P      — САУ вкл/выкл (выкл — ручное управление, контуры разомкнуты)
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
from sim.state import H, P, Q, PHI, THETA, air_data, earth_velocity
from control.controllers import (LateralControlParams, PitchControlParams,
                                 SpeedControlParams, wrap_angle)
from viz.viewer3d import make_app, MAPS
from ursina import window, Vec2
from viz import bench_panel
from GameScenario3D import Game, _wrap180, KH, H_MIN, H_MAX, VA_MIN, VA_MAX

WIN_3D = (900, 720)               # 3D-окно у левого края экрана, окно схемы — у правого

# Толчок рулём из окна схемы: добавка к команде САУ на KICK_T секунд
# (возмущение на входе ЛА — САУ должна его отработать)
KICK = {"de": (0, np.radians(3.0)), "da": (2, np.radians(5.0)), "dr": (3, np.radians(5.0))}
KICK_T = 1.0


def param_spec(aircraft):
    """Параметры на схемах: (вкладка, ключ, подпись, мин, макс, шаг, по умолчанию).
    Углы — в градусах; шаг None — флажок вкл/выкл."""
    lat, pit, spd = LateralControlParams(), PitchControlParams(), SpeedControlParams()
    deg = lambda x: round(float(np.degrees(x)))
    return [
        ("roll", "chi_Kp", "Kp", 0.0, 10.0, 0.05, lat.chi_Kp),
        ("roll", "chi_Ki", "Ki", 0.0, 5.0, 0.05, lat.chi_Ki),
        ("roll", "phi_ref_max", "±φ_max, °", 5.0, 60.0, 1.0, deg(lat.phi_ref_max)),
        ("roll", "phi_Kp", "Kp", 0.0, 3.0, 0.01, lat.phi_Kp),
        ("roll", "phi_Ki", "Ki", 0.0, 1.0, 0.01, lat.phi_Ki),
        ("roll", "phi_Kd", "Kd", 0.0, 0.5, 0.005, lat.phi_Kd),
        ("roll", "da_max", "±δa_max, °", 2.0, 25.0, 1.0, deg(aircraft.delta_a_max)),

        ("pitch", "KH", "KH, рад/м", 0.0, 0.03, 0.0005, KH),
        ("pitch", "theta_Kp", "Kp", 0.0, 5.0, 0.05, pit.theta_Kp),
        ("pitch", "theta_Ki", "Ki", 0.0, 1.0, 0.01, pit.theta_Ki),
        ("pitch", "theta_Kd", "Kd", 0.0, 1.0, 0.01, pit.theta_Kd),
        ("pitch", "q_max", "±q_max, °/с", 5.0, 90.0, 1.0, deg(pit.q_max)),
        ("pitch", "q_Kp", "Kp", 0.0, 2.0, 0.01, pit.q_Kp),
        ("pitch", "q_Ki", "Ki", 0.0, 0.5, 0.005, pit.q_Ki),
        ("pitch", "q_Kd", "Kd", 0.0, 0.5, 0.005, pit.q_Kd),
        ("pitch", "gs", "GS", None, None, None, pit.gain_scheduling),
        ("pitch", "de_max", "±δe_max, °", 2.0, 25.0, 1.0, deg(aircraft.delta_e_max)),

        ("speed", "Va_Kp", "Kp", 0.0, 1.0, 0.01, spd.Va_Kp),
        ("speed", "Va_Ki", "Ki", 0.0, 0.3, 0.005, spd.Va_Ki),
        ("speed", "Va_Kd", "Kd", 0.0, 0.1, 0.005, spd.Va_Kd),

        ("yaw", "beta_hold", "контур β вкл", None, None, None, lat.beta_hold),
        ("yaw", "beta_Kp", "Kp", 0.0, 3.0, 0.05, lat.beta_Kp),
        ("yaw", "beta_Ki", "Ki", 0.0, 3.0, 0.05, lat.beta_Ki),
        ("yaw", "dr_max", "±δr_max, °", 2.0, 25.0, 1.0, deg(aircraft.delta_r_max)),
    ]


def apply_params(g, v):
    """Параметры из окна схемы → регуляторы стенда g (.lat, .pitch, .speed, .KH).
    На лету, интегралы не сбрасываются. Ограничения рулей меняются в копиях
    параметров ЛА у регуляторов — модель ЛА не трогается."""
    lat, pit, spd, r = g.lat, g.pitch, g.speed, np.radians
    lat.pid_chi.Kp, lat.pid_chi.Ki = v["chi_Kp"], v["chi_Ki"]
    lat.params.phi_ref_max = r(v["phi_ref_max"])
    lat.pid_phi.Kp, lat.pid_phi.Ki = v["phi_Kp"], v["phi_Ki"]
    lat.params.phi_Kd = v["phi_Kd"]
    lat.aircraft.delta_a_max = r(v["da_max"])

    g.KH = v["KH"]
    pit.pid_theta.Kp, pit.pid_theta.Ki, pit.pid_theta.Kd = v["theta_Kp"], v["theta_Ki"], v["theta_Kd"]
    pit.q_max, pit.q_min = r(v["q_max"]), -r(v["q_max"])
    pit.pid_q.Kp, pit.pid_q.Ki, pit.pid_q.Kd = v["q_Kp"], v["q_Ki"], v["q_Kd"]
    pit.gain_scheduling = bool(v["gs"])
    pit.aircraft.delta_e_max, pit.aircraft.delta_e_min = r(v["de_max"]), -r(v["de_max"])

    spd.pid_Va.Kp, spd.pid_Va.Ki, spd.pid_Va.Kd = v["Va_Kp"], v["Va_Ki"], v["Va_Kd"]

    lat.params.beta_hold = bool(v["beta_hold"])
    lat.pid_beta.Kp, lat.pid_beta.Ki = v["beta_Kp"], v["beta_Ki"]
    lat.aircraft.delta_r_max = r(v["dr_max"])


class Bench(Game):

    HELP = ("A/D χ_ref -/+10°  W/S h_ref  X/Z Va_ref  P САУ вкл/выкл  "
            "Пробел пауза  R сброс  1-4 камера  Esc выход")

    def __init__(self, aircraft, wind_params, cfg, terrain, q_par, q_tel, **kw):
        self.q_par, self.q_tel = q_par, q_tel
        super().__init__(aircraft, wind_params, cfg, terrain, **kw)
        # Ограничения рулей на стенде меняет только САУ, не модель ЛА
        self.lat.aircraft = dataclasses.replace(aircraft)
        self.pitch.aircraft = dataclasses.replace(aircraft)

    def reset(self):
        super().reset()
        self.chi_unwrap = None             # непрерывный χ для осциллографа
        self.kick = None                   # (индекс руля, добавка, t_конца)
        self._toggle_ap()                  # стенд стартует с САУ
        self.h_hold, self.h_ref = True, float(self.state[H])

    def update(self):
        self._poll_panel()
        super().update()
        self._telemetry()

    def _sau(self, s, Va, beta, dt):
        c = super()._sau(s, Va, beta, dt)
        if self.kick and self.t < self.kick[2]:
            c[self.kick[0]] += self.kick[1]
        return c

    def _poll_panel(self):
        while True:
            try:
                kind, val = self.q_par.get_nowait()
            except queue.Empty:
                return
            if kind == "par":
                apply_params(self, val)
            elif not self.ap:                  # ступеньки и толчки — только с САУ
                continue
            elif kind == "chi":
                self.chi_ref = wrap_angle(self.chi_ref + np.radians(val))
            elif kind == "h":
                self.h_hold = True
                self.h_ref = float(np.clip(self.h_ref + val, H_MIN, H_MAX))
            elif kind == "va":
                self.Va_ref = float(np.clip(self.Va_ref + val, VA_MIN, VA_MAX))
                self.speed.set_Va_ref(self.Va_ref)
            elif kind == "kick":
                idx, amp = KICK[val]
                self.kick = (idx, amp, self.t + KICK_T)
                self._event(f"толчок {val}", "purple")

    def _telemetry(self):
        s, deg, c = self.state, np.degrees, self.controls
        Va, alpha, beta = air_data(s, self.wind_call(s[H], self.t))
        Vx, Vy, _ = earth_velocity(s)
        chi = np.arctan2(Vy, Vx)
        if self.chi_unwrap is None:
            self.chi_unwrap = chi
        self.chi_unwrap += wrap_angle(chi - self.chi_unwrap)
        e_chi = wrap_angle(self.chi_ref - chi)
        phi = np.radians(_wrap180(deg(s[PHI])))
        pit, spd = self.pitch, self.speed
        h_ref = self.h_ref if self.h_hold else s[H]
        gs = (float(np.clip((pit.Va_ref / max(Va, 1.0)) ** 2, pit.gs_scale_min, pit.gs_scale_max))
              if pit.gain_scheduling else 1.0)
        msg = {
            "t": self.t, "ap": self.ap,
            # крен
            "chi": deg(self.chi_unwrap), "chi_ref": deg(self.chi_unwrap + e_chi),
            "e_chi": deg(e_chi),
            "phi_ref": deg(self.lat.phi_ref), "phi": deg(phi),
            "e_phi": deg(self.lat.phi_ref - phi),
            "p": deg(s[P]), "kdp": deg(self.lat.params.phi_Kd * s[P]), "da": deg(c[2]),
            # тангаж / высота
            "h_ref": h_ref, "h": s[H], "e_h": h_ref - s[H],
            "theta_ref": deg(pit.theta_ref), "theta": deg(s[THETA]),
            "e_theta": deg(pit.theta_ref - s[THETA]),
            "q_ref": deg(pit.q_ref), "q": deg(s[Q]), "e_q": deg(pit.q_ref - s[Q]),
            "gs": gs, "de": deg(c[0]),
            # скорость
            "Va_ref": spd.Va_ref, "Va": Va, "e_Va": spd.Va_ref - Va,
            "thr_trim": spd.trim_throttle, "dthr": c[1] - spd.trim_throttle, "thr": c[1],
            # рыскание
            "beta": deg(beta), "dr": deg(c[3]), "alpha": deg(alpha),
        }
        try:
            self.q_tel.put_nowait(msg)
        except queue.Full:                 # окно схемы не успевает / закрыто — не ждём
            pass


def main():
    ap = argparse.ArgumentParser(description="Стенд САУ: 3D + блок-схемы с параметрами")
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
