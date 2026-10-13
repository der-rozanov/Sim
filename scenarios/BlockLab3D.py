# -*- coding: utf-8 -*-
"""
BlockLab3D — лабораторная «Конструктор САУ»: 3D-полёт в реальном времени + окно
редактора структурных схем (viz/block_editor.py), где студент сам собирает канал
управления из блоков: входы (сигналы ЛА) → регуляторы П / ПИ / ПД / ПИД, сумматоры,
ограничения, звенья → выходы на приводы (δe, δt, δa, δr).

Какой выход на какой привод — выбирает студент (блок «Выход»). Приводы, к которым
схема не подключена, ведёт штатная САУ (control/sau.py: TECS, курс → крен, β → δr).
Каналы — все: тангаж, высота, скорость, крен, курс и скольжение. Меню «Шаблоны»:
«Заготовка: …» — входы и выход канала без регулятора; «Штатная САУ: …» — регуляторы
control/controllers.py, собранные из блоков (совпадение проверяет checks/check_blocks.py);
«все каналы» — вся САУ с раздельными контурами высоты и скорости (B&M), а не TECS.
Уставки на входах схемы — от штатной САУ (θ_ref — после TECS, φ_ref — после контура
курса): так канал можно заменять по одному.

Схема считается всегда, когда принята («Применить»): её осциллографы показывают
сигналы и при выключенной схеме — команды тогда идут от штатной САУ.
Включение схемы («Схема ВКЛ») — с нулевыми состояниями блоков. Ошибка счёта
(NaN, ∞) — схема выключается, рули снова у штатной САУ.

Окно редактора — отдельный процесс (tkinter); связь — очереди:
    q_cmd (редактор → 3D): ("scheme", схема) | ("on", bool) |
                           ("theta" | "h" | "va" | "chi", шаг) | ("mode", "pitch" | "alt") |
                           ("kick", "de" | "da" | "dr")
    q_tel (3D → редактор): dict — см. Lab._telemetry

3D-окно (САУ включена, режим — удержание высоты):
  W / S  — уставка высоты ∓10 м (в режиме тангажа — θ_ref ∓1°)
  A / D  — уставка курса ∓10°      X / Z — Va_ref ±1 м/с
  H      — высота ⇄ тангаж          P — САУ вкл/выкл (выкл — ручное управление)
  Пробел — пауза, R — сброс, 1–4 — камера, Esc — выход

Запуск:
    python scenarios/BlockLab3D.py
    python scenarios/BlockLab3D.py --wind-e 5 --map kainki
    python scenarios/BlockLab3D.py --scheme results/my_pitch.json   # своя схема при старте
    python scenarios/BlockLab3D.py --template "Штатная САУ: все каналы"
"""

import sys
import os
import json
import math
import argparse
import queue
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from sim.config import WindParams, SimConfig, AIRCRAFT_TYPES, aircraft_by_name
from sim.state import H, PHI, THETA, air_data, earth_velocity
from control.controllers import wrap_angle
from control.sau import Mode
from control.blocks import Diagram, SchemeError, catalog, sau_signals, templates
from viz.viewer3d import make_app, MAPS
from viz import block_editor
from ursina import window, Vec2
from GameScenario3D import Game, H_MIN, H_MAX, VA_MIN, VA_MAX, THETA_LIM

WIN_3D = (720, 600)               # 3D-окно у левого края экрана, редактор — справа
# Импульс рулём (прямоугольный): привод → (индекс в [δe, δt, δa, δr], добавка, рад, длительность, с)
KICKS = {"de": (0, math.radians(3.0), 1.0), "da": (2, math.radians(5.0), 0.5),
         "dr": (3, math.radians(5.0), 1.0)}


class Lab(Game):

    HELP = ("W/S h_ref (θ_ref)  A/D χ_ref  X/Z Va_ref  H высота/тангаж  P САУ  "
            "Пробел пауза  R сброс  1-4 камера  Esc выход")

    def __init__(self, aircraft, wind_params, cfg, terrain, q_cmd, q_tel, **kw):
        self.q_cmd, self.q_tel = q_cmd, q_tel
        self.dia, self.scheme_on, self.err = None, False, ""
        self.version = 0                   # номер принятой схемы (редактор сверяет подписи)
        super().__init__(aircraft, wind_params, cfg, terrain, **kw)

    def reset(self):
        super().reset()
        self.kick, self.kick_until = None, -1.0
        if self.dia is not None:
            self.dia.reset()
        self._toggle_ap()                  # стартуем с САУ, удержание высоты
        self.sau.refs.h = float(self.state[H])
        self.sau.set_mode(Mode.ALTITUDE)

    # --- схема ------------------------------------------------------------------
    def _sau(self, s, Va, alpha, beta, dt):
        """Штатная САУ; подключённые схемой приводы — от схемы."""
        c = super()._sau(s, Va, alpha, beta, dt).copy()
        if self.dia is not None:
            try:
                out = self.dia.step(sau_signals(self.sau, s, alpha, beta), self.t, dt)
                cmd = self.dia.commands(out)
                if not all(np.isfinite(v) for v in cmd.values()):
                    raise FloatingPointError("выход схемы не число (NaN / ∞)")
            except (ArithmeticError, ValueError, KeyError) as ex:
                self._scheme_fail(str(ex))
                return c
            if self.scheme_on:
                for idx, v in cmd.items():
                    c[idx] = v
        if self.t < self.kick_until:
            idx, d, _ = KICKS[self.kick]
            c[idx] += d
        return c

    def _scheme_fail(self, msg):
        self.err = f"ошибка счёта, схема выключена: {msg}"
        self.dia, self.scheme_on = None, False
        self._event("схема: ошибка", "red")

    def _set_scheme(self, scheme):
        try:
            dia = Diagram(scheme)
        except SchemeError as ex:
            self.err = str(ex)
            return
        self.dia, self.err = dia, ""
        self.version += 1
        if self.scheme_on:
            self._event("схема: заменена", "dodgerblue")

    def _set_on(self, on):
        if on and self.dia is None:
            self.err = "сначала примените схему"
            return
        if on and not self.scheme_on:
            self.dia.reset()               # включение — с нулевыми состояниями блоков
        if on != self.scheme_on:
            self._event("схема вкл" if on else "схема выкл", "dodgerblue" if on else "gray")
        self.scheme_on = on

    # --- окно редактора -----------------------------------------------------------
    def update(self):
        self._poll_editor()
        super().update()
        self._telemetry()

    def _poll_editor(self):
        while True:
            try:
                kind, val = self.q_cmd.get_nowait()
            except queue.Empty:
                return
            if kind == "scheme":
                self._set_scheme(val)
            elif kind == "on":
                self._set_on(bool(val))
            elif not self.ap:              # уставки и толчки — только с САУ
                continue
            elif kind == "theta":
                self._nav_off("маршрут: ручная уставка")
                refs = self.sau.refs
                if self.sau.mode is not Mode.PITCH:
                    self.sau.set_mode(Mode.PITCH)       # θ_ref подхватит текущую команду
                refs.theta = float(np.clip(refs.theta + math.radians(val), -THETA_LIM, THETA_LIM))
            elif kind == "h":
                self._nav_off("маршрут: ручная уставка")
                if self.sau.mode is Mode.PITCH:
                    self.sau.refs.h = float(self.state[H])
                self.sau.refs.h = float(np.clip(self.sau.refs.h + val, H_MIN, H_MAX))
                self.sau.set_mode(Mode.ALTITUDE)
            elif kind == "va":
                self.sau.refs.Va = float(np.clip(self.sau.refs.Va + val, VA_MIN, VA_MAX))
            elif kind == "chi":
                self._nav_off("маршрут: ручная уставка")
                self.sau.refs.chi = wrap_angle(self.sau.refs.chi + math.radians(val))
            elif kind == "mode":
                self._nav_off("маршрут: ручная уставка")
                if val == "alt":
                    self.sau.refs.h = float(self.state[H])
                self.sau.set_mode(Mode.ALTITUDE if val == "alt" else Mode.PITCH)
            elif kind == "kick" and val in KICKS:
                self.kick, self.kick_until = val, self.t + KICKS[val][2]
                self._event(f"импульс {dict(de='δe', da='δa', dr='δr')[val]}", "purple")

    def _telemetry(self):
        s, deg, sau = self.state, math.degrees, self.sau
        Va, alpha, beta = air_data(s, self.wind_call(s[H], self.t))
        Vx, Vy, _ = earth_velocity(s)
        msg = {
            "t": self.t, "ap": self.ap, "on": self.scheme_on, "err": self.err,
            "version": self.version, "paused": self.paused, "crashed": self.crashed,
            "mode": ("тангаж" if sau.mode is Mode.PITCH else
                     "маршрут" if sau.mode is Mode.ROUTE else "высота") if self.ap else "ручной",
            "theta_ref": deg(sau.theta_ref), "h_ref": sau.refs.h, "Va_ref": sau.refs.Va,
            "h": s[H], "Va": Va, "theta": deg(s[THETA]), "alpha": deg(alpha),
            "chi_ref": deg(sau.refs.chi) % 360.0, "chi": deg(math.atan2(Vy, Vx)) % 360.0,
            "phi_ref": deg(sau.lat.phi_ref), "phi": deg(wrap_angle(s[PHI])), "beta": deg(beta),
            "values": dict(self.dia.values) if self.dia is not None else {},
            "scopes": dict(self.dia.scopes) if self.dia is not None else {},
        }
        try:
            self.q_tel.put_nowait(msg)
        except queue.Full:                 # редактор не успевает / закрыт — не ждём
            pass

    def _hud(self):
        super()._hud()
        acts = sorted(self.dia.outputs) if self.dia is not None else []
        names = {"de": "δe", "dt": "δt", "da": "δa", "dr": "δr"}
        line = ("СХЕМА: " + ", ".join(names[a] for a in acts) if self.scheme_on
                else "СХЕМА выкл — штатная САУ")
        self.view.hud.text = line + "\n" + self.view.hud.text


def main():
    ap = argparse.ArgumentParser(description="Конструктор САУ: 3D + редактор структурных схем")
    ap.add_argument("--wind-n", type=float, default=0.0, help="ветер на север, м/с")
    ap.add_argument("--wind-e", type=float, default=0.0, help="ветер на восток, м/с")
    ap.add_argument("--h0", type=float, default=200.0, help="начальная высота, м")
    ap.add_argument("--cam", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--map", default="default", choices=list(MAPS), help="карта мира")
    ap.add_argument("--aircraft", default="fpv", choices=list(AIRCRAFT_TYPES), help="тип ЛА")
    ap.add_argument("--scheme", default=None, help="схема (.json) при старте редактора")
    ap.add_argument("--template", default="Заготовка: тангаж → δe",
                    help="шаблон при старте (название из меню «Шаблоны»)")
    a = ap.parse_args()

    aircraft = aircraft_by_name(a.aircraft)
    wind_params = WindParams(Vw_const=a.wind_n, Vw_cross=a.wind_e)
    cfg = SimConfig(Va0=16.0, h0=a.h0, theta0=0.0, dt=0.01, t_end=1e9)

    tpl = templates(aircraft, cfg.Va0)
    if a.template not in tpl:
        ap.error(f"нет шаблона {a.template!r}; есть: " + "; ".join(tpl))
    start = tpl[a.template]
    if a.scheme:
        with open(a.scheme, encoding="utf-8") as f:
            start = json.load(f)

    q_cmd, q_tel = mp.Queue(maxsize=100), mp.Queue(maxsize=200)
    mp.Process(target=block_editor.run, daemon=True,
               args=(q_cmd, q_tel, catalog(), tpl, start, "-0+0")).start()

    app, terrain = make_app(f"Конструктор САУ — {MAPS[a.map].title}", a.map, size=WIN_3D)
    window.position = Vec2(0, 40)
    game = Lab(aircraft, wind_params, cfg, terrain, q_cmd, q_tel, cam=a.cam)
    try:
        app.run()
    finally:
        game.save()


if __name__ == "__main__":
    main()
