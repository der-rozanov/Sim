"""
TECS — система управления полной энергией (Lambregts, 1983; так устроены продольные
контуры PX4 и ArduPilot): высота и скорость одновременно, газом и тангажом.

Энергии на единицу веса и скорости (безразмерные, рад):
    Ė = γ + V̇/g                      — темп полной энергии (E = h + V²/2g)
    Ḃ = w·γ − (2 − w)·V̇/g            — темп баланса: высота против скорости
    γ = ḣ/Va — угол траектории (по воздушной скорости), w ∈ [0, 2] — вес высоты
    (w = 1 — поровну; w = 0 — тангаж держит только скорость)

Почему газ — для Ė, а тангаж — для Ḃ (горизонтальный полёт, B&M 5.4):
    m·V̇ = T − D − m·g·sin γ   ⇒   Ė = (T − D)/(m·g)       — от тангажа не зависит
    Ḃ = 2γ − (2 − w)·Ė                                     — тангаж меняет γ: ∂Ḃ/∂γ = 2

Внешние контуры (П, с ограничениями):
    ḣ_ref = ḣ_ff + K_h·(h_ref − h),  −Va·γ_sink(Va) ≤ ḣ_ref ≤ Va·γ_climb(Va);   γ_ref = ḣ_ref / Va
  ḣ_ff — темп уставки (наклонный участок маршрута): без него отставание ḣ_ff/K_h
  γ_climb, γ_sink — доля располагаемого темпа энергии на газе 1 и 0 (таблица по Va)
    V̇_ref = K_V·(V_ref − Va),  |V̇_ref| ≤ acc_max
    Ė_ref = γ_ref + V̇_ref/g,   Ḃ_ref = w·γ_ref − (2 − w)·V̇_ref/g

Внутренние контуры (ПИ по ошибке темпа; интеграл темпа — ошибка самой энергии):
    δt = δt*(V_ref) + K_E·Ė_ref + kp_T·e_E + ki_T·∫e_E,           e_E = Ė_ref − Ė
    θ  = α*(Va) + γ_ff + kp_B·e_B + ki_B·∫e_B,                     e_B = Ḃ_ref − Ḃ
    γ_ff = (Ḃ_ref + (2 − w)·Ė_ref)/2   (при w = 1 — просто γ_ref)
  δt*(V), α*(V) — балансировка горизонтального полёта (таблица по скорости),
  K_E = ∂δt/∂Ė = g/a_V2 — упреждение газа по требуемому темпу энергии.

V̇ — производная отфильтрованной Va (фильтр 1-го порядка τ_f); ḣ — вертикальная
скорость GPS. Расчёт коэффициентов — control/tuning.py (design_sau → d.tecs).

Режим PITCH (тангаж задан вручную): γ_ref = γ — высоту не держим, газ держит скорость.
Защита от потери скорости: при Va < Va_min (полностью — на Va_min − Va_band)
w → 0, набор запрещается, газ → максимум.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TECSParams:
    """Параметры TECS. Умолчания — расчёт control/tuning.py (FPV, Va = 16 м/с)."""
    K_h: float = 0.5           # 1/с, контур высоты: ḣ_ref = K_h·Δh
    K_V: float = 0.5           # 1/с, контур скорости: V̇_ref = K_V·ΔV
    acc_max: float = 5.398     # м/с², предел |V̇_ref| = 0.7·(T(1) − D)/m
    w: float = 1.0             # вес высоты в балансе энергии
    tau_f: float = 0.15        # с, фильтр Va перед дифференцированием
    K_E: float = 0.8516        # ∂δt/∂Ė = g/a_V2
    thr_Kp: float = 0.0        # газ: ПИ по e_E (П = 0: шум ПВД через V̇, tuning.py)
    thr_Ki: float = 1.703      # = ω_T·K_E, ω_T = 2 рад/с
    pit_Kp: float = 0.0        # тангаж: ПИ по e_B
    pit_Ki: float = 1.0        # = ω_B/2, ω_B = 2 рад/с
    theta_max: float = np.radians(15.0)
    theta_min: float = np.radians(-15.0)
    Va_min: float = 9.403      # м/с, начало защиты от потери скорости = 1.25·Va_stall
    Va_band: float = 1.5       # м/с, полная защита на Va_min − Va_band
    # Балансировка по скорости: δt*(V), α*(V)
    Va_tab: tuple = (10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0)
    thr_tab: tuple = (0.3297, 0.3982, 0.4826, 0.5751, 0.673, 0.7751, 0.881)
    alpha_tab: tuple = (0.0888, 0.0299, 0.0009, -0.0154, -0.0256, -0.0323, -0.037)
    # Пределы угла траектории, рад: 0.7 располагаемого темпа энергии, не круче ±15°
    gamma_climb_tab: tuple = (0.1812, 0.1812, 0.1812, 0.1812, 0.1812, 0.1812, 0.1411)
    gamma_sink_tab: tuple = (0.0854, 0.1049, 0.1379, 0.1812, 0.1812, 0.1812, 0.1812)


class TECS:
    """
    Продольный внешний контур САУ на энергии: (h_ref, V_ref) → (θ_cmd, δt).
    Тот же интерфейс, что у AltSpeedHold (control/sau.py):
    step(refs, est, dt, hold_alt) — hold_alt = False в режиме PITCH (θ_cmd = refs.theta).
    """

    def __init__(self, aircraft, params: TECSParams = None):
        self.ac = aircraft
        self.params = params or TECSParams()
        self.reset()

    @property
    def thr_trim(self):
        """Балансировочный газ на уставке скорости (для защиты по α)."""
        pr = self.params
        return float(np.interp(self.V_ref, pr.Va_tab, pr.thr_tab))

    def reset(self):
        self.iE = 0.0          # ∫e_E — ошибка полной энергии
        self.reset_alt()
        self._Vf = None        # отфильтрованная Va
        self.V_ref = self.params.Va_tab[1]
        # Последний шаг — для журнала и окна стенда
        self.E_ref = self.E = self.B_ref = self.B = 0.0
        self.hdot_ref = self.Vdot_ref = self.Vdot = 0.0
        self.underspeed = 0.0

    def reset_alt(self):
        """Новая уставка высоты / смена режима: интеграл баланса с нуля."""
        self.iB = 0.0

    def step(self, refs, est, dt, hold_alt=True):
        pr, ac, g = self.params, self.ac, self.ac.g
        V = max(est["Va"], 1.0)
        self.V_ref = refs.Va

        # V̇ — производная отфильтрованной Va
        if self._Vf is None:
            self._Vf = V
        Vf = self._Vf + dt / (pr.tau_f + dt) * (V - self._Vf)
        self.Vdot, self._Vf = (Vf - self._Vf) / dt, Vf
        gamma = float(np.clip(est["hdot"] / V, -1.0, 1.0))

        # Защита от потери скорости: 0 — нет, 1 — полная
        k = float(np.clip((pr.Va_min - V) / pr.Va_band, 0.0, 1.0))
        self.underspeed = k
        w = pr.w * (1.0 - k)

        # Внешние контуры → требуемые темпы
        self.Vdot_ref = float(np.clip(pr.K_V * (refs.Va - V), -pr.acc_max, pr.acc_max))
        if not hold_alt:
            self.hdot_ref = est["hdot"]
            gamma_ref = gamma
        else:
            g_up = np.interp(V, pr.Va_tab, pr.gamma_climb_tab)
            g_dn = np.interp(V, pr.Va_tab, pr.gamma_sink_tab)
            self.hdot_ref = float(np.clip(refs.hdot + pr.K_h * (refs.h - est["h"]),
                                          -V * g_dn, V * g_up))
            if self.hdot_ref > 0:
                self.hdot_ref *= 1.0 - k             # при потере скорости не набираем
            gamma_ref = self.hdot_ref / V
        self.E_ref = gamma_ref + self.Vdot_ref / g
        self.E = gamma + self.Vdot / g
        self.B_ref = w * gamma_ref - (2.0 - w) * self.Vdot_ref / g
        self.B = w * gamma - (2.0 - w) * self.Vdot / g

        # Газ ← полная энергия
        e_E = self.E_ref - self.E
        self.iE += e_E * dt
        raw = (np.interp(refs.Va, pr.Va_tab, pr.thr_tab) + pr.K_E * self.E_ref
               + pr.thr_Kp * e_E + pr.thr_Ki * self.iE)
        throttle = float(np.clip(raw, ac.throttle_min, ac.throttle_max))
        if raw != throttle and e_E * (raw - throttle) > 0:
            self.iE -= e_E * dt                      # газ в упоре — интеграл не копим
        throttle += k * (ac.throttle_max - throttle)

        # Тангаж ← баланс энергии
        if not hold_alt:
            return refs.theta, throttle
        e_B = self.B_ref - self.B
        self.iB += e_B * dt
        gamma_ff = 0.5 * (self.B_ref + (2.0 - w) * self.E_ref)
        raw = (np.interp(V, pr.Va_tab, pr.alpha_tab) + gamma_ff
               + pr.pit_Kp * e_B + pr.pit_Ki * self.iB)
        theta = float(np.clip(raw, pr.theta_min, pr.theta_max))
        if raw != theta and e_B * (raw - theta) > 0:
            self.iB -= e_B * dt
        return theta, throttle
