# -*- coding: utf-8 -*-
"""
Графики симулятора. Все подписи — на русском.

plot_dynamics(log, params)          -- траектории состояний и управления
plot_trajectory(log, wind_params)   -- путь ЛА в 2D пространстве
plot_energy(log, params)            -- механическая энергия
plot_integrator_check(...)          -- проверка качества интегратора
"""

import numpy as np
import matplotlib.pyplot as plt

from sim.state import U, W, Q, THETA, X, H as H_IDX
from sim.config import AircraftParams, WindParams

# Русские шрифты в matplotlib на Windows работают через rcParams
plt.rcParams["font.family"] = "DejaVu Sans"


# ---------------------------------------------------------------------------
# Фигура 1: динамика (time histories)
# ---------------------------------------------------------------------------

def plot_dynamics(log, params: AircraftParams, title: str = "") -> plt.Figure:
    """
    Состояния и управление по времени. 4 строки × 2 столбца:
      Воздушная скорость  |  Угол атаки (+ пороги)
      Угол тангажа        |  Угловая скорость тангажа
      Высота              |  Угол наклона траектории
      Руль высоты         |  Газ
    """
    t     = log.t
    Va    = log.Va
    alpha = np.degrees(log.alpha)
    theta = np.degrees(log.state[:, THETA])
    q     = np.degrees(log.state[:, Q])
    h     = log.state[:, H_IDX]
    de    = np.degrees(log.controls[:, 0])
    thr   = log.controls[:, 1]
    gamma = np.degrees(log.state[:, THETA] - log.alpha)

    fig, axes = plt.subplots(4, 2, figsize=(13, 10), sharex=True)
    fig.suptitle(title if title else "Динамика полёта", fontsize=13, fontweight="bold")
    fig.subplots_adjust(hspace=0.42, wspace=0.38)

    # --- Воздушная скорость ---
    _ts(axes[0, 0], t, Va, "Воздушная скорость, м/с", "steelblue")

    # --- Угол атаки с порогами ---
    _ts(axes[0, 1], t, alpha, "Угол атаки (УА), °", "crimson")
    warn_deg = np.degrees(params.alpha_warning)
    crit_deg = np.degrees(params.alpha_crit)
    a_max = alpha.max()
    y_lo  = min(alpha.min() - 2, -2)
    y_hi  = max(a_max * 1.5 + 1, a_max + 2)
    if warn_deg <= y_hi * 1.8:
        _hline(axes[0, 1], warn_deg, "orange", f"предупреждение {warn_deg:.0f}°")
        _hline(axes[0, 1], crit_deg, "red",    f"критический {crit_deg:.0f}°")
        axes[0, 1].legend(fontsize=7, loc="upper right")
    else:
        axes[0, 1].set_ylim(y_lo, y_hi)
        axes[0, 1].text(0.98, 0.95,
                        f"пред.={warn_deg:.0f}°  крит.={crit_deg:.0f}°",
                        transform=axes[0, 1].transAxes,
                        ha="right", va="top", fontsize=8, color="gray")

    # --- Угол тангажа ---
    _ts(axes[1, 0], t, theta, "Угол тангажа θ, °", "seagreen")

    # --- Угловая скорость тангажа ---
    _ts(axes[1, 1], t, q, "Угловая скорость тангажа q, °/с", "mediumpurple")

    # --- Высота ---
    _ts(axes[2, 0], t, h, "Высота h, м", "royalblue")

    # --- Угол наклона траектории ---
    _ts(axes[2, 1], t, gamma, "Угол наклона траектории γ, °", "darkorange")
    _hline(axes[2, 1], 0, "gray", "")

    # --- Руль высоты ---
    _ts(axes[3, 0], t, de, "Отклонение руля высоты δe, °", "saddlebrown")
    _hline(axes[3, 0], 0, "gray", "")

    # --- Газ ---
    _ts(axes[3, 1], t, thr, "Газ (тяга), о.е.", "darkgreen")
    axes[3, 1].set_ylim(-0.05, 1.05)

    for ax in axes[3, :]:
        ax.set_xlabel("Время, с", fontsize=9)

    return fig


# ---------------------------------------------------------------------------
# Фигура 2: 2D траектория
# ---------------------------------------------------------------------------

def plot_trajectory(log, wind_params: WindParams = None,
                    title: str = "") -> plt.Figure:
    """Путь ЛА в вертикальной плоскости (горизонтальная дальность vs высота)."""
    x = log.state[:, X]
    h = log.state[:, H_IDX]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(title if title else "Траектория полёта в вертикальной плоскости",
                 fontsize=13, fontweight="bold")

    ax.plot(x, h, "b-", linewidth=1.8, label="траектория")
    ax.plot(x[0],  h[0],  "go", markersize=9, zorder=5, label="старт")
    ax.plot(x[-1], h[-1], "rs", markersize=9, zorder=5, label="финиш")

    if wind_params is not None and wind_params.dV_shear != 0.0:
        lo, hi = wind_params.h_shear_lo, wind_params.h_shear_hi
        ax.axhspan(lo, hi, alpha=0.15, color="orange",
                   label=f"слой сдвига ветра ({lo:.0f}–{hi:.0f} м)")

    ax.set_xlabel("Горизонтальная дальность x, м", fontsize=10)
    ax.set_ylabel("Высота h, м", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="datalim")

    return fig


# ---------------------------------------------------------------------------
# Фигура 3: механическая энергия
# ---------------------------------------------------------------------------

def plot_energy(log, params: AircraftParams, title: str = "") -> plt.Figure:
    """
    Верхний subplot: кинетическая, потенциальная и полная энергия.
    Нижний subplot: отклонение полной энергии от начального значения.
    """
    t  = log.t
    Ek = log.E_kin   / 1000
    Ep = log.E_pot   / 1000
    Et = log.E_total / 1000
    dE = (log.E_total - log.E_total[0]) / 1000
    pct = 100 * (log.E_total[-1] - log.E_total[0]) / log.E_total[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(title if title else "Механическая энергия ЛА",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(hspace=0.35)

    axes[0].plot(t, Ek, label="Кинетическая E_к",   color="tomato",    linewidth=1.5)
    axes[0].plot(t, Ep, label="Потенциальная E_п",   color="steelblue", linewidth=1.5)
    axes[0].plot(t, Et, label="Полная E = E_к + E_п", color="black",   linewidth=2.2)
    axes[0].set_ylabel("Энергия, кДж", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_title("Обмен между кинетической и потенциальной энергией", fontsize=9)

    axes[1].plot(t, dE, color="black", linewidth=1.8)
    axes[1].axhline(0, color="gray", linestyle="--")
    axes[1].set_ylabel("ΔE_полн., кДж", fontsize=10)
    axes[1].set_xlabel("Время, с", fontsize=10)
    axes[1].set_title(f"Изменение полной энергии за прогон: {pct:+.2f}%", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    return fig


# ---------------------------------------------------------------------------
# Фигура 4: проверка качества интегратора
# ---------------------------------------------------------------------------

def plot_integrator_check(aircraft: AircraftParams,
                          wind_params: WindParams,
                          cfg,
                          t_end: float = 40.0,
                          dt_euler: float = 0.05) -> plt.Figure:
    """
    Тримовый полёт из точных балансировочных начальных условий.
    Тяга = сопротивление => E_total должна оставаться постоянной.
    Отсутствие роста энергии подтверждает достоверность интегратора (RK4).
    """
    from runner import run, compute_trim, trim_state
    from sim.config import SimConfig
    import copy

    alpha_tr, de_tr, thr_tr = compute_trim(aircraft, cfg.Va0)
    ctrl_arr = np.array([de_tr, thr_tr])
    s0 = trim_state(aircraft, cfg)

    cfg2 = copy.copy(cfg)
    cfg2.t_end = t_end
    log = run(lambda t, s, V, a: ctrl_arr, aircraft, wind_params, cfg2, state0=s0)

    t  = log.t
    Ek = log.E_kin   / 1000
    Ep = log.E_pot   / 1000
    Et = log.E_total / 1000
    dE = (log.E_total - log.E_total[0]) / 1000
    pct = 100 * (log.E_total[-1] - log.E_total[0]) / log.E_total[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f"Проверка интегратора (RK4, шаг dt={cfg.dt} с, длит. {t_end} с)",
        fontsize=12, fontweight="bold")
    fig.subplots_adjust(hspace=0.38)

    axes[0].plot(t, Ek, label="Кинетическая",  color="tomato",    linewidth=1.5)
    axes[0].plot(t, Ep, label="Потенциальная", color="steelblue", linewidth=1.5)
    axes[0].plot(t, Et, label="Полная",        color="black",     linewidth=2.2)
    axes[0].set_ylabel("Энергия, кДж", fontsize=10)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_title("Тримовый полёт: тяга ≈ сопротивление  ⟹  E_полн ≈ const",
                      fontsize=9)

    axes[1].plot(t, dE, color="black", linewidth=1.8)
    axes[1].axhline(0, color="gray", linestyle=":")
    axes[1].set_ylabel("ΔE_полн., кДж (отклонение от t=0)", fontsize=10)
    axes[1].set_xlabel("Время, с", fontsize=10)
    axes[1].set_title(
        f"Суммарный дрейф за {t_end} с: {pct:+.3f}%  "
        f"({log.E_total[-1] - log.E_total[0]:.0f} Дж)",
        fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    return fig


# ---------------------------------------------------------------------------
# утилиты
# ---------------------------------------------------------------------------

def _ts(ax, t, y, ylabel, color):
    ax.plot(t, y, color=color, linewidth=1.3)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)


def _hline(ax, y, color, label):
    kw = dict(color=color, linestyle="--", linewidth=1.2, alpha=0.85)
    if label:
        kw["label"] = label
    ax.axhline(y, **kw)


def show_all():
    plt.tight_layout()
    plt.show()
