"""
Визуализация результатов Q-learning.

Запуск после обучения:  python rl_control/visualize.py
Или вызвать plot_training(rewards, episode_logs) из train.py напрямую.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_control.agent import (
    QLearningAgent, ACTION_OFFSETS, N_ACTIONS,
    N_THETA_BINS, N_Q_BINS,
    THETA_ERR_MIN, THETA_ERR_MAX,
    Q_RATE_MIN, Q_RATE_MAX,
)

SAVE_DIR = os.path.dirname(__file__)


def plot_training(rewards: list, episode_logs: list) -> None:
    """
    Три панели:
      1. Кривая обучения (суммарная награда + скользящее среднее)
      2. Траектории тангажа для каждого записанного эпизода
      3. Тепловая карта Q-таблицы: лучшее действие для каждой ячейки состояния
    """
    agent = QLearningAgent()
    qt_path = os.path.join(SAVE_DIR, "q_table.npy")
    if os.path.exists(qt_path):
        agent.load(qt_path)

    rewards_arr = np.array(rewards)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Панель 1: кривая обучения ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rewards_arr, alpha=0.25, color="steelblue", linewidth=0.6)
    window = 100
    if len(rewards_arr) >= window:
        ma = np.convolve(rewards_arr, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, len(rewards_arr)), ma,
                 color="steelblue", linewidth=2, label=f"MA-{window}")
    ax1.set_xlabel("Эпизод")
    ax1.set_ylabel("Суммарная награда")
    ax1.set_title("Кривая обучения")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Панель 2: траектории тангажа ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    cmap   = plt.cm.viridis
    n_logs = len(episode_logs)
    dt     = 0.01
    for i, log in enumerate(episode_logs):
        states    = log["states"]
        t_arr     = np.arange(len(states)) * dt
        theta_deg = np.degrees(states[:, 3])   # THETA = индекс 3
        color     = cmap(i / max(n_logs - 1, 1))
        label     = f"ep {log['ep']}  θref={log['theta_ref_deg']:.1f}°"
        ax2.plot(t_arr, theta_deg, color=color, alpha=0.8, linewidth=1.2, label=label)
        ax2.axhline(log["theta_ref_deg"], color=color, linestyle="--",
                    alpha=0.35, linewidth=0.8)
    ax2.set_xlabel("Время, с")
    ax2.set_ylabel("Тангаж θ, °")
    ax2.set_title("Траектории каждые 500 эпизодов")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ── Панель 3: тепловая карта Q-таблицы ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :])

    best_actions = np.argmax(agent.q_table, axis=2)   # (N_THETA_BINS, N_Q_BINS)

    theta_ticks = np.linspace(np.degrees(THETA_ERR_MIN), np.degrees(THETA_ERR_MAX), N_THETA_BINS)
    q_ticks     = np.linspace(Q_RATE_MIN, Q_RATE_MAX, N_Q_BINS)

    im = ax3.imshow(
        best_actions.T,
        aspect="auto", origin="lower",
        cmap="RdYlGn_r",
        vmin=0, vmax=N_ACTIONS - 1,
        extent=[np.degrees(THETA_ERR_MIN), np.degrees(THETA_ERR_MAX), Q_RATE_MIN, Q_RATE_MAX],
    )
    cbar = plt.colorbar(im, ax=ax3, ticks=range(N_ACTIONS))
    cbar.set_ticklabels([f"δe = {np.degrees(a):.0f}°" for a in ACTION_OFFSETS])
    cbar.set_label("Лучшее действие")

    ax3.axvline(0, color="white", linewidth=1.5, linestyle="--", alpha=0.6)
    ax3.axhline(0, color="white", linewidth=1.0, linestyle=":",  alpha=0.4)
    ax3.set_xlabel("Ошибка тангажа  θ_ref − θ, °")
    ax3.set_ylabel("Угловая скорость  q, рад/с")
    ax3.set_title("Q-таблица: лучшее действие для каждой ячейки состояния\n"
                  "(красный → руль вниз, зелёный → руль вверх)")

    fig.suptitle("Q-Learning: стабилизация тангажа БПЛА", fontsize=14, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    rewards_path = os.path.join(SAVE_DIR, "rewards.npy")
    logs_path    = os.path.join(SAVE_DIR, "episode_logs.npz")

    if not os.path.exists(rewards_path) or not os.path.exists(logs_path):
        print("Files not found. Run:  python rl_control/train.py")
        sys.exit(1)

    rewards_arr = np.load(rewards_path)
    data        = np.load(logs_path)
    episode_logs = [
        {
            "ep":            int(data["ep_nums"][i]),
            "theta_ref_deg": float(data["ep_refs"][i]),
            "states":        data["ep_states"][i],
            "total_reward":  float(data["ep_rewards"][i]),
        }
        for i in range(len(data["ep_nums"]))
    ]

    plot_training(list(rewards_arr), episode_logs)
