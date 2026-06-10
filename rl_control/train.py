"""
Тренировочный цикл Q-learning агента.

Запуск (однопоточно):    python rl_control/train.py
Запуск (параллельно):    python rl_control/train.py --workers 8

Выходные файлы (в папке rl_control/):
    q_table.npy       — обученная Q-таблица
    rewards.npy       — суммарная награда по эпизодам
    episode_logs.npz  — траектории каждого log_every-го эпизода
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.config      import AircraftParams, WindParams
from sim.state       import Q, THETA, H, U, W, N_STATES, air_velocity
from sim.integrators import step_rk4
from runner          import compute_trim
from rl_control.agent import QLearningAgent

SAVE_DIR = os.path.dirname(__file__)

# --- Гиперпараметры обучения ---
ACTION_SMOOTH_K = 3.0    # штраф за изменение руля (action smoothness penalty)
                          # уменьшен с 20→3: агент теперь может маневрировать


def _progress_bar(ep: int, n_episodes: int, epsilon: float,
                  r_avg: float, bar_width: int = 35) -> None:
    frac   = (ep + 1) / n_episodes
    filled = int(frac * bar_width)
    bar    = "=" * filled + ">" + " " * max(bar_width - filled - 1, 0)
    print(f"\r  [{bar}] {100 * frac:5.1f}%  "
          f"ep {ep+1}/{n_episodes}  eps={epsilon:.3f}  R_avg={r_avg:8.1f}",
          end="", flush=True)


def _make_init_state(Va: float, theta_init: float, h0: float) -> np.ndarray:
    state = np.zeros(N_STATES)
    state[U]     = Va * np.cos(theta_init)
    state[W]     = Va * np.sin(theta_init)
    state[THETA] = theta_init
    state[H]     = h0
    return state


def train(
    n_episodes:  int   = 5000,
    log_every:   int   = 500,
    dt:          float = 0.01,
    ep_duration: float = 10.0,   # увеличено 5→10 с: динамика успевает развиться
    Va_ref:      float = 30.0,
    h0:          float = 100.0,
    seed:        int   = 42,
    _worker_id:  int   = 0,      # зарезервировано
    silent:      bool  = False,  # True в воркерах — без прогресс-бара
) -> tuple:
    """
    Обучить агента. Возвращает (agent, rewards_history, episode_logs).
    """
    aircraft = AircraftParams()
    wind_fn  = lambda h, t: (0.0, 0.0)

    _, _, thr_trim   = compute_trim(aircraft, Va_ref)
    alpha_trim, de_trim, _ = compute_trim(aircraft, Va_ref)

    if not silent:
        print(f"Trim: alpha={np.degrees(alpha_trim):.2f}  "
              f"de_trim={np.degrees(de_trim):.2f}  throttle={thr_trim:.3f}")

    steps_per_ep = int(round(ep_duration / dt))
    # Действия центрированы вокруг тримового положения руля
    agent = QLearningAgent(alpha=0.1, gamma=0.95, de_trim=de_trim)
    rng   = np.random.default_rng(seed=seed)

    rewards_history = []
    episode_logs    = []
    r_avg_display   = 0.0

    for ep in range(n_episodes):
        epsilon = max(0.05, 1.0 - ep / (n_episodes * 0.8))

        theta_ref     = rng.uniform(-10.0, 10.0) * np.pi / 180.0
        theta_perturb = rng.uniform(-5.0,   5.0) * np.pi / 180.0
        state         = _make_init_state(Va_ref, alpha_trim + theta_perturb, h0)

        log_this  = (ep % log_every == 0) or (ep == n_episodes - 1)
        ep_states = [] if log_this else None

        total_reward    = 0.0
        t               = 0.0
        prev_action_idx = 2   # нейтральное смещение (0.0 рад)

        for _ in range(steps_per_ep):
            theta_err  = theta_ref - state[THETA]
            q_val      = state[Q]

            action_idx = agent.select_action(theta_err, q_val, epsilon)
            controls   = np.array([agent.actions[action_idx], thr_trim])

            state_next = step_rk4(state, controls, dt, t, aircraft, wind_fn)

            _, alpha_next  = air_velocity(state_next, wind_fn(state_next[H], t + dt))
            theta_err_next = theta_ref - state_next[THETA]
            q_next         = state_next[Q]

            de_change = np.degrees(
                agent.actions[action_idx] - agent.actions[prev_action_idx]
            )
            reward = (
                -abs(np.degrees(theta_err_next))
                - 0.1  * np.degrees(q_next) ** 2
                - ACTION_SMOOTH_K * de_change ** 2 / 1000.0
                - 50.0 * float(abs(alpha_next) > aircraft.alpha_stall)
            )
            total_reward += reward

            agent.update(theta_err, q_val, action_idx, reward, theta_err_next, q_next)

            if ep_states is not None:
                ep_states.append(state.copy())

            state           = state_next
            prev_action_idx = action_idx
            t              += dt

            if state[H] < 10.0 or abs(state[THETA]) > np.radians(60.0):
                total_reward -= 500.0
                break

        rewards_history.append(total_reward)

        if log_this:
            episode_logs.append({
                "ep":            ep,
                "theta_ref_deg": float(np.degrees(theta_ref)),
                "states":        np.array(ep_states),
                "total_reward":  total_reward,
            })

        if len(rewards_history) >= 100:
            r_avg_display = float(np.mean(rewards_history[-100:]))
        elif rewards_history:
            r_avg_display = float(np.mean(rewards_history))

        if silent:
            # Обновляем общий счётчик для прогресс-бара главного процесса
            if _shared_counter is not None and (ep + 1) % 50 == 0:
                with _shared_counter.get_lock():
                    _shared_counter.value += 50
        else:
            _progress_bar(ep, n_episodes, epsilon, r_avg_display)

    if not silent:
        print()

    return agent, rewards_history, episode_logs


# ---------------------------------------------------------------------------
# Параллельный запуск (Вариант 1: независимые агенты, берём лучшего)
# ---------------------------------------------------------------------------

# Глобальный счётчик завершённых эпизодов — заполняется из воркеров
_shared_counter = None

def _init_worker(counter):
    global _shared_counter
    _shared_counter = counter


def _worker_fn(args: tuple) -> tuple:
    """Воркер для multiprocessing.Pool. Возвращает (q_table, de_trim, r_avg, rewards)."""
    worker_id, seed, n_episodes, log_every, ep_duration, Va_ref, h0 = args
    agent, rewards, _ = train(
        n_episodes=n_episodes,
        log_every=log_every,
        ep_duration=ep_duration,
        Va_ref=Va_ref,
        h0=h0,
        seed=seed,
        _worker_id=worker_id,
        silent=True,
    )
    r_avg = float(np.mean(rewards[-200:])) if len(rewards) >= 200 else float(np.mean(rewards))
    return agent.q_table.copy(), agent.de_trim, r_avg, rewards


def train_parallel(
    n_workers:   int   = None,
    n_episodes:  int   = 5000,
    log_every:   int   = 500,
    ep_duration: float = 10.0,
    Va_ref:      float = 30.0,
    h0:          float = 100.0,
) -> tuple:
    """
    Запустить n_workers независимых агентов параллельно.
    Возвращает лучшего по финальной средней награде.
    """
    import ctypes
    import threading
    import time
    from multiprocessing import Pool, Value, cpu_count

    if n_workers is None:
        n_workers = cpu_count()

    total_eps = n_workers * n_episodes
    counter   = Value(ctypes.c_int, 0)

    print(f"Запускаю {n_workers} воркеров x {n_episodes} эпизодов  "
          f"(всего {total_eps} эп)")

    # --- Фоновый поток: читает счётчик и рисует прогресс-бар ---
    bar_width = 35
    def _show_progress():
        while True:
            done  = counter.value
            frac  = min(done / total_eps, 1.0)
            filled = int(frac * bar_width)
            bar   = "=" * filled + ">" + " " * max(bar_width - filled - 1, 0)
            print(f"\r  [{bar}] {100 * frac:5.1f}%  "
                  f"{done}/{total_eps} эп  ({n_workers} воркеров)",
                  end="", flush=True)
            if done >= total_eps:
                break
            time.sleep(0.3)

    progress_thread = threading.Thread(target=_show_progress, daemon=True)
    progress_thread.start()

    args_list = [
        (i, 42 + i * 7, n_episodes, log_every, ep_duration, Va_ref, h0)
        for i in range(n_workers)
    ]

    with Pool(n_workers,
              initializer=_init_worker,
              initargs=(counter,)) as pool:
        results = pool.map(_worker_fn, args_list)

    counter.value = total_eps   # гарантируем 100% на баре
    progress_thread.join(timeout=1.0)
    print()   # перенос строки после прогресс-бара

    r_avgs = [r[2] for r in results]
    best_i = int(np.argmax(r_avgs))

    print("Результаты воркеров:")
    for i, (_, _, r_avg, _) in enumerate(results):
        marker = " <-- лучший" if i == best_i else ""
        print(f"  worker {i}: R_avg200={r_avg:.1f}{marker}")

    best_agent = QLearningAgent(de_trim=results[best_i][1])
    best_agent.q_table = results[best_i][0]

    return best_agent, results[best_i][3], []


# ---------------------------------------------------------------------------
# Сохранение артефактов
# ---------------------------------------------------------------------------

def save_artifacts(agent: QLearningAgent, rewards: list,
                   episode_logs: list) -> None:
    agent.save(os.path.join(SAVE_DIR, "q_table.npy"))
    np.save(os.path.join(SAVE_DIR, "rewards.npy"), np.array(rewards))

    if episode_logs:
        ep_nums    = np.array([lg["ep"]            for lg in episode_logs])
        ep_refs    = np.array([lg["theta_ref_deg"] for lg in episode_logs])
        ep_rewards = np.array([lg["total_reward"]  for lg in episode_logs])
        max_len    = max(len(lg["states"]) for lg in episode_logs)
        ep_states  = np.stack([
            np.pad(lg["states"], ((0, max_len - len(lg["states"])), (0, 0)), mode="edge")
            for lg in episode_logs
        ])
        np.savez(os.path.join(SAVE_DIR, "episode_logs.npz"),
                 ep_nums=ep_nums, ep_refs=ep_refs,
                 ep_rewards=ep_rewards, ep_states=ep_states)

    np.save(os.path.join(SAVE_DIR, "de_trim.npy"), np.array([agent.de_trim]))
    print(f"Артефакты сохранены в {SAVE_DIR}")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from rl_control.visualize import plot_training

    parser = argparse.ArgumentParser(description="Q-learning pitch controller training")
    parser.add_argument("--workers",  type=int, default=1,
                        help="Число параллельных воркеров (1 = однопоточно)")
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Эпизодов на воркер")
    args = parser.parse_args()

    if args.workers > 1:
        agent, rewards, logs = train_parallel(
            n_workers=args.workers, n_episodes=args.episodes
        )
    else:
        agent, rewards, logs = train(n_episodes=args.episodes)

    save_artifacts(agent, rewards, logs)

    if logs:
        print("Строю графики обучения...")
        plot_training(rewards, logs)
    else:
        print("Параллельный режим: запусти visualize.py отдельно после "
              "однопоточного обучения.")
