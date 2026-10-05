"""Q-learning агент для стабилизации тангажа."""

import numpy as np

# --- Пространство состояний ---
THETA_ERR_MIN = np.radians(-20.0)   # рад
THETA_ERR_MAX = np.radians(+20.0)
N_THETA_BINS  = 21                  # шаг 2°

Q_RATE_MIN = -1.5                   # рад/с
Q_RATE_MAX = +1.5
N_Q_BINS   = 15                     # шаг 0.2 рад/с

# --- Пространство действий: смещения относительно тримового положения руля ---
# Агент учится двигать руль вокруг de_trim, а не вокруг нуля.
# Реальный delta_e = de_trim + ACTION_OFFSETS[i]
ACTION_OFFSETS = np.array([-0.30, -0.15, 0.0, +0.15, +0.30])   # рад
N_ACTIONS      = len(ACTION_OFFSETS)


class QLearningAgent:
    """
    Табличный Q-learning для продольного канала.

    Состояние: (theta_err, q) — ошибка тангажа и угловая скорость.
    Действие:  дискретное отклонение руля высоты delta_e = de_trim + смещение.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95,
                 de_trim: float = 0.0):
        self.alpha   = alpha
        self.gamma   = gamma
        self.de_trim = de_trim
        self.actions = de_trim + ACTION_OFFSETS   # реальные углы руля, рад
        # Ось 2 — предыдущее действие (часть состояния MDP), ось 3 — текущее действие.
        # Форма: (N_THETA_BINS, N_Q_BINS, N_ACTIONS_prev, N_ACTIONS_curr)
        self.q_table = np.zeros((N_THETA_BINS, N_Q_BINS, N_ACTIONS, N_ACTIONS))

    def _discretize(self, theta_err: float, q: float) -> tuple:
        ti = int(np.clip(
            (theta_err - THETA_ERR_MIN) / (THETA_ERR_MAX - THETA_ERR_MIN) * (N_THETA_BINS - 1),
            0, N_THETA_BINS - 1
        ))
        qi = int(np.clip(
            (q - Q_RATE_MIN) / (Q_RATE_MAX - Q_RATE_MIN) * (N_Q_BINS - 1),
            0, N_Q_BINS - 1
        ))
        return ti, qi

    def select_action(self, theta_err: float, q: float,
                      prev_action_idx: int, epsilon: float) -> int:
        """Epsilon-жадная стратегия: исследование vs эксплуатация."""
        if np.random.random() < epsilon:
            return np.random.randint(N_ACTIONS)
        ti, qi = self._discretize(theta_err, q)
        return int(np.argmax(self.q_table[ti, qi, prev_action_idx]))

    def update(self,
               theta_err: float, q: float, prev_action_idx: int,
               action: int, reward: float,
               theta_err_next: float, q_next: float) -> None:
        """Обновление Q-таблицы по уравнению Беллмана.

        Следующее состояние: prev_action в нём равен текущему action.
        """
        ti,    qi    = self._discretize(theta_err,      q)
        ti_n,  qi_n  = self._discretize(theta_err_next, q_next)

        td_target = reward + self.gamma * np.max(self.q_table[ti_n, qi_n, action])
        td_error  = td_target - self.q_table[ti, qi, prev_action_idx, action]
        self.q_table[ti, qi, prev_action_idx, action] += self.alpha * td_error

    def get_action(self, theta_err: float, q: float,
                   prev_action_idx: int) -> tuple:
        """Жадная политика без исследования. Возвращает (action_idx, delta_e)."""
        ti, qi = self._discretize(theta_err, q)
        ai = int(np.argmax(self.q_table[ti, qi, prev_action_idx]))
        return ai, float(self.actions[ai])

    def get_delta_e(self, theta_err: float, q: float,
                    prev_action_idx: int) -> float:
        """Жадная политика без исследования — для демонстрации."""
        _, de = self.get_action(theta_err, q, prev_action_idx)
        return de

    def save(self, path: str) -> None:
        np.save(path, self.q_table)
        print(f"Q-table сохранена: {path}")

    def load(self, path: str) -> None:
        loaded = np.load(path)
        expected = (N_THETA_BINS, N_Q_BINS, N_ACTIONS, N_ACTIONS)
        if loaded.shape != expected:
            print(f"Q-table: форма {loaded.shape} != ожидаемой {expected} — таблица сброшена")
            return
        self.q_table = loaded
        print(f"Q-table загружена: {path}")
