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
        self.q_table = np.zeros((N_THETA_BINS, N_Q_BINS, N_ACTIONS))

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

    def select_action(self, theta_err: float, q: float, epsilon: float) -> int:
        """Epsilon-жадная стратегия: исследование vs эксплуатация."""
        if np.random.random() < epsilon:
            return np.random.randint(N_ACTIONS)
        ti, qi = self._discretize(theta_err, q)
        return int(np.argmax(self.q_table[ti, qi]))

    def update(self,
               theta_err: float, q: float, action: int, reward: float,
               theta_err_next: float, q_next: float) -> None:
        """Обновление Q-таблицы по уравнению Беллмана."""
        ti,  qi  = self._discretize(theta_err,      q)
        ti_n, qi_n = self._discretize(theta_err_next, q_next)

        td_target = reward + self.gamma * np.max(self.q_table[ti_n, qi_n])
        td_error  = td_target - self.q_table[ti, qi, action]
        self.q_table[ti, qi, action] += self.alpha * td_error

    def get_delta_e(self, theta_err: float, q: float) -> float:
        """Жадная политика без исследования — для демонстрации."""
        ti, qi = self._discretize(theta_err, q)
        return float(self.actions[np.argmax(self.q_table[ti, qi])])

    def save(self, path: str) -> None:
        np.save(path, self.q_table)
        print(f"Q-table сохранена: {path}")

    def load(self, path: str) -> None:
        self.q_table = np.load(path)
        print(f"Q-table загружена: {path}")
