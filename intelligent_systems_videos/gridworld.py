# %%

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow


# ══════════════════════════════════════════════════════════
# 1.  Environment: Windy 10 × 10 GridWorld
# ══════════════════════════════════════════════════════════
class GridWorld:
    """
    10 × 10 grid-world with stochastic wind.

    State    : (row, col)
    Actions  : 0-up, 1-right, 2-down, 3-left
    Rewards  : +1 at goal, −0.1 step cost
    """

    def __init__(self, size=10, num_blocks=10,
                 wind_prob=0.20, max_steps=None, seed=None):

        self.size = size
        self.num_blocks = num_blocks
        self.wind_prob = wind_prob
        self.max_steps = max_steps or num_blocks * 4          # safety cap

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.reset()

    # ------------------------------------------------------
    def reset(self):
        """Randomise start/goal/obstacles; return initial state."""
        self.blocks = set()
        self.start = self._rand_cell()
        self.goal = self._rand_cell(exclude={self.start})

        while len(self.blocks) < self.num_blocks:
            cell = self._rand_cell(exclude={self.start, self.goal}.union(self.blocks))
            self.blocks.add(cell)

        self.agent = self.start
        self.steps_taken = 0
        return self.agent

    # ------------------------------------------------------
    def step(self, action):
        """
        Apply action with wind, return (next_state, reward, done).
        done triggers when goal reached OR step budget exhausted.
        """
        self.steps_taken += 1

        if random.random() < self.wind_prob:
            action = random.choice([a for a in range(4) if a != action])

        r, c = self.agent
        if action == 0:
            r = max(r - 1, 0)
        elif action == 1:
            c = min(c + 1, self.size - 1)
        elif action == 2:
            r = min(r + 1, self.size - 1)
        elif action == 3:
            c = max(c - 1, 0)

        nxt = (r, c) if (r, c) not in self.blocks else self.agent
        self.agent = nxt

        reached_goal = self.agent == self.goal
        timeout = self.steps_taken >= self.max_steps
        done = reached_goal or timeout
        reward = 1.0 if reached_goal else -0.1
        return nxt, reward, done

    # ------------------------------------------------------
    def render(self):
        """Quick matplotlib snapshot of the grid."""
        img = np.ones((self.size, self.size, 3))
        for (r, c) in self.blocks:
            img[r, c] = [0, 0, 0]               # obstacles black
        sr, sc = self.start
        img[sr, sc] = [0, 1, 0]                 # start green
        gr, gc = self.goal
        img[gr, gc] = [1, 1, 0]                 # goal yellow
        ar, ac = self.agent
        img[ar, ac] = [1, 0, 0]                 # agent red

        plt.figure(figsize=(4, 4))
        plt.imshow(img, interpolation="nearest")
        plt.xticks(range(self.size))
        plt.yticks(range(self.size))
        plt.grid(color="gray", lw=0.5)
        plt.show()

    # ------------------------------------------------------
    def _rand_cell(self, exclude=set()):
        while True:
            cell = (random.randrange(self.size), random.randrange(self.size))
            if cell not in exclude:
                return cell


# ══════════════════════════════════════════════════════════
# 2.  ε-greedy helper
# ══════════════════════════════════════════════════════════
def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        return random.randrange(4)
    r, c = state
    return int(np.argmax(Q[r, c]))

def epsilon_decay(ep, total, start=1.0, end=0.01):
    """
    Linearly decay ε from `start` to `end`
    over the first half of `total` episodes.
    After that, keep it fixed at `end`.
    """
    half = total // 2
    if ep >= half:
        return end
    frac = ep / half                     # ∈ [0, 1)
    return start - frac * (start - end)  # linear interpolation

# ══════════════════════════════════════════════════════════
# 3.  Algorithms: Q-Learning & SARSA
# ══════════════════════════════════════════════════════════
def q_learning(env, episodes=3000, alpha=0.10, gamma=0.99, epsilon=0.10):
    Q = np.zeros((env.size, env.size, 4))

    for ep in range(episodes):
        state = env.reset()
        done = False
        epsilon = epsilon_decay(ep, episodes)

        while not done:
            a = epsilon_greedy(Q, state, epsilon)
            nxt, reward, done = env.step(a)

            r, c = state
            nr, nc = nxt
            td_target = reward + gamma * np.max(Q[nr, nc])
            Q[r, c, a] += alpha * (td_target - Q[r, c, a])
            state = nxt

    return Q


def sarsa(env, episodes=3000, alpha=0.10, gamma=0.99, epsilon=0.10):
    Q = np.zeros((env.size, env.size, 4))

    for ep in range(episodes):
        state = env.reset()
        epsilon = epsilon_decay(ep, episodes)
        a = epsilon_greedy(Q, state, epsilon)
        done = False

        while not done:
            nxt, reward, done = env.step(a)
            na = epsilon_greedy(Q, nxt, epsilon)

            r, c = state
            nr, nc = nxt
            td_target = reward + gamma * Q[nr, nc, na]
            Q[r, c, a] += alpha * (td_target - Q[r, c, a])
            state, a = nxt, na

    return Q


# ══════════════════════════════════════════════════════════
# 4.  Utilities (policy print, path draw)
# ══════════════════════════════════════════════════════════
def extract_policy(Q):
    return np.argmax(Q, axis=2)


def print_policy(policy):
    arrow = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    for row in policy:
        print(" ".join(arrow[a] for a in row))


def run_greedy_episode(env, Q):
    path = []
    state = env.reset()
    done = False

    while not done:
        r, c = state
        a = int(np.argmax(Q[r, c]))
        path.append((state, a))
        state, _, done = env.step(a)
    return path


def draw_path(env, path, title):
    env.render()
    ax = plt.gca()
    move = {0: (-0.8, 0), 1: (0, 0.8), 2: (0.8, 0), 3: (0, -0.8)}
    for (r, c), a in path:
        dr, dc = move[a]
        ax.add_patch(FancyArrow(c, r, dc, dr,
                                width=0.1,
                                head_width=0.3,
                                head_length=0.3,
                                length_includes_head=True,
                                color="blue"))
    plt.title(title)
    plt.show()

# %%
# ══════════════════════════════════════════════════════════
# 5.  Demo run
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    env = GridWorld(num_blocks=7, wind_prob=0)

    Q_q = q_learning(env)
    Q_s = sarsa(env)

    print("Greedy policy (Q-Learning)")
    print_policy(extract_policy(Q_q))
    draw_path(env, run_greedy_episode(env, Q_q), "Path ­— Q-Learning")

    print("\nGreedy policy (SARSA)")
    print_policy(extract_policy(Q_s))
    draw_path(env, run_greedy_episode(env, Q_s), "Path ­— SARSA")
