import numpy as np

# Frozen Lake 4x4 (deterministic). Actions: 0=Left,1=Down,2=Right,3=Up
MAP = ["SFFF", "FHFH", "FFFH", "HFFG"]
N_STATES, N_ACTIONS = 16, 4
MOVES = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
ACTION_NAMES = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
ARROWS = {0: "<", 1: "v", 2: ">", 3: "^"}

def is_terminal(s):
    return MAP[s // 4][s % 4] in "HG"

def step(s, a):
    """Deterministic transition. Invalid move -> stay. Reward +1 only on Goal."""
    r, c = divmod(s, 4)
    dr, dc = MOVES[a]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < 4 and 0 <= nc < 4):
        nr, nc = r, c
    s2 = nr * 4 + nc
    cell = MAP[nr][nc]
    return s2, (1.0 if cell == "G" else 0.0), cell in "HG"

TIE_BREAK = "random"   # "random" (default) or "first" (plain np.argmax)

def greedy_action(Q, s, rng):
    """argmax with RANDOM tie-breaking (important when Q-row is all zeros)."""
    if TIE_BREAK == "first":
        return int(np.argmax(Q[s]))
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(rng.choice(best))

def epsilon_greedy(Q, s, eps, rng):
    """Returns (action, was_exploratory)."""
    if rng.random() < eps:
        return int(rng.integers(N_ACTIONS)), True
    return greedy_action(Q, s, rng), False

def td_update(Q, s, a, r, s2, done, alpha, gamma):
    target = r if done else r + gamma * np.max(Q[s2])
    Q[s, a] += alpha * (target - Q[s, a])

def train(episodes, alpha=0.5, gamma=0.9, eps_fn=lambda ep: 0.2, max_steps=100, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((N_STATES, N_ACTIONS))
    successes, explore_total = [], 0
    for ep in range(episodes):
        eps = eps_fn(ep)
        s, ok = 0, 0
        for _ in range(max_steps):
            a, expl = epsilon_greedy(Q, s, eps, rng)
            explore_total += expl
            s2, r, done = step(s, a)
            td_update(Q, s, a, r, s2, done, alpha, gamma)
            s = s2
            if done:
                ok = int(r == 1.0)
                break
        successes.append(ok)
    return Q, np.array(successes), explore_total

def greedy_policy(Q):
    return np.argmax(Q, axis=1)

def print_policy(Q):
    pol = greedy_policy(Q)
    for r in range(4):
        row = []
        for c in range(4):
            s = r * 4 + c
            ch = MAP[r][c]
            row.append(ch if ch in "HG" else ("." if not Q[s].any() else ARROWS[int(pol[s])]))
        print("  " + " ".join(row))
    print("  (arrows = greedy action; '.' = all-zero Q row, i.e. no learned preference)")

def greedy_reaches_goal(Q, max_steps=100):
    s = 0
    for _ in range(max_steps):
        s, r, done = step(s, int(np.argmax(Q[s])))
        if done:
            return r == 1.0
    return False

def window_rate(succ, w=100):
    return succ[: len(succ) // w * w].reshape(-1, w).mean(axis=1)
