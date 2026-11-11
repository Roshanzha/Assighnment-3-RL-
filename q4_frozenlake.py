import numpy as np
from common import *
np.set_printoptions(precision=4, suppress=True)
Q, succ, expl = train(2000, 0.5, 0.9, lambda ep: 0.2, max_steps=100, seed=42)
print("Final Q-table (rows = states 0..15; cols = Left, Down, Right, Up)")
print("state |   Left     Down    Right      Up")
for s in range(16):
    print(f"{s:5d} | " + " ".join(f"{q:8.4f}" for q in Q[s]))
pol = greedy_policy(Q)
print("\nGreedy policy pi(s)=argmax_a Q(s,a):")
for s in range(16):
    tag = " (terminal)" if is_terminal(s) else ""
    print(f"  state {s:2d}: {pol[s]} ({ACTION_NAMES[int(pol[s])]}){tag}")
print("\nPolicy grid:"); print_policy(Q)
s, path = 0, [0]
for _ in range(100):
    s, r, d = step(s, int(pol[s])); path.append(s)
    if d: break
print("\nGreedy path from start:", path, "-> reached goal" if path[-1]==15 else "-> failed")
print(f"Training successes: {succ.sum()}/2000, last-100 success rate: {succ[-100:].mean():.2f}")
