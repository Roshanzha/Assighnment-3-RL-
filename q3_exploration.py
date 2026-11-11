import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
from common import *
EPISODES, W = 2000, 100
plt.figure(figsize=(8,5))
for eps in [0, 0.1, 0.3, 0.5]:
    Q, succ, expl = train(EPISODES, 0.5, 0.9, lambda ep, e=eps: e, seed=42)
    rate = window_rate(succ, W)
    plt.plot(np.arange(1, len(rate)+1)*W, rate, marker="o", ms=3, label=f"eps={eps}")
    print(f"===== epsilon = {eps} =====")
    print(f"  Total successful episodes : {succ.sum()} / {EPISODES}")
    print(f"  Final success rate (last {W} episodes): {succ[-W:].mean():.2f}   (overall: {succ.mean():.3f})")
    print(f"  Total exploratory actions : {expl}")
    print(f"  Greedy policy reaches goal: {greedy_reaches_goal(Q)}")
    print("  Learned greedy policy:"); print_policy(Q); print()
plt.xlabel("Training episodes"); plt.ylabel(f"Success rate (window = {W} episodes)")
plt.title("Q3: Success rate vs episodes for different epsilon"); plt.grid(alpha=.3); plt.legend()
plt.tight_layout(); plt.savefig("outputs/q3_success_rate.png", dpi=150)

# ---- Extra: purely greedy agent with DETERMINISTIC tie-breaking (plain np.argmax, ties -> Left)
common.TIE_BREAK = "first"
Q, succ, expl = common.train(EPISODES, 0.5, 0.9, lambda ep: 0.0, seed=42)
print("===== EXTRA: eps = 0 with deterministic argmax tie-breaking (first index = Left) =====")
print(f"  Total successful episodes : {succ.sum()} / {EPISODES}; final success rate: {succ[-W:].mean():.2f}; exploratory actions: {expl}")
print("  Learned greedy policy:"); common.print_policy(Q)
print("  Q-table non-zero entries:", int((Q != 0).sum()))
