import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
np.set_printoptions(precision=4, suppress=True)
EP, W = 2000, 100
eps_min, decay = 0.05, 0.995
fixed_fn = lambda ep: 0.2
decay_fn = lambda ep: max(eps_min, 1.0 * decay ** ep)   # eps_{t+1}=max(eps_min, eps_t*decay), eps_0=1.0
res = {}
for name, fn in [("Fixed eps=0.2", fixed_fn), ("Decaying eps", decay_fn)]:
    Q, succ, expl = train(EP, 0.5, 0.9, fn, seed=42)
    res[name] = (Q, succ, expl)
    print(f"===== {name} =====")
    print(f"  Final success rate (last {W} training episodes): {succ[-W:].mean():.2f}  (overall {succ.mean():.3f}, total successes {succ.sum()})")
    print(f"  Total exploratory actions: {expl}")
    print(f"  Greedy policy reaches goal: {greedy_reaches_goal(Q)}")
    print("  Final Q-table (Left, Down, Right, Up):")
    for s in range(16): print(f"   {s:2d}: " + " ".join(f"{q:8.4f}" for q in Q[s]))
    print("  Learned greedy policy:"); print_policy(Q); print()
print(f"Decay epsilon reaches eps_min=0.05 at episode ~{int(np.ceil(np.log(eps_min)/np.log(decay)))}")
plt.figure(figsize=(8,5))
for name,(Q,succ,_) in res.items():
    r = window_rate(succ, W); plt.plot(np.arange(1,len(r)+1)*W, r, marker="o", ms=3, label=name)
plt.xlabel("Training episodes"); plt.ylabel(f"Success rate (window = {W})")
plt.title("Q5: Fixed vs decaying epsilon"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
plt.savefig("outputs/q5_success_rate.png", dpi=150)
