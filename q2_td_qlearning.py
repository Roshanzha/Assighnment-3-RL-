import numpy as np
alpha, gamma = 0.5, 0.9
Q = {("A","right"):0.0, ("A","down"):0.0, ("B","right"):0.0, ("B","down"):0.0,
     ("C","right"):0.0, ("C","down"):0.0}
# Q-table with a few non-zero entries so max_a' Q(s',a') is meaningful
Q[("B","right")] = 0.6; Q[("B","down")] = 0.2; Q[("C","right")] = 0.4; Q[("C","down")] = 0.1
Q[("A","right")] = 0.3
actions = ["right", "down"]
terminal = {"G"}
# (s, a, r, s', description)
transitions = [
    ("A", "right", 0.0, "B", "non-terminal"),
    ("B", "down",  0.0, "C", "non-terminal"),
    ("C", "right", 1.0, "G", "terminal (goal)"),
    ("A", "right", 0.0, "B", "non-terminal, same (s,a) again"),
]
print(f"alpha={alpha}, gamma={gamma}\n")
for i, (s, a, r, s2, d) in enumerate(transitions, 1):
    cur = Q[(s, a)]
    if s2 in terminal:
        target = r
    else:
        target = r + gamma * max(Q[(s2, x)] for x in actions)
    err = target - cur
    Q[(s, a)] = cur + alpha * err
    print(f"Transition {i}: <{s}, {a}, r={r}, {s2}>  [{d}]")
    print(f"  Current Q   = {cur:.4f}")
    print(f"  TD Target   = {target:.4f}")
    print(f"  TD Error    = {err:.4f}")
    print(f"  Updated Q   = {Q[(s,a)]:.4f}\n")
