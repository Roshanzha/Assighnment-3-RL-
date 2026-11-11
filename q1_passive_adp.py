import numpy as np
gamma = 0.9
R = {"s11": -0.04, "s21": -0.04, "s12": 1.0, "s22": -1.0}
# Grid:  s11 s12(+1) / s21 s22(-1). Terminals: s12, s22 (V = R).
# Initially N(s,a)=5 for every (s,a).
# Observed pattern for the 5 prior tries: intended 3x, each perpendicular direction 1x.

# ---- (s11, Down): intended -> s21 (3), perpendicular Left -> wall -> s11 (1), Right -> s12 (1)
N_sa = 5
N_sas = {"s21": 3, "s11": 1, "s12": 1}
print("BEFORE observation <s11, Down, s21, -0.04>")
print(f"  N(s11,Down)={N_sa}, N(s11,Down,s21)={N_sas['s21']}, P(s21|s11,Down)={N_sas['s21']/N_sa:.4f}")
# ---- New observation
N_sa += 1
N_sas["s21"] += 1
P11 = {s: n / N_sa for s, n in N_sas.items()}
print("\n(i) AFTER observation")
print(f"  N(s11,Down)       = {N_sa}")
print(f"  N(s11,Down,s21)   = {N_sas['s21']}")
print(f"  P(s21|s11,Down)   = {N_sas['s21']}/{N_sa} = {P11['s21']:.4f}")
print(f"  P(s11|s11,Down)   = {P11['s11']:.4f},  P(s12|s11,Down) = {P11['s12']:.4f}  (sum={sum(P11.values()):.2f})")

# ---- (s21, Right): intended -> s22 (3), perpendicular Up -> s11 (1), Down -> wall -> s21 (1)
P21 = {"s22": 3/5, "s11": 1/5, "s21": 1/5}
print("\n  Assumed model for (s21,Right) with N=5:", P21)

# ---- (ii) Bellman equations under fixed policy: V(s)=R(s)+gamma*sum_s' P(s'|s,pi(s)) V(s')
# Unknowns x=[V11, V21]; V(s12)=+1, V(s22)=-1 (terminal)
# V11 = R11 + g*(P11[s11]*V11 + P11[s21]*V21 + P11[s12]*1)
# V21 = R21 + g*(P21[s11]*V11 + P21[s21]*V21 + P21[s22]*(-1))
A = np.array([
    [1 - gamma * P11["s11"], -gamma * P11["s21"]],
    [-gamma * P21["s11"],     1 - gamma * P21["s21"]],
])
b = np.array([
    R["s11"] + gamma * P11["s12"] * R["s12"],
    R["s21"] + gamma * P21["s22"] * R["s22"],
])
print("\n(ii) Bellman equations:")
print(f"  V(s11) = -0.04 + 0.9[{P11['s11']:.4f} V(s11) + {P11['s21']:.4f} V(s21) + {P11['s12']:.4f}(+1)]")
print(f"  V(s21) = -0.04 + 0.9[{P21['s11']:.4f} V(s11) + {P21['s21']:.4f} V(s21) + {P21['s22']:.4f}(-1)]")
V11, V21 = np.linalg.solve(A, b)
print(f"\n  V(s11) = {V11:.4f}\n  V(s21) = {V21:.4f}")
print(f"  (terminal: V(s12)=+1, V(s22)=-1)")
# verification
print("\nCheck residuals:",
      V11 - (R['s11'] + gamma*(P11['s11']*V11 + P11['s21']*V21 + P11['s12']*1)),
      V21 - (R['s21'] + gamma*(P21['s11']*V11 + P21['s21']*V21 + P21['s22']*-1)))
