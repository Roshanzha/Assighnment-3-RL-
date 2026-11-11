# CS6109 – Reinforcement Learning – Assignment 3
**Name:** Roshan Jha  **Roll No:** 2611CS23

All algorithms are implemented from scratch with Python/NumPy (no RL libraries). Matplotlib is used only for plots.

## Files
| File | Purpose |
|---|---|
| `common.py` | Frozen Lake env, ε-greedy, TD/Q update, training loop |
| `q1_passive_adp.py` … `q5_decay.py` | One script per question (run `python qN_*.py`) |
| `outputs/*_output.txt` | Console output of each script |
| `outputs/q3_success_rate.png`, `outputs/q5_success_rate.png` | Required plots |

Run: `pip install numpy matplotlib` then `python q1_passive_adp.py` etc. Fixed seed = 42.

## Q1 – Passive ADP
Prior counts (5 tries of (s11,Down)): intended→s21 ×3, perpendicular Left→wall→s11 ×1, Right→s12 ×1.
After ⟨s11,Down,s21,−0.04⟩: **N(s11,Down)=6, N(s11,Down,s21)=4, P(s21|s11,Down)=4/6=0.6667** (P(s11)=P(s12)=1/6).
For (s21,Right) the same 3/1/1 pattern is assumed: s22 0.6, s11 0.2 (Up), s21 0.2 (Down→wall).
Bellman equations (V(s12)=+1, V(s22)=−1):
- V(s11) = −0.04 + 0.9[⅙V(s11) + ⅔V(s21) + ⅙(1)]
- V(s21) = −0.04 + 0.9[0.2V(s11) + 0.2V(s21) + 0.6(−1)]

Solution: **V(s11) = −0.4377, V(s21) = −0.8034**.

## Q2 – TD / Q-learning
Four sample transitions (including one terminal, where TD target = r); see `outputs/q2_td_qlearning_output.txt`.

## Q3 – Exploration vs exploitation (2000 episodes, window 100)
| ε | Successes | Final success rate (last 100) | Exploratory actions |
|---|---|---|---|
| 0 | 1900 | 1.00 | 0 |
| 0.1 | 1712 | 0.91 | 1242 |
| 0.3 | 1250 | 0.64 | 4197 |
| 0.5 | 750 | 0.46 | 7191 |

Greedy policies are printed in `outputs/q3_exploration_output.txt`.
**Note on ε = 0:** with *random tie-breaking* among equal Q-values the "greedy" agent still explores implicitly (all Q are 0 at start), so it succeeds. With *deterministic* `argmax` (ties → Left) the same ε=0 agent gets **0 successes**: it bumps into the wall at state 0 forever, all rewards stay 0, Q never changes (extra run at end of the Q3 output).
**Why pure greedy can fail:** reward is sparse (only at the goal). Until the goal is reached, every Q-value is 0, so a greedy agent always repeats the same action and never sees a reward; with no signal, Q never changes and it never discovers a better policy. It may also lock onto the first mediocre path it finds and never test alternatives. ε>0 gives random actions that eventually reach the goal. Larger ε (0.5) explores more but wastes many steps and lowers *training* success rate, since 50% of actions are random (the learned greedy policy is still correct).

## Q4 – Frozen Lake (α=0.5, γ=0.9, ε=0.2, 2000 episodes)
Final Q-table and policy in `outputs/q4_frozenlake_output.txt`. Learned greedy policy:
```
v > v <
v H v H
> > v H
H > > G
```
Greedy path from start: 0→4→8→9→10→14→15 (6 steps, reaches goal). Q(14,Right)=1.0, Q(13/10,·)=0.9, … decaying by γ per step.

## Q5 – Decaying exploration (ε₀=1.0, ε_min=0.05, decay 0.995 per episode; ε reaches ε_min at ≈ episode 598)
| Agent | Final success rate (last 100 training eps) | Total exploratory actions | Greedy path reaches goal |
|---|---|---|---|
| Fixed ε=0.2 | 0.77 | 2643 | Yes |
| Decaying ε | 0.96 | 1908 | Yes |

Effect: early on high ε explores the whole map and spreads reward information; later low ε exploits the learned values, so training success rate rises to ~0.96 versus a ceiling of ≈0.8 for fixed ε=0.2 (which keeps taking random actions, some into holes). Decaying ε trades exploration for exploitation over time, giving higher return and fewer wasted random actions. Q-tables/policies in `outputs/q5_decay_output.txt`.

## Short concept notes (for viva)
- **Model-based ADP** learns P(s'|s,a) from counts and solves Bellman equations for V; **model-free Q-learning** updates Q directly from samples without a model.
- **TD target** = r + γ max Q(s',a') (= r at terminal); **TD error** = target − Q(s,a); **α** = step size; **γ** = discount; **ε-greedy** = random action with prob ε else argmax Q.
- Implementation detail: ties in argmax are broken randomly (otherwise all-zero rows always pick "Left").
- Note: because results depend on the random seed, exact numbers can differ with another seed; the learned policies reach the goal for all 20 seeds tested.
