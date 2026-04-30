# RL Uncertainty Demo — Experiment Progress

## Current Best Result: v6 (collision=-25)
- **Config:** continuous_nav, ensemble K=5, 200k steps, seed=42, collision=-25, goal=50, shaping=20
- **Training:** 98% goal rate late, 2% collision
- **Eval:** Risk-neutral 15/15 goals, LCB (beta=0.3) 15/15 goals, LCB (beta=3.0) 10/10 goals
- **EU Heatmap:** Strong teardrop hot spot around center obstacle
- **LCB:** Robust at all beta values tested (0.3-3.0) — sweet spot found
- **Figure:** `/tmp/claude/rl_v6/triptych.png`, `/tmp/claude/rl_v6/quad_triptych.png`
- **Training time:** 174s (~3 min)

## Architecture Changes Summary

### Reward Redesign (Critical Fix)
The original reward (-1/step, -10 collision, +10 goal) caused **suicide strategy**: the agent preferred crashing quickly (-20 total) over timing out (-200 total). Fixed iteratively:

| Version | Step | Collision | Goal | Shaping | Result |
|---------|------|-----------|------|---------|--------|
| v1 (original) | -1.0 | -10 | +10 | 0 | 97.8% collision — agent suicides |
| v2 | -0.1 | -50 | +10 | 5 | 0% goals — shaping too weak |
| v3 | -0.1 | -50 | +50 | 20 | 27% goals at 50k steps |
| v4 | 0.0 | -50 | +50 | 20 | 100% goals! But LCB freezes agent |
| v5 | 0.0 | -10 | +50 | 20 | 100% goals, LCB works, weak heatmap |
| **v6** | **0.0** | **-25** | **+50** | **20** | **98% goals, LCB robust, good heatmap** |

**Key insight:** Distance-based reward shaping (`reward += 20 * (prev_dist - cur_dist)`) was essential. Without it, the +50 goal reward is too sparse for random exploration to discover.

### Obstacle Layout Simplified
Original 4 obstacles (too dense, narrow corridors) -> 2 obstacles:
- 1 large center obstacle (0.5, 0.5) r=0.15 — blocks direct start-to-goal path
- 1 small upper-left obstacle (0.3, 0.75) r=0.07 — adds asymmetry

Goal radius increased 0.05 -> 0.08 for easier random discovery.

### Visualization Improvements
- **Spaghetti plot:** Added per-ensemble-member greedy trajectory visualization. Each member's Q-network picks its own greedy action, producing different paths. The spread visualizes epistemic uncertainty in action space.
- **4-panel prototype:** Tested layout with (a) member spaghetti, (b) greedy vs LCB, (c) EU heatmap, (d) decomposition.
- **Decomposition smoothing:** Fixed crash when uncertainty log has few entries (window > data size).

## LCB Deep-Dive (Key Finding)

Analyzed Q-values step-by-step along the greedy trajectory for v4 (collision=-50):

```
At t=4 (x=0.31, approaching obstacle):
  right: Q=51.6 +/- 2.31  (high Q but high uncertainty)
  down:  Q=51.5 +/- 1.57  (similar Q but lower uncertainty)
  → LCB switches from "right" to "down" — retreats to safety
```

**The trade-off:** Higher collision penalty -> ensemble members learn more diverse avoidance strategies -> higher Q-std near obstacles -> LCB becomes too conservative. Lower collision penalty -> members converge -> lower Q-std -> LCB works but heatmap is less dramatic.

**v6 (collision=-25) hits the sweet spot:** enough diversity for a good heatmap, but not so much that LCB breaks.

## Method Comparison

### Ensemble (v6, K=5) — Best overall
- 98% goal rate, LCB robust
- EU range: 0.05-0.35, concentrated teardrop around obstacle
- Per-member trajectories show clear path diversity

### Credal Relative Likelihood — Different uncertainty profile
- **Config:** single agent trained 200k steps, wrapped with `credal_relative_likelihood(K=5)`
- **Training:** 96% goal rate (2064/2730 goals)
- **Eval:** greedy 10/10 goals, LCB (beta=1.0) 10/10 goals
- **EU range:** 1.12-1.33 — much higher baseline, less contrast
- **Heatmap:** More uniform EU everywhere, hot spots still around obstacles but less dramatic
- **Decomposition:** Shows epistemic decreasing and aleatoric increasing — interesting dynamics
- **No member trajectories** (credal wraps a single network, no separate agents)

### K=10 Ensemble — Denser spaghetti (evaluated, awaiting figure)
- **Training:** 98% goal rate, same as K=5
- **Training time:** 337s (~5.6 min) — 2x slower than K=5
- 10 member trajectories should produce a denser, more visually striking spaghetti plot
- **Status:** trained, needs evaluation + figure generation

## Racetrack Results

### v1 (original rewards)
- **Config:** +1/step, -10 wall, +20 finish, K=5, 200k steps
- **Result:** 0% finish, 97% wall crashes. Agent doesn't learn to drive.

### v2 (angle shaping)
- **Config:** angle_shaping=10, wall=-25, finish=+50, K=5, 200k steps
- **Result:** 18% finish rate late, 78% wall crashes. Improving but still poor.
- **EU range:** 0.44-0.54 — high and increasing (agents disagree a lot)
- **Diagnosis:** 4D state space (x,y,vx,vy) with momentum physics is much harder for DQN. May need:
  - More training steps (500k+)
  - Larger network (128 hidden instead of 64)
  - Easier track geometry (wider corridor)
  - Different algorithm (SAC/PPO for continuous control)
- **Status:** Evaluated but poor quality for a figure

## Experiment Log (Chronological)

| Run | Env | Method | Collision | Steps | Goal Rate (late) | LCB | Heatmap | Notes |
|-----|-----|--------|-----------|-------|-----------------|-----|---------|-------|
| v1 | nav | ensemble K=3 | -10 | 5k | 0% | N/A | N/A | Smoke test only |
| v2 | nav | ensemble K=5 | -10 (orig) | 50k | 1.2% | N/A | N/A | Suicide strategy |
| v3 | nav | ensemble K=5 | -50 | 50k | 0% | N/A | N/A | Shaping too weak |
| v3b | nav | ensemble K=5 | -50 | 50k | 27% | N/A | N/A | Better shaping |
| v4 | nav | ensemble K=5 | -50 | 200k | 100% | Broken | Excellent | Best heatmap |
| v5 | nav | ensemble K=5 | -10 | 200k | 100% | Works (0.3) | Weak | LCB works |
| **v6** | **nav** | **ensemble K=5** | **-25** | **200k** | **98%** | **Robust (3.0)** | **Good** | **Best overall** |
| v7 | nav | ensemble K=10 | -25 | 200k | 98% | TBD | TBD | Denser spaghetti |
| credal | nav | credal K=5 | -25 | 200k | 96% | Works (1.0) | Different | Higher EU baseline |
| race v1 | track | ensemble K=5 | -10 | 200k | 0% | N/A | N/A | Agent can't drive |
| race v2 | track | ensemble K=5 | -25 | 200k | 18% | TBD | TBD | Still poor |

## Open Issues

1. **Racetrack needs more work.** DQN struggles with the 4D state + momentum physics. Options: more training, bigger network, wider track, or accept it as a "hard problem" and only show continuous nav.

2. **Decomposition panel (c) is the weakest panel.** Epistemic doesn't clearly decrease over training — it fluctuates. The credal method shows clearer dynamics (epistemic down, aleatoric up). Consider: using credal for panel (c), or showing a histogram of EU values instead.

3. **K=10 spaghetti not yet visualized.** Expected to look more striking with 10 colored paths.

4. **No penalized reward training yet.** The plan included training with `PenalizedRewardWrapper` (reward -= lambda * EU during training). This would produce a truly risk-averse agent for comparison. Not yet attempted.

## Files Changed Since Last Commit
- `experiments/rl_uncertainty/envs/continuous_nav.py` — collision_reward=-25 (was -10 after last commit)
- `experiments/rl_uncertainty/envs/racetrack.py` — added angle_shaping, wall_reward, finish_reward
- `experiments/rl_uncertainty/scripts/evaluate.py` — added per-member rollouts
- `experiments/rl_uncertainty/viz/trajectories.py` — added spaghetti plot function
- `experiments/rl_uncertainty/viz/triptych.py` — auto-select member plot

## Next Steps
- [ ] Evaluate and visualize K=10 ensemble (v7) — expect denser spaghetti
- [ ] Try penalized reward training (risk-averse agent for comparison)
- [ ] Decide: racetrack worth pursuing or focus on continuous nav?
- [ ] Polish figure aesthetics (font sizes, colors, NeurIPS column width)
- [ ] Try more seeds for robustness check
- [ ] Commit racetrack reward redesign + progress
