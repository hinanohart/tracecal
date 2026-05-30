# tracecal

**Conformal-calibrated, URDF physics-gated validity & abstention auditing for LeRobot datasets.**

`tracecal` audits a [LeRobot](https://github.com/huggingface/lerobot) robot-learning dataset and
returns, per episode, a calibrated verdict — **accept / hold / reject** — built on three things no
existing LeRobot data-quality tool combines:

1. **A multiplicative URDF joint-limit physics gate.** For industrial arms (Franka Panda, KUKA
   iiwa, and any arm you point at a plain URDF) an episode that drives a joint past its hard
   position/velocity limit is *kinematically impossible*, so its score is forced to `Q = 0` —
   irrespective of how clean the video or smooth the motion looks.
2. **Distribution-free conformal abstention.** When validity cannot be called confidently the
   episode is held back (`abstain`) rather than guessed.
3. **Degrade-first-class honesty.** Embodiments with no resolvable URDF (SO-101, Koch, LeKiwi —
   the majority of Hub datasets by count) are reported in a physics-skipped `hold` state with
   `coverage = None`; they are never silently treated as validated.

This is deliberately **not** a CV/heuristic quality scorer like
[`score_lerobot_episodes`](https://github.com/RoboticsData/score_lerobot_episodes) (blur, motion
smoothness, optional VLM checks) or a motion-consistency filter — those are complementary.
tracecal's value is a *physically grounded, distribution-free* validity gate with explicit
abstention. GPU-free, torch-free at runtime, MIT.

> **Scope (honest).** The physics-gate claim targets **industrial-arm** datasets — fewer in number
> on the Hub but the largest by episode volume (professional labs). Hobbyist arms run in degrade
> mode. See *Measured results* for exactly what is and isn't demonstrated in v0.1.0a1.

## Status

**v0.1.0a1 — pre-alpha.** The validated claim is the physics gate; conformal coverage is a
diagnostic that becomes a guarantee only under the label precondition below.

## Install

```bash
pip install tracecal              # core (numpy / scipy / pydantic)
pip install "tracecal[physics]"   # + URDF joint-limit gate (yourdfpy, robot_descriptions)
pip install "tracecal[hub]"       # + load real LeRobotDataset v3 from the Hub (pyarrow)
```

## Quickstart

```python
from tracecal import evaluate_dataset

report = evaluate_dataset("lerobot/pusht", confidence=0.9)   # local dir or HF repo id
print(report.n_accept, report.n_hold, report.n_reject, report.n_degraded)
```

```bash
tracecal run path/to/dataset --confidence 0.9 --format html -o report.html
tracecal selftest          # self-contained physics-gate check (no network/GPU)
tracecal list-embodiments  # which robot_types resolve to a URDF vs degrade
```

Use it as a CI gate via the pytest plugin:

```python
def test_dataset_is_clean(tracecal_audit):
    report = tracecal_audit("path/or/repo_id", confidence=0.9)
    tracecal_audit.assert_coverage_holds(report)   # no-op in reference-mode; fails on a breach
```

## Conformal coverage: the label precondition

Conformal coverage is a **validated finite-sample guarantee only when real binary validity labels
are supplied** for calibration. With no such labels tracecal runs in **reference-mode**
(`coverage = None`) and reports the physics gate plus a self-supervised abstention heuristic
without claiming a coverage number. Calibration figures in this repo are synthetic and for
**algorithm validation only**.

## Measured results (v0.1.0a1)

All numbers below are generated from `results/*.json` (`python scripts/measure.py`); they are not
hand-written.

| Tier | What | Result |
|---|---|---|
| Physics gate (CLAIM) | Adversarial synthetic trajectories on a **real Franka Panda URDF**: fraction of kinematically-invalid episodes forced to `Q = 0` | **1.00** |
| Calibration (synthetic, `algorithm validation only`) | Split-conformal holdout coverage vs target `0.90` over 80 trials | **0.90** empirical |
| Calibration (synthetic) | Isotonic reliability error (ECE) of the calibrated validity score | **0.045** |
| Real data (live) | `lerobot/pusht`, 50 episodes — unmapped robot_type → degrade-first-class (`hold`, `coverage = None`) | degrade demonstrated |

What is **not** claimed in v0.1.0a1: a physics-gate firing on a *real* industrial-arm episode (real
robots stay within their own limits, so the gate is demonstrated with adversarial synthetic
trajectories on a real URDF); cross-embodiment normalization; and a success-probability coverage
guarantee. Live industrial-arm dataset runs are **deferred to v0.1.1**.

## How it works

```
load v3 episodes ─▶ resolve embodiment URDF ─▶ kinematic gate ─▶ conformal / reference-mode ─▶ verdict
   (pyarrow)          (robot_descriptions)       (Q=hard·quality)   (split/Mondrian, abstain)   accept/hold/reject
```

* `reject` ⟺ a hard kinematic violation forced `Q = 0`.
* `hold` ⟺ degrade (no URDF) or non-confident (non-singleton conformal set / reference-mode outlier).
* `accept` ⟺ hard-valid and confidently valid.

## License

MIT. See `LICENSE` and `NOTICE` (third-party / URDF source license matrix). tracecal bundles
no manufacturer URDF, dataset, or model weights.
