# tracecal

**Conformal-calibrated, URDF physics-gated validity & abstention auditing for LeRobot datasets.**

`tracecal` audits a [LeRobot](https://github.com/huggingface/lerobot) robot-learning dataset
and returns, per episode, a calibrated validity verdict — **accept / hold / reject** — built
on three things no existing LeRobot data-quality tool combines:

1. **A multiplicative URDF joint-limit physics gate.** For industrial arms (Franka Panda,
   Universal Robots, KUKA iiwa, UFACTORY xArm) whose URDF is resolvable, an episode that drives
   a joint past its hard position/velocity limit is *kinematically impossible*, so its score is
   forced to `Q = 0` — irrespective of how clean the video or smooth the motion looks.
2. **Distribution-free conformal abstention.** When validity cannot be called confidently the
   episode is held back (`abstain`) rather than guessed.
3. **Degrade-first-class honesty.** Embodiments with no resolvable URDF (SO-101, Koch, LeKiwi —
   the majority of Hub datasets by count) are *not* silently "validated": they are reported in a
   physics-skipped `hold` state with `coverage=None`.

This is deliberately **not** a CV/heuristic quality scorer like
[`score_lerobot_episodes`](https://github.com/RoboticsData/score_lerobot_episodes) (blur, motion
smoothness, VLM checks) or motion-consistency filters — those are complementary. tracecal's value
is a *physically grounded, distribution-free* validity gate.

> Scope (honest): the physics-gate CLAIM targets **industrial-arm** datasets — fewer in number on
> the Hub but the largest by episode volume (professional labs). Hobbyist arms run in degrade mode.

GPU-free. torch-free at runtime. Apache-2.0.

## Status

v0.1.0a1 — pre-alpha. <!--MEASURED@S6: headline numbers are injected from results/*.json at S7; do not hand-write.-->

## Install

```bash
pip install tracecal              # core (numpy/scipy/pydantic)
pip install "tracecal[physics]"   # + URDF joint-limit gate (yourdfpy, robot_descriptions)
pip install "tracecal[hub]"       # + load real LeRobotDataset v3 from the Hub
```

## Quickstart

<!--MEASURED@S6--> (filled at S7 from the measured run; outputs marked `# illustrative` are not measured.)

## Conformal coverage: the label precondition

Conformal coverage is a **validated finite-sample guarantee only when real binary validity labels
are supplied** for calibration. With no such labels tracecal runs in **reference-mode**
(`coverage=None`) and reports the physics gate + abstention without claiming a coverage number.
Synthetic calibration figures in this repo are for **algorithm validation only** and are labelled
as such.

## License

Apache-2.0. See `LICENSE` and `NOTICE` (third-party / URDF source license matrix).
