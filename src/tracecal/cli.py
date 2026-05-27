"""tracecal command-line interface (stdlib argparse; no extra runtime dependency).

Commands:
  run <source>          audit a dataset (local dir / Hub repo id) -> JSON/CSV/HTML report.
                        Exits 2 when a measured conformal coverage breaches the target.
  selftest              run a self-contained physics-gate check (core install, no network/GPU).
  list-embodiments      show which robot_types resolve to a URDF vs degrade-first-class.
"""

from __future__ import annotations

import argparse
import sys

from tracecal import __version__


def _cmd_run(args: argparse.Namespace) -> int:
    from tracecal.api import evaluate_dataset
    from tracecal.report import emit

    report = evaluate_dataset(
        args.source,
        confidence=args.confidence,
        mondrian_by=None if args.mondrian_by == "none" else args.mondrian_by,
        physics=args.physics,
        max_episodes=args.max_episodes,
        state_key=args.state_key,
    )
    rendered = {"json": emit.to_json, "csv": emit.to_csv, "html": emit.to_html}[args.format](report)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(rendered)
        print(f"wrote {args.format} report to {args.output}", file=sys.stderr)
    else:
        print(rendered)

    print(
        f"summary: accept={report.n_accept} hold={report.n_hold} reject={report.n_reject} "
        f"degraded={report.n_degraded}",
        file=sys.stderr,
    )
    cov = report.coverage
    if cov is not None and ((cov.ci_high != cov.ci_high) or cov.ci_high < args.confidence):
        print(
            f"coverage breach: empirical {cov.empirical_coverage:.3f} (95% CI upper "
            f"{cov.ci_high:.3f}) < target {args.confidence:.3f}",
            file=sys.stderr,
        )
        return 2
    return 0


def _cmd_selftest(_args: argparse.Namespace) -> int:
    """Exercise the physics hard-gate end-to-end on synthetic data (no network, no yourdfpy)."""
    import numpy as np

    from tracecal.physics.gate import gated_score
    from tracecal.physics.kinematics import check_episode
    from tracecal.report.card import decide_verdict
    from tracecal.schema import EmbodimentSpec, JointLimits

    spec = EmbodimentSpec(
        robot_type="selftest_2dof",
        resolved=True,
        source="selftest",
        dof=2,
        joints=(
            JointLimits(name="j1", lower=-1.0, upper=1.0, velocity=5.0),
            JointLimits(name="j2", lower=-1.0, upper=1.0, velocity=5.0),
        ),
    )
    t = np.linspace(0, 1, 20)
    clean = np.stack([0.5 * np.sin(t), 0.3 * np.cos(t)], axis=1)
    bad = clean.copy()
    bad[:, 0] = np.linspace(0.9, 1.5, 20)  # ramp past j1 limit (1.0): position-only violation

    pc = check_episode(clean, spec=spec, fps=10.0, episode_id="clean")
    pb = check_episode(bad, spec=spec, fps=10.0, episode_id="bad")
    vc = decide_verdict(pc, gated_score(pc.pass_rate, pc), abstain=False)
    vb = decide_verdict(pb, gated_score(pb.pass_rate, pb), abstain=False)

    ok = vc.verdict == "accept" and vc.Q > 0.0 and vb.verdict == "reject" and vb.Q == 0.0
    if not ok:
        print(
            f"selftest FAILED: clean={vc.verdict}/{vc.Q} bad={vb.verdict}/{vb.Q}", file=sys.stderr
        )
        return 1
    print("selftest OK: clean episode accepted (Q>0), kinematically-invalid episode rejected (Q=0)")
    return 0


def _cmd_list_embodiments(_args: argparse.Namespace) -> int:
    from tracecal.physics.registry import iter_registry, known_no_urdf_keys

    print("Resolvable robot_type substrings -> robot_descriptions module (physics-gated):")
    for key, module in iter_registry():
        print(f"  {key:10s} -> {module}")
    print("\nKnown degrade-first-class embodiments (no resolvable URDF; reported as hold):")
    print("  " + ", ".join(known_no_urdf_keys()))
    print(
        "\nNote: Franka Panda and KUKA iiwa expose a plain URDF with joint limits and are the "
        "confidently-gated arms in v0.1.0a1; others may degrade unless a URDF is supplied."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tracecal", description=__doc__.splitlines()[0] if __doc__ else None
    )
    p.add_argument("--version", action="version", version=f"tracecal {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="audit a LeRobot dataset")
    run.add_argument("source", help="local v3 dataset dir, or a Hugging Face dataset repo id")
    run.add_argument(
        "--confidence", type=float, default=0.9, help="target coverage 1-alpha (default 0.9)"
    )
    run.add_argument(
        "--mondrian-by", choices=["embodiment", "none"], default="embodiment", dest="mondrian_by"
    )
    run.add_argument("--physics", choices=["auto", "off"], default="auto")
    run.add_argument("--format", choices=["json", "csv", "html"], default="json")
    run.add_argument("--output", "-o", default=None, help="write report to FILE (default: stdout)")
    run.add_argument("--max-episodes", type=int, default=None, dest="max_episodes")
    run.add_argument("--state-key", default="observation.state", dest="state_key")
    run.set_defaults(func=_cmd_run)

    st = sub.add_parser("selftest", help="run a self-contained physics-gate check")
    st.set_defaults(func=_cmd_selftest)

    le = sub.add_parser(
        "list-embodiments", help="list resolvable vs degrade-first-class embodiments"
    )
    le.set_defaults(func=_cmd_list_embodiments)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
