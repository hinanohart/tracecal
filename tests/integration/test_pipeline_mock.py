"""S3 integration: io -> physics -> gate -> curate on synthetic episodes (no network/yourdfpy)."""

from __future__ import annotations

import numpy as np

from tracecal.curate import curate
from tracecal.io.lerobot_v3 import from_arrays
from tracecal.physics.gate import gated_score
from tracecal.physics.kinematics import check_episode
from tracecal.schema import DatasetReport, EpisodeVerdict


def _clean(n: int = 30) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return np.stack([0.5 * np.sin(t), 0.3 * np.cos(t)], axis=1)


def _violating(n: int = 30) -> np.ndarray:
    pos = _clean(n)
    pos[n // 2, 0] = 3.0  # drive j1 past its 1.5708 limit -> kinematically impossible
    return pos


def test_full_offline_pipeline_gates_invalid_to_zero(synth_2dof_spec) -> None:
    ds = from_arrays(
        robot_type="synth_2dof",
        fps=10.0,
        joint_names=("j1", "j2"),
        episodes=[("clean0", _clean()), ("clean1", _clean()), ("bad0", _violating())],
        dataset="synthetic",
    )
    verdicts: list[EpisodeVerdict] = []
    n_reject = n_accept = 0
    for ep in ds.episodes:
        phys = check_episode(
            ep.positions, spec=synth_2dof_spec, fps=ds.fps, episode_id=ep.record.id
        )
        # reference-mode quality = physics pass-rate (no labels available)
        g = gated_score(phys.pass_rate, phys)
        if g.gated_out:
            verdict, n_reject = "reject", n_reject + 1
        else:
            verdict, n_accept = "accept", n_accept + 1
        verdicts.append(
            EpisodeVerdict(
                episode_id=ep.record.id,
                verdict=verdict,
                Q=g.Q,
                hard_valid=phys.hard_valid,
                abstain=False,
                degraded=False,
            )
        )

    # the violating episode must be rejected with Q exactly 0; clean ones accepted with Q>0
    by_id = {v.episode_id: v for v in verdicts}
    assert by_id["bad0"].verdict == "reject"
    assert by_id["bad0"].Q == 0.0
    assert by_id["clean0"].verdict == "accept"
    assert by_id["clean0"].Q > 0.0

    report = DatasetReport(
        dataset="synthetic",
        n_episodes=3,
        n_accept=n_accept,
        n_hold=0,
        n_reject=n_reject,
        n_degraded=0,
        verdicts=tuple(verdicts),
    )
    result = curate(report)
    assert set(result.kept_ids) == {"clean0", "clean1"}
    assert set(result.rejected_ids) == {"bad0"}
    assert result.weights["bad0"] == 0.0
    assert result.weights["clean0"] == 1.0


def test_degraded_embodiment_flows_to_hold(synth_2dof_spec) -> None:
    from tracecal.physics.gate import degraded_result

    phys = degraded_result("e_deg", "no-urdf-known", n_steps=20)
    g = gated_score(0.5, phys)
    assert g.degraded is True and g.gated_out is False
    v = EpisodeVerdict(
        episode_id="e_deg", verdict="hold", Q=g.Q, hard_valid=None, abstain=True, degraded=True
    )
    report = DatasetReport(
        dataset="d", n_episodes=1, n_accept=0, n_hold=1, n_reject=0, n_degraded=1, verdicts=(v,)
    )
    result = curate(report)
    assert result.held_ids == ("e_deg",)
    assert result.weights["e_deg"] == 0.0
