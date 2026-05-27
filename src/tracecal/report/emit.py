"""Serialise a :class:`DatasetReport` to JSON, CSV, or a self-contained HTML card.

The HTML is a single file with inline CSS and no external dependencies (open it directly in a
browser). All three formats surface the honesty caveats verbatim — reference-mode (no coverage
claim), degrade-first-class counts, and the conformal label precondition — so a downstream
reader cannot mistake a degraded/reference run for a validated coverage guarantee.
"""

from __future__ import annotations

import csv
import html
import io
import json

from tracecal.schema import DatasetReport


def to_json(report: DatasetReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True)


def to_csv(report: DatasetReport) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["episode_id", "verdict", "Q", "hard_valid", "abstain", "degraded", "reasons"])
    for v in report.verdicts:
        writer.writerow(
            [
                v.episode_id,
                v.verdict,
                f"{v.Q:.6f}",
                "" if v.hard_valid is None else str(v.hard_valid),
                str(v.abstain),
                str(v.degraded),
                "; ".join(v.reasons),
            ]
        )
    return buf.getvalue()


def _coverage_block(report: DatasetReport) -> str:
    if report.coverage is None:
        return (
            '<p class="caveat"><b>Reference-mode:</b> no binary validity labels were supplied, '
            "so no conformal coverage is claimed (<code>coverage = None</code>). Abstention here "
            "is a self-supervised heuristic, not a distribution-free guarantee.</p>"
        )
    c = report.coverage
    viol = "violated" if c.nominal_violated else "met"
    cav = (
        f'<p class="caveat">{html.escape(c.exchangeability_caveat)}</p>'
        if c.exchangeability_caveat
        else ""
    )
    return (
        f'<table class="kv"><tr><th>target coverage</th><td>{c.target_coverage:.3f}</td></tr>'
        f"<tr><th>empirical coverage</th><td>{c.empirical_coverage:.3f} ({viol})</td></tr>"
        f"<tr><th>95% CI</th><td>[{c.ci_low:.3f}, {c.ci_high:.3f}]</td></tr>"
        f"<tr><th>holdout n</th><td>{c.n_holdout}</td></tr></table>{cav}"
    )


def to_html(report: DatasetReport) -> str:
    rows = []
    cls = {"accept": "ok", "hold": "warn", "reject": "bad"}
    for v in report.verdicts:
        hv = "—" if v.hard_valid is None else ("✓" if v.hard_valid else "✗")
        rows.append(
            f'<tr class="{cls[v.verdict]}"><td>{html.escape(v.episode_id)}</td>'
            f"<td>{v.verdict}</td><td>{v.Q:.3f}</td><td>{hv}</td>"
            f"<td>{'yes' if v.degraded else ''}</td>"
            f"<td>{html.escape('; '.join(v.reasons))}</td></tr>"
        )
    emb_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{'resolved' if s.resolved else 'degraded'}</td>"
        f"<td>{s.dof}</td><td>{html.escape(s.source)}</td></tr>"
        for k, s in sorted(report.embodiments.items())
    )
    warns = "".join(f"<li>{html.escape(w)}</li>" for w in report.warnings)
    prov = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(val))}</td></tr>"
        for k, val in sorted(report.provenance.items())
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tracecal report — {html.escape(str(report.dataset))}</title>
<style>
 body{{font:14px/1.5 system-ui,sans-serif;margin:2rem;color:#1a1a1a;max-width:1000px}}
 h1{{font-size:1.4rem}} h2{{font-size:1.1rem;margin-top:1.6rem;border-bottom:1px solid #ddd}}
 table{{border-collapse:collapse;width:100%;margin:.5rem 0}}
 th,td{{border:1px solid #ddd;padding:.3rem .5rem;text-align:left;font-size:13px}}
 .kv{{width:auto}} .kv th{{background:#f6f6f6}}
 .summary span{{display:inline-block;padding:.3rem .7rem;margin:.2rem;border-radius:.4rem}}
 .summary span{{font-weight:600}}
 .ok{{background:#e6f4ea}}
 .warn{{background:#fef7e0}}
 .bad{{background:#fce8e6}}
 .caveat{{background:#fff4e5;border-left:4px solid #f0a000;padding:.5rem .8rem;margin:.5rem 0}}
 code{{background:#f0f0f0;padding:0 .2rem;border-radius:3px}}
</style></head><body>
<h1>tracecal validity report</h1>
<p>dataset: <code>{html.escape(str(report.dataset))}</code> · {report.n_episodes} episodes</p>
<div class="summary">
 <span class="ok">accept {report.n_accept}</span>
 <span class="warn">hold {report.n_hold}</span>
 <span class="bad">reject {report.n_reject}</span>
 <span>degraded {report.n_degraded}</span>
</div>
<h2>Conformal coverage</h2>
{_coverage_block(report)}
<h2>Embodiments</h2>
<table><tr><th>robot_type</th><th>status</th><th>dof</th><th>source</th></tr>{emb_rows}</table>
<h2>Episodes</h2>
<table><tr><th>episode</th><th>verdict</th><th>Q</th><th>hard_valid</th>
<th>degraded</th><th>reasons</th></tr>
{"".join(rows)}</table>
<h2>Notes</h2>
<ul>{warns or "<li>none</li>"}</ul>
<h2>Provenance</h2>
<table class="kv">{prov}</table>
</body></html>
"""
