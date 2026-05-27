#!/usr/bin/env bash
# Honest-marketing + measurement-provenance gate (S7).
# ERE only (no BRE "\|" dead-grep); exit codes verified on a real run.
# Fails (exit 1) on overclaim phrases, missing disclaimers, missing artifacts, or README
# numbers that are NOT traceable to the committed measurement JSON.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README="$ROOT/README.md"
GATE="$ROOT/results/gate_v0.1.0a1.json"
CAL="$ROOT/results/calibration_v0.1.0a1.json"
fail=0

# 1) No overclaim phrases in README (ERE alternation, case-insensitive).
BANNED='world.?s first|first[ -]ever|the only (tool|harness|framework|library|method|way)|完全に自動|完全自動|永続的|fully automatic|guarantees? (correctness|validity|safety)|state-of-the-art|beats all|production[- ]ready|battle[- ]tested'
if grep -niE "$BANNED" "$README"; then
  echo "FAIL: overclaim phrase found in README.md"; fail=1
fi

# 2) Degrade honesty guard: tracecal must NOT claim to validate the physics of arms whose
#    URDF it cannot resolve (SO-101/Koch/LeKiwi). Forbid that claim so it cannot creep back.
if grep -niE '(validat|certif|guarantee)[a-z]* (the )?(physics|kinematics|validity) of (so-?10[01]|koch|lekiwi)' "$README"; then
  echo "FAIL: README claims physics validation for a no-URDF (degrade-only) embodiment"; fail=1
fi

# 3) Measurement artifacts must exist and be non-empty.
for f in "$GATE" "$CAL"; do
  if [ ! -s "$f" ]; then echo "FAIL: missing measurement artifact $f"; fail=1; fi
done

# 4) Synthetic calibration MUST carry a disclaimer in the JSON and surface it in the README.
if [ -s "$CAL" ] && ! grep -qi 'disclaimer' "$CAL"; then
  echo "FAIL: calibration JSON lacks a disclaimer"; fail=1
fi
if ! grep -qi 'algorithm validation only' "$README"; then
  echo "FAIL: README must surface the synthetic 'algorithm validation only' disclaimer"; fail=1
fi

# 5) Real-data gate must cite provenance; README must state the coverage label precondition.
if [ -s "$GATE" ] && ! grep -qi 'data_provenance' "$GATE"; then
  echo "FAIL: gate JSON lacks data_provenance"; fail=1
fi
if ! grep -qiE 'reference[- ]mode|requires .*(binary )?validity labels|only a (validated )?guarantee when' "$README"; then
  echo "FAIL: README must state the conformal coverage label precondition (reference-mode default)"; fail=1
fi

# 6) Provenance: every headline number in the README must exist in the measurement JSON
#    at the README's displayed precision (anti-theater — no hand-written numbers).
if [ -s "$GATE" ] && [ -s "$CAL" ]; then
  if ! python3 - "$README" "$GATE" "$CAL" <<'PYC'
import json, sys
readme = open(sys.argv[1], encoding="utf-8").read()
gate = json.load(open(sys.argv[2], encoding="utf-8"))
cal = json.load(open(sys.argv[3], encoding="utf-8"))
need = []
need.append(("%.2f" % gate["gate_demonstration"]["physics_caught_rate"], "physics caught rate"))
need.append(("%.2f" % cal["target_coverage"], "target coverage"))
need.append(("%.2f" % cal["empirical_coverage"], "empirical coverage"))
need.append(("%.3f" % cal["ece"], "ECE"))
missing = [(v, d) for v, d in need if v not in readme]
for v, d in missing:
    print(f"FAIL: README is missing measured value {v} ({d}) -> numbers must come from JSON")
sys.exit(1 if missing else 0)
PYC
  then
    echo "FAIL: README numbers not traceable to measurement JSON"; fail=1
  fi
fi

if [ "$fail" -ne 0 ]; then echo "honest-marketing gate FAILED"; exit 1; fi
echo "honest-marketing gate OK"
