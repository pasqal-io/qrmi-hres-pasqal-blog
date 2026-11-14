# Copyright (c) 2026 Pasqal. Licensed under the Pasqal Open-Source Software License
# (MIT-derived); see LICENSE in the repository root.
"""Check that measured counts show the two-atom Rydberg blockade.

Ideal outcome: '01' and '10' with probability 1/2 each, '11' never. The bounds leave room for a QPU's
readout errors: with FRESNEL_CAN1's specified 8% false negatives, 1.2% false positives and 1.8%
preparation errors, P(01)+P(10) is about 0.88; outside the blockade it is about 0.3. With N shots the
'01'/'10' imbalance has a standard deviation of at most 1/sqrt(N); 4/sqrt(N) is allowed.
"""

import json
import math
import sys

MAX_BOTH_EXCITED = 0.05
MIN_ONE_EXCITED = 0.75


def check(counts: dict[str, int]) -> list[str]:
    """Return the blockade conditions that the counts violate.

Args:
    counts (dict[str, int]): Shots per bitstring, first character for atom 0, '1' = Rydberg state.

Returns:
    (list[str]): Violated conditions, empty if the counts pass.
"""
    shots = sum(counts.values())
    p = {k: counts.get(k, 0) / shots for k in ("00", "01", "10", "11")}
    print(f"{shots} shots: " + ", ".join(f"P({k})={v:.3f}" for k, v in p.items()))
    failures = []
    if p["11"] > MAX_BOTH_EXCITED:
        failures.append(f"P(11)={p['11']:.3f} > {MAX_BOTH_EXCITED}")
    if p["01"] + p["10"] < MIN_ONE_EXCITED:
        failures.append(f"P(01)+P(10)={p['01'] + p['10']:.3f} < {MIN_ONE_EXCITED}")
    if abs(p["01"] - p["10"]) > 4 / math.sqrt(shots):
        failures.append(f"|P(01)-P(10)|={abs(p['01'] - p['10']):.3f} > 4/sqrt({shots})")
    return failures


if __name__ == "__main__":
    failures = check(json.load(open(sys.argv[1]))["counter"])
    print("\n".join(failures) or "blockade OK")
    sys.exit(1 if failures else 0)
