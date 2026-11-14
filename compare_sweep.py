# Copyright (c) 2026 Pasqal. Licensed under the Pasqal Open-Source Software License
# (MIT-derived); see LICENSE in the repository root.
"""Compare the blockade sweep of an emulation job with the same sweep from a QPU job.

Without the blockade each atom is excited with probability sin^2(pi / (2 sqrt 2)) = 0.80, so P(11) rises
from 0 to about 0.64 as the atoms move apart. The QPU sweep should follow the emulated one within
TOLERANCE at every distance.
"""

import glob
import json
import re
import sys

TOLERANCE = 0.1


def sweep(job: str) -> dict[float, dict[str, float]]:
    """Read the probabilities of a sweep job.

Args:
    job (str): Slurm job ID of a sweep.sbatch job.

Returns:
    (dict[float, dict[str, float]]): Probability per bitstring, per atom distance in um.
"""
    result = {}
    for path in glob.glob(f"sweep_{job}_*.json"):
        counts = json.load(open(path))["counter"]
        shots = sum(counts.values())
        distance = float(re.search(rf"sweep_{job}_(.+)\.json$", path)[1])
        result[distance] = {k: counts.get(k, 0) / shots for k in ("00", "01", "10", "11")}
    return result


if __name__ == "__main__":
    emulated, measured = sweep(sys.argv[1]), sweep(sys.argv[2])
    print("distance (um)   P(11) emulated   P(11) QPU   P(01)+P(10) emulated   P(01)+P(10) QPU")
    worst = 0.0
    for d in sorted(emulated.keys() & measured.keys()):
        e, m = emulated[d], measured[d]
        print(f"{d:13g}   {e['11']:14.3f}   {m['11']:9.3f}   {e['01'] + e['10']:20.3f}   {m['01'] + m['10']:15.3f}")
        worst = max(worst, abs(e["11"] - m["11"]), abs(e["01"] + e["10"] - m["01"] - m["10"]))
    ok = len(emulated) > 0 and emulated.keys() == measured.keys() and worst <= TOLERANCE
    print(f"largest difference {worst:.3f}: " + ("QPU matches the emulation" if ok else f"MISMATCH (tolerance {TOLERANCE})"))
    sys.exit(0 if ok else 1)
