# Copyright (c) 2026 Pasqal. Licensed under the Pasqal Open-Source Software License
# (MIT-derived); see LICENSE in the repository root.
"""Excite two atoms under the Rydberg blockade with CUDA-Q, through QRMI or a local GPU emulation."""

import argparse
import json
import math
import sys

import cudaq
from cudaq.dynamics import Schedule
from cudaq.operators import RydbergHamiltonian, ScalarOperator

OMEGA = math.pi * 1e6  # Rabi frequency, rad/s (pi rad/us, as in blockade.py)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emulate", action="store_true", help="emulate on the local GPU instead of using QRMI")
    parser.add_argument("--runs", type=int, default=500, help="number of shots")
    parser.add_argument("--distance", type=float, default=5.0, help="distance between the atoms in um")
    parser.add_argument("--rydberg-c6", help="with --emulate: C6/hbar of the device to emulate in rad m^6/s (default: FRESNEL)")
    parser.add_argument("--output", type=argparse.FileType("w"), default=sys.stdout)
    args = parser.parse_args()

    # QRMI: the resource the scheduler acquired for this job (QRMI_JOB_QPU_RESOURCES)
    emulation = {"emulate": True} | ({"emulation_rydberg_c6": args.rydberg_c6} if args.rydberg_c6 else {})
    cudaq.set_target("pasqal", **(emulation if args.emulate else {"machine": "qrmi"}))
    # The pi pulse of blockade.py: sqrt(2) faster under the blockade, rounded to 4 ns
    duration = round(1e9 * math.pi / (math.sqrt(2) * OMEGA) / 4) * 4e-9
    result = cudaq.evolve(
        RydbergHamiltonian(
            atom_sites=[(0.0, 0.0), (args.distance * 1e-6, 0.0)],  # meters
            amplitude=ScalarOperator.const(OMEGA),
            phase=ScalarOperator.const(0.0),
            delta_global=ScalarOperator.const(0.0),
        ),
        schedule=Schedule([0.0, duration], ["t"]),
        shots_count=args.runs,
    )
    print(json.dumps({"counter": dict(result.items())}), file=args.output)
