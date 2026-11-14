# Copyright (c) 2026 Pasqal. Licensed under the Pasqal Open-Source Software License
# (MIT-derived); see LICENSE in the repository root.
"""Excite two atoms under the Rydberg blockade with Pulser, through QRMI or on the CPU."""

import argparse
import itertools
import json
import math
import sys

from pulser import Pulse, QPUBackend, Register, Sequence
from pulser.devices import Device, DigitalAnalogDevice

OMEGA = math.pi  # Rabi frequency, rad/us


def blockade_register(device: Device, distance: float) -> Register:
    """Place two atoms distance apart, on calibrated traps when a pre-calibrated layout has such a pair.

On a QPU such as FRESNEL_CAN1 the atoms then sit on traps calibrated in advance. Devices without such a
pair, e.g. the simulated QPU behind Warden, get an automatic layout.

Args:
    device (Device): Device whose rules the register must respect.
    distance (float): Distance between the two atoms in micrometers.

Returns:
    (Register): Register with atoms "q0" and "q1".
"""
    for layout in device.pre_calibrated_layouts:
        coords = layout.coords
        pairs = [(a, b) for a, b in itertools.combinations(range(len(coords)), 2)
                 if math.isclose(math.dist(coords[a], coords[b]), distance, abs_tol=1e-3)]
        if pairs:  # the pair nearest the center of the layout
            center = coords.mean(axis=0)
            a, b = min(pairs, key=lambda p: math.dist(center, (coords[p[0]] + coords[p[1]]) / 2))
            return layout.define_register(a, b, qubit_ids=["q0", "q1"])
    return Register({"q0": (-distance / 2, 0), "q1": (distance / 2, 0)}).with_automatic_layout(device)


def blockade_sequence(device: Device, distance: float) -> Sequence:
    """Build a global pi pulse on two atoms that leaves them in (|gr> + |rg>)/sqrt(2) under the blockade.

Args:
    device (Device): Device whose rules the sequence must respect.
    distance (float): Distance between the two atoms in micrometers.

Returns:
    (Sequence): Measured Pulser sequence on the device.
"""
    sequence = Sequence(blockade_register(device, distance), device)
    sequence.declare_channel("rydberg", "rydberg_global")
    radius = device.rydberg_blockade_radius(OMEGA)
    print(f"Atoms {distance:g} um apart; blockade radius on {device.name}: {radius:.1f} um", file=sys.stderr)
    # Under the blockade the pair oscillates sqrt(2) times faster; durations are in ns, multiples of 4 ns.
    pi_pulse_ns = round(1000 * math.pi / (math.sqrt(2) * OMEGA) / 4) * 4
    sequence.add(Pulse.ConstantPulse(pi_pulse_ns, OMEGA, 0, 0), "rydberg")
    sequence.measure("ground-rydberg")
    return sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulate", action="store_true", help="simulate on the CPU instead of using QRMI")
    parser.add_argument("--runs", type=int, default=500, help="number of shots")
    parser.add_argument("--distance", type=float, default=5.0, help="distance between the atoms in um")
    parser.add_argument("--output", type=argparse.FileType("w"), default=sys.stdout)
    args = parser.parse_args()

    if args.simulate:
        # Imported here: pulser-simulation and QRMI are each needed by one route only.
        from pulser_simulation import QutipBackendV2

        results = QutipBackendV2(blockade_sequence(DigitalAnalogDevice, args.distance)).run()
        counts = dict(results.final_state.sample(num_shots=args.runs))
    else:
        from qrmi.pulser import PulserQRMIConnection

        # The scheduler acquired the QPU for this job; the connection uses that acquisition.
        connection = PulserQRMIConnection()
        # Build for the device the resource reports, so the same program is valid on hardware.
        device = list(connection.fetch_available_devices().values())[0]
        backend = QPUBackend(blockade_sequence(device, args.distance), connection)
        remote = backend.run(job_params=[{"runs": args.runs}], wait=True)
        (result,) = remote.get_available_results().values()
        counts = result.final_bitstrings
    print(json.dumps({"counter": counts}), file=args.output)
