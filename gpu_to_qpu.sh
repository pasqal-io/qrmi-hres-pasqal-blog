#!/bin/bash
# Copyright (c) 2026 Pasqal. Licensed under the Pasqal Open-Source Software License
# (MIT-derived); see LICENSE in the repository root.
# From GPU to QPU: emulate the blockade sweep on a GPU, run the same sweep on the QPU only if the
# emulation succeeded, then compare. Each job holds only what it needs: the GPU job no QPU, the QPU job no GPU.
#   ./gpu_to_qpu.sh PASQAL_LOCAL
# GPU_REQUEST: the GPU request on your cluster, e.g. --gres=gpu:1. EMU_ARGS: arguments for the emulation,
# e.g. --rydberg-c6 <C6 of the target device>.
set -euo pipefail
QPU=${1:?usage: gpu_to_qpu.sh QPU}
emu=$(sbatch --parsable ${GPU_REQUEST-} sweep.sbatch --emulate ${EMU_ARGS-})
qpu=$(sbatch --parsable --dependency="afterok:$emu" --kill-on-invalid-dep=yes --resources="$QPU:1" sweep.sbatch)
cmp=$(sbatch --parsable --dependency="afterok:$qpu" --kill-on-invalid-dep=yes --job-name=compare \
    --output="compare_%j.out" --wrap "python3 compare_sweep.py $emu $qpu")
echo "emulation $emu -> $QPU $qpu -> compare $cmp (output in compare_$cmp.out)"
