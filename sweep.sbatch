#!/bin/bash
# Copyright (c) 2026 Pasqal. Licensed under the Pasqal Open-Source Software License
# (MIT-derived); see LICENSE in the repository root.
# The CUDA-Q blockade program at several atom distances, emulated on a GPU or on the QPU the job requests:
#   sbatch sweep.sbatch --emulate                    # GPU emulation, no QPU
#   sbatch --resources=PASQAL_LOCAL:1 sweep.sbatch    # the QPU resource
# DISTANCES (um) sets the sweep; arguments are passed to blockade_cudaq.py.
#SBATCH --job-name=sweep
#SBATCH --output=sweep_%j.out
#SBATCH --ntasks=1
set -euo pipefail
source /opt/qrmi/venv/bin/activate
echo "QPU resources: ${QRMI_JOB_QPU_RESOURCES:-none, emulation}"
for d in ${DISTANCES:-4 6 8 10 12 14}; do
    srun python blockade_cudaq.py --distance "$d" --output "sweep_${SLURM_JOB_ID}_$d.json" "$@"
done
