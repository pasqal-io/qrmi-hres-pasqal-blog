#!/bin/bash
# Copyright (c) 2026 Pasqal. Licensed under the Pasqal Open-Source Software License
# (MIT-derived); see LICENSE in the repository root.
# Two-atom Rydberg blockade on the QPU resource the job requests with --resources:
#   sbatch --resources=EMU_FREE:1 blockade.sbatch       # EMU_FREE emulator on Pasqal Cloud
#   sbatch --resources=FRESNEL_CAN1:1 blockade.sbatch   # FRESNEL_CAN1 QPU on Pasqal Cloud
#   sbatch --resources=PASQAL_LOCAL:1 blockade.sbatch   # on-premises QPU behind Warden
#   sbatch --resources=EMU_FREE:1 blockade.sbatch blockade_cudaq.py   # the CUDA-Q version
# Arguments after the program name are passed to it, e.g. blockade.sbatch blockade.py --runs 200
#SBATCH --job-name=blockade
#SBATCH --output=blockade_%j.out
#SBATCH --ntasks=1
set -euo pipefail
source /opt/qrmi/venv/bin/activate
echo "QPU resources: ${QRMI_JOB_QPU_RESOURCES:?request a QPU with --resources=<QPU>:1} ($QRMI_JOB_QPU_TYPES)"
program=${1:-blockade.py}
shift $(($# > 0))
srun python "$program" --output "blockade_$SLURM_JOB_ID.json" "$@"
python check_counts.py "blockade_$SLURM_JOB_ID.json"
