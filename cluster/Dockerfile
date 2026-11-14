# Slurm image with the Python environment for the jobs on the login and compute nodes
#
# /opt/qrmi/venv: QRMI and Pulser 1.9.0. QRMI from PyPI loads libmunge.so.2 at runtime.
# Build the slurm-hres-plugins base image (containers/compose.yml) first.
ARG SLURM_IMAGE=slurm-docker-cluster:26.11.0.0rc1
FROM ${SLURM_IMAGE}
ARG QRMI_VERSION=0.27.0
RUN python3.12 -m venv /opt/qrmi/venv \
 && /opt/qrmi/venv/bin/pip install --no-cache-dir "qrmi==${QRMI_VERSION}" \
    pulser-core==1.9.0 pulser-simulation==1.9.0 \
 && /opt/qrmi/venv/bin/pip check
