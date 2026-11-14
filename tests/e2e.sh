#!/bin/bash
# Copyright (c) 2026 Pasqal. Licensed under the Pasqal Open-Source Software License
# (MIT-derived); see LICENSE in the repository root.
# End-to-end test of one QPU resource on the container cluster (compose project "hres"), run on the host:
#   tests/e2e.sh EMU_FREE       # Pasqal Cloud
#   tests/e2e.sh PASQAL_LOCAL   # Warden; also checks the Warden sessions and an unreachable QPU
#   tests/e2e.sh EMU_FREE blockade_cudaq.py   # the CUDA-Q version (needs the CUDA-Q image)
# Checks: the job runs the program and passes check_counts.py; a second job starts only after the first
# ended; the controller holds a token while a job runs and releases it after the job and after scancel.
set -euo pipefail
QPU=${1:?usage: tests/e2e.sh QPU [PROGRAM]}
PROGRAM=${2:-blockade.py}
PREFIX=${PREFIX:-hres}
STATE=/var/spool/slurmctld/hres_qrmi
START=$(date -u +%Y-%m-%dT%H:%M:%S)  # Warden sessions that end after this belong to this run
RELEASE_WAIT=${RELEASE_WAIT:-40}  # seconds; releases happen in maintenance (maintenance_cycle 10)

as_user() { docker exec -u user -w /home/user/example "$PREFIX-login" bash -c "$1"; }
ctld() { docker exec "$PREFIX-slurmctld" bash -c "$1"; }
fail() { echo "FAIL: $*"; exit 1; }
ok() { echo "ok: $*"; }
job_field() { ctld "scontrol -o show job $1" | tr ' ' '\n' | sed -n "s/^$2=//p"; }
# A failed run must not leave a job holding the QPU or Warden stopped for the next run
trap 'docker start "$PREFIX-warden" >/dev/null; as_user "scancel -u user"' EXIT

wait_state() {  # job, state regex, timeout in s; fails at once on another final state
    local state
    for _ in $(seq "$3"); do
        state=$(job_field "$1" JobState)
        [[ $state =~ ^($2)$ ]] && return 0
        [[ $state =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)$ ]] && break
        sleep 1
    done
    fail "job $1 is $state, expected $2"
}

held_tokens() { ctld "ls $STATE/tokens $STATE/release | grep -c '^$QPU\.$1\.' || true"; }

wait_released() {  # job
    for _ in $(seq "$RELEASE_WAIT"); do
        [ "$(held_tokens "$1")" = 0 ] && break
        sleep 1
    done
    [ "$(held_tokens "$1")" = 0 ] || fail "job $1: token not released after ${RELEASE_WAIT}s"
    if [ "$QPU" = PASQAL_LOCAL ]; then
        # Warden accounting lists ended sessions only; admin request as root (uid 0)
        local ended
        ended=$(ctld "curl -fsS -H \"X-Munge-Cred: \$(munge -n)\" 'http://warden:4207/accounting/sessions?limit=100&ended_after=$START'" |
            python3 -c "import json, sys; print(sum(s['slurm_job_id'] == '$1' and s['revoked_at'] is not None for s in json.load(sys.stdin)['data']))")
        [ "$ended" = 1 ] || fail "job $1: $ended ended Warden sessions"
    fi
    ok "job $1: token released"
}

docker exec -u user -w /home/user "$PREFIX-login" bash -c \
    "rm -rf example && mkdir example && cp /opt/example/*.py /opt/example/*.sbatch example/"

# 1. Run the example
job=$(as_user "sbatch --parsable --resources=$QPU:1 blockade.sbatch $PROGRAM")
wait_state "$job" COMPLETED 900
as_user "cat blockade_$job.out"
as_user "grep -q '^QPU resources: $QPU ' blockade_$job.out && grep -q '^blockade OK' blockade_$job.out" \
    || fail "job $job: wrong resource or blockade check failed"
ok "job $job: $PROGRAM blockade on $QPU"
wait_released "$job"

# 2. count 1: a second job starts only after the first ended
first=$(as_user "sbatch --parsable --resources=$QPU:1 --wrap 'sleep 20'")
wait_state "$first" RUNNING 120
second=$(as_user "sbatch --parsable --resources=$QPU:1 --wrap true")
wait_state "$first" COMPLETED 120
wait_state "$second" COMPLETED 180
[[ ! $(job_field "$second" StartTime) < $(job_field "$first" EndTime) ]] ||
    fail "job $second started before job $first ended"
ok "job $second started after job $first ended"
wait_released "$first"
wait_released "$second"

# 3. A running job holds a token; scancel releases it
job=$(as_user "sbatch --parsable --resources=$QPU:1 --wrap 'sleep 300'")
wait_state "$job" RUNNING 120
[ "$(held_tokens "$job")" = 1 ] || fail "job $job: no token while running"
ok "job $job: token held while running"
as_user "scancel $job"
wait_state "$job" CANCELLED 60
wait_released "$job"

# 4. Pasqal Local: a stopped Warden disables the resource and jobs wait (HRESDisabled)
if [ "$QPU" = PASQAL_LOCAL ]; then
    docker stop "$PREFIX-warden" >/dev/null
    for _ in $(seq 60); do
        ctld "scontrol show license $QPU" | grep -q 'DisableLayer=true' && break
        sleep 1
    done
    ctld "scontrol show license $QPU" | grep -q 'DisableLayer=true' || fail "$QPU not disabled with Warden stopped"
    job=$(as_user "sbatch --parsable --resources=$QPU:1 --wrap true")
    for _ in $(seq 30); do
        [ "$(job_field "$job" Reason)" = HRESDisabled ] && break
        sleep 1
    done
    [ "$(job_field "$job" Reason)" = HRESDisabled ] || fail "job $job: $(job_field "$job" JobState) $(job_field "$job" Reason)"
    ok "job $job pending while Warden is stopped: HRESDisabled"
    docker start "$PREFIX-warden" >/dev/null
    wait_state "$job" COMPLETED 120
    wait_released "$job"
fi
echo "PASS: $QPU"
