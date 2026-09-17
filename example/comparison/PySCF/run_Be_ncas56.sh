#!/usr/bin/env bash
set -euo pipefail

# Run only Be CAS(4e,56o) as a seventh, independent detached job.
# This script has its own PID/status files and never scans, signals, or changes
# the existing six CASSCF sessions.
#
# Target-node CPU layout:
#   NUMA 0: 0-31,64-95; physical-core SMT siblings differ by 64.
# Current six jobs use 0-23 and 32-55.  CPUs 24-31 are therefore unused first
# hardware threads on NUMA 0; CPUs 88-95 are their SMT siblings and are avoided.
#
# Usage:
#   ./run_Be_ncas56_independent.sh start
#   ./run_Be_ncas56_independent.sh status
#   ./run_Be_ncas56_independent.sh monitor
#   ./run_Be_ncas56_independent.sh logs
#   ./run_Be_ncas56_independent.sh stop

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUN_SCRIPT=${RUN_SCRIPT:-${SCRIPT_DIR}/run_Be_CASSCF.sh}
PY_SCRIPT=${PY_SCRIPT:-${SCRIPT_DIR}/Be_CASSCF_timed.py}
PYTHON=${PYTHON:-python}
CPU_LIST=${CPU_LIST:-24-31}
THREADS=${THREADS:-8}
MEMORY_MB=${MEMORY_MB:-32000}
MAX_MACRO=${MAX_MACRO:-100}
NICE_LEVEL=${NICE_LEVEL:-5}

STATE_DIR=${STATE_DIR:-${SCRIPT_DIR}/.be_ncas56_state}
RESULTS_ROOT=${RESULTS_ROOT:-${SCRIPT_DIR}/Results}
PID_FILE=${STATE_DIR}/pid
RUN_ID_FILE=${STATE_DIR}/run_id
LOG_FILE_RECORD=${STATE_DIR}/log_file
STATUS_FILE=${STATE_DIR}/exitcode

mkdir -p "${STATE_DIR}" "${RESULTS_ROOT}"

is_alive() {
    local pid="$1"
    kill -0 "${pid}" 2>/dev/null
}

is_our_session() {
    local pid="$1"
    local sid
    sid="$(ps -o sid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
    [[ -n "${sid}" && "${sid}" == "${pid}" ]]
}

read_recorded_pid() {
    local pid=""
    [[ -f "${PID_FILE}" ]] && read -r pid < "${PID_FILE}"
    printf '%s' "${pid}"
}

preflight() {
    command -v taskset >/dev/null || { echo "ERROR: taskset not found" >&2; exit 2; }
    command -v setsid  >/dev/null || { echo "ERROR: setsid not found" >&2; exit 2; }
    command -v nice    >/dev/null || { echo "ERROR: nice not found" >&2; exit 2; }
    command -v "${PYTHON}" >/dev/null || { echo "ERROR: ${PYTHON} not found in PATH" >&2; exit 2; }
    [[ -f "${RUN_SCRIPT}" ]] || { echo "ERROR: missing ${RUN_SCRIPT}" >&2; exit 2; }
    [[ -f "${PY_SCRIPT}" ]]  || { echo "ERROR: missing ${PY_SCRIPT}" >&2; exit 2; }
    "${PYTHON}" -c 'import pyscf, basis_set_exchange' >/dev/null 2>&1 \
        || { echo "ERROR: current python cannot import pyscf and basis_set_exchange" >&2; exit 2; }
    taskset -c "${CPU_LIST}" true >/dev/null 2>&1 \
        || { echo "ERROR: CPU range ${CPU_LIST} is not allowed on this node" >&2; exit 2; }

    local available_mb
    available_mb=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
    if (( available_mb < 65536 )); then
        echo "ERROR: only ${available_mb} MB memory is available; refusing to add a seventh job." >&2
        exit 2
    fi
    echo "Preflight OK: CPU=${CPU_LIST}, threads=${THREADS}, nice=${NICE_LEVEL}, memory cap=${MEMORY_MB} MB, MemAvailable=${available_mb} MB"
}

start_job() {
    preflight

    local old_pid
    old_pid="$(read_recorded_pid)"
    if [[ -n "${old_pid}" ]] && is_alive "${old_pid}" && is_our_session "${old_pid}"; then
        echo "ERROR: the independent Be ncas=56 job is already running: PID/SID=${old_pid}" >&2
        exit 1
    fi

    local run_id result_dir logfile launcher_log prefix
    run_id="Be_ncas56_$(date '+%Y%m%d_%H%M%S')"
    result_dir="${RESULTS_ROOT}/${run_id}"
    logfile="${result_dir}/Be_ncas56.log"
    launcher_log="${result_dir}/Be_ncas56.launcher.log"
    prefix="${result_dir}/Be_CASSCF_ncas56"

    mkdir -p "${result_dir}"
    printf '%s\n' "${run_id}" > "${RUN_ID_FILE}"
    printf '%s\n' "${logfile}" > "${LOG_FILE_RECORD}"
    rm -f "${STATUS_FILE}"

    nohup setsid \
        taskset -c "${CPU_LIST}" \
        nice -n "${NICE_LEVEL}" \
        env \
            PYTHON="${PYTHON}" \
            SCRIPT="${PY_SCRIPT}" \
            NCAS_VALUES=56 \
            THREADS="${THREADS}" \
            MEMORY_MB="${MEMORY_MB}" \
            MAX_MACRO="${MAX_MACRO}" \
            RUN_ID="${run_id}" \
            RESULT_DIR="${result_dir}" \
            LOG="${logfile}" \
            PREFIX="${prefix}" \
            OMP_PROC_BIND=close \
            OMP_PLACES=cores \
        bash -c '
            set -euo pipefail
            run_script=$1
            status_file=$2

            finalize_job() {
                rc=$?
                trap - EXIT
                tmp_status="${status_file}.tmp.$$"
                printf "%s\n" "${rc}" > "${tmp_status}"
                mv -f "${tmp_status}" "${status_file}"
                exit "${rc}"
            }

            trap finalize_job EXIT
            bash "${run_script}"
        ' _ "${RUN_SCRIPT}" "${STATUS_FILE}" \
        > "${launcher_log}" 2>&1 < /dev/null &

    local pid=$!
    printf '%s\n' "${pid}" > "${PID_FILE}"
    sleep 0.5

    if ! is_alive "${pid}"; then
        echo "ERROR: Be ncas=56 failed to start. Launcher output:" >&2
        tail -n 30 "${launcher_log}" >&2 || true
        exit 1
    fi
    if ! is_our_session "${pid}"; then
        echo "ERROR: Be ncas=56 did not start as the expected detached session." >&2
        exit 1
    fi

    echo "Started independent Be CAS(4e,56o)"
    echo "  PID/SID : ${pid}"
    echo "  CPUs    : ${CPU_LIST}"
    echo "  threads : ${THREADS}"
    echo "  nice    : ${NICE_LEVEL} (existing jobs retain priority)"
    echo "  memory  : ${MEMORY_MB} MB cap"
    echo "  maxmacro: ${MAX_MACRO}"
    echo "  result  : ${result_dir}"
    echo "  log     : ${logfile}"
    echo
    echo "The existing six CASSCF jobs were not modified."
}

job_state() {
    local pid="$1"
    local rc
    if [[ -n "${pid}" ]] && is_alive "${pid}" && is_our_session "${pid}"; then
        printf 'RUNNING'
    elif [[ -n "${pid}" ]] && is_alive "${pid}"; then
        printf 'STALE_PID'
    elif [[ -f "${STATUS_FILE}" ]]; then
        read -r rc < "${STATUS_FILE}" || rc=unknown
        if [[ "${rc}" == "0" ]]; then
            printf 'COMPLETED'
        else
            printf 'FAILED(rc=%s)' "${rc}"
        fi
    else
        printf 'NOT_STARTED/UNKNOWN'
    fi
}

show_status() {
    local pid run_id logfile state progress
    pid="$(read_recorded_pid)"
    run_id="unknown"
    logfile=""
    [[ -f "${RUN_ID_FILE}" ]] && read -r run_id < "${RUN_ID_FILE}"
    [[ -f "${LOG_FILE_RECORD}" ]] && read -r logfile < "${LOG_FILE_RECORD}"
    state="$(job_state "${pid}")"

    echo "Run ID : ${run_id}"
    echo "Job    : Be ncas=56"
    echo "PID/SID: ${pid:-none}"
    echo "CPUs   : ${CPU_LIST}"
    echo "State  : ${state}"
    echo "Log    : ${logfile:-none}"

    if [[ -n "${logfile}" && -f "${logfile}" ]]; then
        progress="$(grep -E '\[MACRO_TIMING\]|CASSCF energy|Saved summary JSON' "${logfile}" | tail -n 1 || true)"
        [[ -n "${progress}" ]] && echo "Progress: ${progress}"
    fi

    if [[ "${state}" == "RUNNING" ]]; then
        echo
        ps -e -o pid,ppid,sid,psr,%cpu,%mem,etime,rss,comm,args \
            | awk -v sid="${pid}" 'NR == 1 || $3 == sid'
    fi
}

show_logs() {
    local logfile
    logfile=""
    [[ -f "${LOG_FILE_RECORD}" ]] && read -r logfile < "${LOG_FILE_RECORD}"
    if [[ -z "${logfile}" ]]; then
        echo "No recorded log file."
        exit 0
    fi
    if [[ -f "${logfile}" ]]; then
        tail -n 40 "${logfile}"
    elif [[ -f "${logfile%.log}.launcher.log" ]]; then
        tail -n 40 "${logfile%.log}.launcher.log"
    else
        echo "Log has not been created: ${logfile}"
    fi
}

stop_job() {
    local pid
    pid="$(read_recorded_pid)"
    if [[ -z "${pid}" ]]; then
        echo "No recorded independent Be ncas=56 job."
        exit 0
    fi

    if is_alive "${pid}" && is_our_session "${pid}"; then
        echo "Sending SIGTERM only to Be ncas=56 PID/SID=${pid}"
        kill -TERM -- "-${pid}"
        sleep 2
    elif is_alive "${pid}"; then
        echo "SKIP: PID ${pid} exists but SID validation failed; no signal sent." >&2
        exit 1
    else
        echo "Be ncas=56 is already stopped."
    fi

    show_status
}

monitor_job() {
    echo "Press Ctrl+C to leave the monitor; the job will keep running."
    sleep 1
    while true; do
        clear || true
        echo "Independent Be ncas=56 monitor -- $(date '+%F %T')"
        echo
        show_status
        echo
        echo "Refresh: 10 s"
        sleep 10
    done
}

case "${1:-status}" in
    start)   start_job ;;
    status)  show_status ;;
    monitor) monitor_job ;;
    logs)    show_logs ;;
    stop)    stop_job ;;
    *)
        echo "Usage: $0 {start|status|monitor|logs|stop}" >&2
        exit 2
        ;;
esac
