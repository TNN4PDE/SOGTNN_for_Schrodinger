#!/usr/bin/env bash
set -euo pipefail

# Independent detached launcher for the two He/aug-cc-pV7Z calculations.
# It deliberately uses its own state files and result directory, so it neither
# modifies nor controls the existing four-system CASSCF workflow.
#
# CPU layout on the target node:
#   NUMA 0: 0-31,64-95   (SMT siblings differ by 64)
#   NUMA 1: 32-63,96-127
# Existing four jobs occupy physical cores 0-15 and 32-47.  The defaults below
# therefore use unused first hardware threads on separate sockets.
#
# Usage:
#   ./run_He_aug7z_parallel.sh start
#   ./run_He_aug7z_parallel.sh status
#   ./run_He_aug7z_parallel.sh monitor
#   ./run_He_aug7z_parallel.sh logs [all|singlet|triplet]
#   ./run_He_aug7z_parallel.sh stop [all|singlet|triplet]

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

THREADS=${THREADS:-8}
MEMORY_MB=${MEMORY_MB:-64000}
SINGLET_CORES=${SINGLET_CORES:-16-23}
TRIPLET_CORES=${TRIPLET_CORES:-48-55}
BASIS_FILE=${BASIS_FILE:-${SCRIPT_DIR}/../../../data/He_aug-cc-pV7Z.nwchem}

STATE_DIR=${STATE_DIR:-${SCRIPT_DIR}/.casscf_aug7z_state}
RESULTS_ROOT=${RESULTS_ROOT:-${SCRIPT_DIR}/Results}
PID_FILE=${STATE_DIR}/jobs.tsv
RUN_ID_FILE=${STATE_DIR}/run_id
RUN_DIR_FILE=${STATE_DIR}/run_dir

mkdir -p "${STATE_DIR}" "${RESULTS_ROOT}"

timestamp() {
    date '+%Y%m%d_%H%M%S'
}

is_alive() {
    local pid="$1"
    kill -0 "${pid}" 2>/dev/null
}

# Every job is launched by setsid.  Requiring SID=PID prevents a stale reused
# PID from causing status/stop to touch an unrelated process.
is_our_session() {
    local pid="$1"
    local sid
    sid="$(ps -o sid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
    [[ -n "${sid}" && "${sid}" == "${pid}" ]]
}

matches_target() {
    local name="$1"
    local target="$2"
    case "${target}" in
        all)     return 0 ;;
        singlet) [[ "${name}" == "He_singlet_7Z" ]] ;;
        triplet) [[ "${name}" == "He_triplet_7Z" ]] ;;
        *)       return 1 ;;
    esac
}

check_target() {
    case "$1" in
        all|singlet|triplet) ;;
        *)
            echo "ERROR: target must be all, singlet, or triplet" >&2
            exit 2
            ;;
    esac
}

preflight() {
    command -v taskset >/dev/null || { echo "ERROR: taskset not found" >&2; exit 2; }
    command -v setsid  >/dev/null || { echo "ERROR: setsid not found" >&2; exit 2; }
    command -v python  >/dev/null || { echo "ERROR: python not found in PATH" >&2; exit 2; }

    local required=(
        "${SCRIPT_DIR}/He_singlet_CASSCF_aug7z.py"
        "${SCRIPT_DIR}/He_triplet_CASSCF_aug7z.py"
        "${SCRIPT_DIR}/run_He_singlet_CASSCF_aug7z.sh"
        "${SCRIPT_DIR}/run_He_triplet_CASSCF_aug7z.sh"
        "${BASIS_FILE}"
    )
    local path
    for path in "${required[@]}"; do
        [[ -f "${path}" ]] || { echo "ERROR: required file not found: ${path}" >&2; exit 2; }
    done
}

start_job() {
    local name="$1"
    local cores="$2"
    local run_script="$3"
    local logfile="$4"
    local prefix="$5"
    local run_dir="$6"
    local status_file="${run_dir}/.${name}.exitcode"
    local launcher_log="${logfile}.launcher"

    nohup setsid \
        taskset -c "${cores}" \
        env \
            THREADS="${THREADS}" \
            MEMORY_MB="${MEMORY_MB}" \
            BASIS_FILE="${BASIS_FILE}" \
            LOG="${logfile}" \
            PREFIX="${prefix}" \
            OMP_PROC_BIND=close \
            OMP_PLACES=cores \
        bash -c '
            set -euo pipefail
            run_script=$1
            run_dir=$2
            status_file=$3

            finalize_job() {
                rc=$?
                trap - EXIT
                tmp_status="${status_file}.tmp.$$"
                printf "%s\n" "${rc}" > "${tmp_status}"
                mv -f "${tmp_status}" "${status_file}"
                exit "${rc}"
            }

            trap finalize_job EXIT
            cd "${run_dir}"
            bash "${run_script}"
        ' _ "${run_script}" "${run_dir}" "${status_file}" \
        > "${launcher_log}" 2>&1 < /dev/null &

    local pid=$!
    sleep 0.5

    if ! is_alive "${pid}"; then
        echo "ERROR: ${name} failed to start. Launcher output:" >&2
        tail -n 30 "${launcher_log}" >&2 || true
        return 1
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${pid}" "${cores}" "${run_script}" "${logfile}" "${status_file}" \
        >> "${PID_FILE}"

    echo "Started ${name}: PID/SID=${pid}, CPUs=${cores}"
    echo "  log: ${logfile}"
}

start_all() {
    preflight

    if [[ -s "${PID_FILE}" ]]; then
        local active=0
        while IFS=$'\t' read -r name pid cores run_script logfile status_file; do
            [[ -z "${pid:-}" ]] && continue
            if is_alive "${pid}" && is_our_session "${pid}"; then
                echo "ERROR: recorded 7Z job still running: ${name} PID=${pid} CPUs=${cores}" >&2
                active=1
            fi
        done < "${PID_FILE}"
        if (( active )); then
            echo "Use '$0 status' or '$0 stop all' before starting another 7Z run." >&2
            exit 1
        fi
    fi

    local run_id run_dir
    run_id="$(timestamp)"
    run_dir="${RESULTS_ROOT}/He_aug7z_${run_id}"
    mkdir -p "${run_dir}"

    printf '%s\n' "${run_id}" > "${RUN_ID_FILE}"
    printf '%s\n' "${run_dir}" > "${RUN_DIR_FILE}"
    : > "${PID_FILE}"

    echo "Run ID: ${run_id}"
    echo "Results: ${run_dir}"
    echo "THREADS/job=${THREADS}, MEMORY_MB/job=${MEMORY_MB}"
    echo "CPU plan: singlet=${SINGLET_CORES} (NUMA0), triplet=${TRIPLET_CORES} (NUMA1)"
    echo

    start_job \
        "He_singlet_7Z" "${SINGLET_CORES}" \
        "${SCRIPT_DIR}/run_He_singlet_CASSCF_aug7z.sh" \
        "${run_dir}/He_singlet_aug7z_${run_id}.log" \
        "${run_dir}/He_singlet_CASSCF_aug7z" \
        "${run_dir}"

    start_job \
        "He_triplet_7Z" "${TRIPLET_CORES}" \
        "${SCRIPT_DIR}/run_He_triplet_CASSCF_aug7z.sh" \
        "${run_dir}/He_triplet_aug7z_${run_id}.log" \
        "${run_dir}/He_triplet_CASSCF_aug7z" \
        "${run_dir}"

    echo
    echo "Both 7Z jobs are detached and will survive SSH logout."
    echo "Existing 6Z He/Li/Be sessions were not modified."
    echo "Check with: $0 status"
}

job_state() {
    local pid="$1"
    local status_file="$2"
    local rc

    if is_alive "${pid}" && is_our_session "${pid}"; then
        printf 'RUNNING'
    elif is_alive "${pid}"; then
        printf 'STALE_PID'
    elif [[ -f "${status_file}" ]]; then
        read -r rc < "${status_file}" || rc=unknown
        if [[ "${rc}" == "0" ]]; then
            printf 'COMPLETED'
        else
            printf 'FAILED(rc=%s)' "${rc}"
        fi
    else
        printf 'FINISHED?'
    fi
}

show_progress() {
    local logfile="$1"
    local line
    if [[ -f "${logfile}" ]]; then
        line="$(grep -E '\[MACRO_TIMING\]|CASSCF energy|Saved summary JSON' "${logfile}" | tail -n 1 || true)"
        if [[ -n "${line}" ]]; then
            echo "  progress: ${line}"
        else
            line="$(grep -E 'Spherical NAO|Smallest eig\(S\)|converged SCF energy' "${logfile}" | tail -n 1 || true)"
            [[ -n "${line}" ]] && echo "  progress: ${line}"
        fi
    fi
}

status_all() {
    local run_id="unknown"
    [[ -f "${RUN_ID_FILE}" ]] && read -r run_id < "${RUN_ID_FILE}"
    echo "Run ID: ${run_id}"

    if [[ ! -s "${PID_FILE}" ]]; then
        echo "No recorded He/aug-cc-pV7Z jobs."
        exit 0
    fi

    printf '%-15s %-10s %-9s %-18s %s\n' "JOB" "PID" "CPU" "STATE" "LOG"
    printf '%-15s %-10s %-9s %-18s %s\n' "---------------" "----------" "---------" "------------------" "---"

    while IFS=$'\t' read -r name pid cores run_script logfile status_file; do
        [[ -z "${pid:-}" ]] && continue
        local state
        state="$(job_state "${pid}" "${status_file}")"
        printf '%-15s %-10s %-9s %-18s %s\n' \
            "${name}" "${pid}" "${cores}" "${state}" "${logfile}"
        show_progress "${logfile}"
    done < "${PID_FILE}"

    echo
    echo "Live process snapshot for recorded 7Z sessions:"
    while IFS=$'\t' read -r name pid cores run_script logfile status_file; do
        [[ -z "${pid:-}" ]] && continue
        if is_alive "${pid}" && is_our_session "${pid}"; then
            echo "--- ${name} (SID=${pid}, CPUs=${cores}) ---"
            ps -e -o pid,ppid,sid,psr,%cpu,%mem,etime,rss,comm,args \
                | awk -v sid="${pid}" 'NR == 1 || $3 == sid'
        fi
    done < "${PID_FILE}"
}

monitor_all() {
    echo "Press Ctrl+C to leave the monitor; calculations will keep running."
    sleep 1
    while true; do
        clear || true
        echo "He/aug-cc-pV7Z monitor -- $(date '+%F %T')"
        echo
        status_all
        echo
        echo "Refresh: 10 s"
        sleep 10
    done
}

stop_jobs() {
    local target="$1"
    check_target "${target}"

    if [[ ! -s "${PID_FILE}" ]]; then
        echo "No recorded He/aug-cc-pV7Z jobs."
        exit 0
    fi

    local matched=0
    while IFS=$'\t' read -r name pid cores run_script logfile status_file; do
        [[ -z "${pid:-}" ]] && continue
        matches_target "${name}" "${target}" || continue
        matched=1

        if is_alive "${pid}" && is_our_session "${pid}"; then
            echo "Sending SIGTERM to ${name}: PID/SID=${pid}"
            kill -TERM -- "-${pid}"
        elif is_alive "${pid}"; then
            echo "SKIP ${name}: PID ${pid} exists but SID validation failed"
        else
            echo "${name}: already stopped"
        fi
    done < "${PID_FILE}"

    (( matched )) || { echo "No matching recorded job for target: ${target}"; exit 1; }

    sleep 2
    echo
    status_all
}

show_logs() {
    local target="$1"
    check_target "${target}"

    if [[ ! -s "${PID_FILE}" ]]; then
        echo "No recorded He/aug-cc-pV7Z jobs."
        exit 0
    fi

    while IFS=$'\t' read -r name pid cores run_script logfile status_file; do
        [[ -z "${logfile:-}" ]] && continue
        matches_target "${name}" "${target}" || continue
        echo "===== ${name}: ${logfile} ====="
        if [[ -f "${logfile}" ]]; then
            tail -n 30 "${logfile}"
        elif [[ -f "${logfile}.launcher" ]]; then
            tail -n 30 "${logfile}.launcher"
        else
            echo "(log not created yet)"
        fi
        echo
    done < "${PID_FILE}"
}

cmd=${1:-status}
target=${2:-all}

case "${cmd}" in
    start)   start_all ;;
    status)  status_all ;;
    monitor) monitor_all ;;
    stop)    stop_jobs "${target}" ;;
    logs)    show_logs "${target}" ;;
    *)
        echo "Usage: $0 {start|status|monitor|stop [all|singlet|triplet]|logs [all|singlet|triplet]}" >&2
        exit 2
        ;;
esac
