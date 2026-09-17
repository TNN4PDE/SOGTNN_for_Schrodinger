#!/usr/bin/env bash
set -euo pipefail

# Robust detached launcher for the four unfinished CASSCF system scans.
#
# Default `start` scope (based on the interrupted 2026-08-31 run):
#   He_singlet: 88
#   He_triplet: 96
#   Li        : 64 72 80
#   Be        : 48 56 64 72
#
# Usage:
#   ./run_all_CASSCF_parallel.sh preflight
#   ./run_all_CASSCF_parallel.sh start
#   ./run_all_CASSCF_parallel.sh status
#   ./run_all_CASSCF_parallel.sh monitor
#   ./run_all_CASSCF_parallel.sh logs
#   ./run_all_CASSCF_parallel.sh stop
#
# To recompute every originally requested active space:
#   RUN_SCOPE=full ./run_all_CASSCF_parallel.sh start
#
# nohup + setsid + /dev/null detach all jobs from the SSH terminal. This
# survives SSH logout/disconnect, but cannot survive a node reboot or SIGKILL.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON=${PYTHON:-python}
THREADS=${THREADS:-8}
MEMORY_MB=${MEMORY_MB:-64000}
RUN_SCOPE=${RUN_SCOPE:-remaining}

STATE_DIR=${STATE_DIR:-${SCRIPT_DIR}/.casscf_state}
RESULTS_ROOT=${RESULTS_ROOT:-${SCRIPT_DIR}/Results}
PID_FILE=${STATE_DIR}/jobs.tsv
CURRENT_RUN_FILE=${STATE_DIR}/current_run.txt

mkdir -p "${STATE_DIR}" "${RESULTS_ROOT}"

timestamp() {
  date '+%Y%m%d_%H%M%S'
}

is_alive() {
  local pid=$1
  kill -0 "${pid}" 2>/dev/null
}

# Every job is created by setsid, so its session ID must equal the stored PID.
# This avoids treating an unrelated process that reused a stale PID as our job.
is_our_session() {
  local pid=$1
  local sid
  sid=$(ps -o sid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)
  [[ -n "${sid}" && "${sid}" == "${pid}" ]]
}

require_command() {
  local cmd=$1
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: required command not found: ${cmd}" >&2
    return 1
  fi
}

preflight_all() {
  local failed=0
  local cmd
  local path
  local cores

  if [[ "${RUN_SCOPE}" != "remaining" && "${RUN_SCOPE}" != "full" ]]; then
    echo "ERROR: RUN_SCOPE must be 'remaining' or 'full' (got: ${RUN_SCOPE})" >&2
    failed=1
  fi

  for cmd in nohup setsid taskset ps awk tail df; do
    require_command "${cmd}" || failed=1
  done
  require_command "${PYTHON}" || failed=1

  for path in \
    run_He_singlet_CASSCF.sh He_singlet_CASSCF_timed.py \
    run_He_triplet_CASSCF.sh He_triplet_CASSCF_timed.py \
    run_Li_CASSCF.sh Li_CASSCF_timed.py \
    run_Be_CASSCF.sh Be_CASSCF_timed.py; do
    if [[ ! -f "${SCRIPT_DIR}/${path}" ]]; then
      echo "ERROR: missing ${SCRIPT_DIR}/${path}" >&2
      failed=1
    fi
  done

  if [[ ! -f "${BASIS_FILE:-${SCRIPT_DIR}/../../../data/aug-cc-pCV5Z.gbs}" ]]; then
    echo "ERROR: Li basis file missing: ${BASIS_FILE:-${SCRIPT_DIR}/../../../data/aug-cc-pCV5Z.gbs}" >&2
    failed=1
  fi

  if (( failed == 0 )); then
    if ! "${PYTHON}" -c 'import pyscf, basis_set_exchange' >/dev/null 2>&1; then
      echo "ERROR: ${PYTHON} cannot import pyscf and basis_set_exchange." >&2
      echo "Activate the same pytorch/conda environment used for the first run." >&2
      failed=1
    fi
  fi

  # These are physical-core ranges on the supplied 2 x 32-core topology.
  for cores in 0-7 8-15 32-39 40-47; do
    if ! taskset -c "${cores}" true >/dev/null 2>&1; then
      echo "ERROR: CPU range ${cores} is outside this session's allowed CPU set." >&2
      failed=1
    fi
  done

  if (( failed )); then
    return 1
  fi

  local available_mb
  local free_kb
  available_mb=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
  free_kb=$(df -Pk "${SCRIPT_DIR}" | awk 'NR==2 {print $4}')

  echo "Preflight OK"
  echo "  Python      : $(command -v "${PYTHON}")"
  echo "  Run scope   : ${RUN_SCOPE}"
  echo "  Threads/job : ${THREADS}"
  echo "  Memory cap  : ${MEMORY_MB} MB/job (cap, not preallocated)"
  echo "  MemAvailable: ${available_mb} MB"
  echo "  Disk free   : $((free_kb / 1024)) MB at ${SCRIPT_DIR}"
  echo "  CPU sets    : 0-7, 8-15 (NUMA 0); 32-39, 40-47 (NUMA 1)"

  if (( available_mb < 131072 )); then
    echo "WARNING: less than 128 GiB memory is currently available." >&2
  fi
  if (( free_kb < 20971520 )); then
    echo "WARNING: less than 20 GiB disk space is currently free." >&2
  fi
}

# jobs.tsv fields:
# name, session_pid, cores, run_script, main_log, launcher_log, status_file
start_job() {
  local name=$1
  local cores=$2
  local run_script=$3
  local logfile=$4
  local prefix=$5
  local launcher_log=${logfile}.launcher
  local status_file=${logfile}.exit_status

  rm -f -- "${status_file}"

  # The wrapper always writes a launcher start line and, on ordinary exit or
  # catchable termination, an atomic numeric exit-status file. SIGKILL and a
  # machine reboot cannot run an exit trap and therefore remain distinguishable.
  nohup setsid taskset -c "${cores}" env \
    PYTHON="${PYTHON}" \
    THREADS="${THREADS}" \
    MEMORY_MB="${MEMORY_MB}" \
    RUN_SCOPE="${RUN_SCOPE}" \
    RUN_ID="${RUN_ID}" \
    RESULT_DIR="${RESULT_DIR}" \
    LOG="${logfile}" \
    PREFIX="${prefix}" \
    bash -c '
      run_script=$1
      status_file=$2
      job_name=$3

      finalize_job() {
        rc=$?
        trap - EXIT
        tmp_status="${status_file}.tmp.$$"
        printf "%s\n" "${rc}" > "${tmp_status}"
        mv -f -- "${tmp_status}" "${status_file}"
        printf "[LAUNCHER] end=%s job=%s rc=%s pid=%s\n" \
          "$(date "+%F %T%z")" "${job_name}" "${rc}" "$$"
        exit "${rc}"
      }

      trap finalize_job EXIT
      printf "[LAUNCHER] start=%s job=%s pid=%s script=%s\n" \
        "$(date "+%F %T%z")" "${job_name}" "$$" "${run_script}"
      bash "${run_script}"
    ' launcher-wrapper "${SCRIPT_DIR}/${run_script}" "${status_file}" "${name}" \
    > "${launcher_log}" 2>&1 < /dev/null &

  local pid=$!
  sleep 0.5

  if ! is_alive "${pid}"; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${name}" "${pid}" "${cores}" "${run_script}" "${logfile}" \
      "${launcher_log}" "${status_file}" >> "${PID_FILE}"

    if [[ -s "${status_file}" ]]; then
      local early_rc
      read -r early_rc < "${status_file}"
      if [[ "${early_rc}" == "0" ]]; then
        echo "Finished ${name} during startup check: rc=0"
        return 0
      fi
      echo "ERROR: ${name} exited during startup with rc=${early_rc}." >&2
    else
      echo "ERROR: ${name} exited during startup without a status file." >&2
    fi
    tail -n 30 "${launcher_log}" >&2 || true
    return 1
  fi
  if ! is_our_session "${pid}"; then
    echo "ERROR: ${name} started without the expected detached session." >&2
    return 1
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${pid}" "${cores}" "${run_script}" "${logfile}" \
    "${launcher_log}" "${status_file}" >> "${PID_FILE}"

  echo "Started ${name}: PID/SID=${pid}, CPU=${cores}"
  echo "  log      : ${logfile}"
  echo "  launcher : ${launcher_log}"
}

ensure_no_active_jobs() {
  if [[ ! -f "${PID_FILE}" ]]; then
    return 0
  fi

  local active=0
  local name pid cores run_script logfile launcher_log status_file
  while IFS=$'\t' read -r name pid cores run_script logfile launcher_log status_file; do
    [[ -z "${pid:-}" ]] && continue
    if is_alive "${pid}" && is_our_session "${pid}"; then
      echo "ERROR: existing job still running: ${name} PID=${pid} CPU=${cores}" >&2
      active=1
    fi
  done < "${PID_FILE}"

  if (( active )); then
    echo "Use '$0 status' or '$0 stop' before starting another set." >&2
    return 1
  fi
}

start_all() {
  preflight_all
  ensure_no_active_jobs

  RUN_ID=$(timestamp)
  RESULT_DIR=${RESULTS_ROOT}/${RUN_ID}
  export RUN_ID RESULT_DIR
  mkdir -p "${RESULT_DIR}"
  : > "${PID_FILE}"
  printf '%s\n' "${RUN_ID}" > "${CURRENT_RUN_FILE}"

  echo
  echo "Launching four detached jobs"
  echo "  run ID    : ${RUN_ID}"
  echo "  Results   : ${RESULT_DIR}"
  echo "  run scope : ${RUN_SCOPE}"
  echo

  start_job He_singlet 0-7 run_He_singlet_CASSCF.sh \
    "${RESULT_DIR}/He_singlet_${RUN_ID}.log" \
    "${RESULT_DIR}/He_singlet_CASSCF_${RUN_ID}"
  start_job He_triplet 8-15 run_He_triplet_CASSCF.sh \
    "${RESULT_DIR}/He_triplet_${RUN_ID}.log" \
    "${RESULT_DIR}/He_triplet_CASSCF_${RUN_ID}"
  start_job Li 32-39 run_Li_CASSCF.sh \
    "${RESULT_DIR}/Li_${RUN_ID}.log" \
    "${RESULT_DIR}/Li_CASSCF_${RUN_ID}"
  start_job Be 40-47 run_Be_CASSCF.sh \
    "${RESULT_DIR}/Be_${RUN_ID}.log" \
    "${RESULT_DIR}/Be_CASSCF_${RUN_ID}"

  echo
  echo "All jobs are detached. SSH logout/disconnect will not stop them."
  echo "Check with: $0 status"
}

job_state() {
  local pid=$1
  local status_file=$2

  if is_alive "${pid}" && is_our_session "${pid}"; then
    printf 'RUNNING'
  elif is_alive "${pid}"; then
    printf 'STALE_PID'
  elif [[ -s "${status_file}" ]]; then
    local rc
    read -r rc < "${status_file}"
    if [[ "${rc}" == "0" ]]; then
      printf 'SUCCESS'
    else
      printf 'FAILED(rc=%s)' "${rc}"
    fi
  else
    printf 'ABORTED(no-status)'
  fi
}

last_progress() {
  local logfile=$1
  if [[ ! -s "${logfile}" ]]; then
    printf 'log not created yet'
    return
  fi

  awk '
    /^Active space:/ {active=$0}
    /^\[MACRO_TIMING\]/ {timing=$0}
    /CASSCF converged flag/ {finished=active}
    END {
      if (timing != "") print timing
      else if (active != "") print active
      else print "initializing"
    }
  ' "${logfile}"
}

status_all() {
  if [[ ! -s "${PID_FILE}" ]]; then
    echo "No recorded CASSCF jobs."
    return 0
  fi

  if [[ -s "${CURRENT_RUN_FILE}" ]]; then
    echo "Run ID: $(<"${CURRENT_RUN_FILE}")"
  fi
  printf '%-12s %-10s %-9s %-20s %s\n' JOB PID CPU STATE LOG
  printf '%-12s %-10s %-9s %-20s %s\n' ------------ ---------- --------- -------------------- ---

  local name pid cores run_script logfile launcher_log status_file state
  while IFS=$'\t' read -r name pid cores run_script logfile launcher_log status_file; do
    [[ -z "${pid:-}" ]] && continue
    state=$(job_state "${pid}" "${status_file}")
    printf '%-12s %-10s %-9s %-20s %s\n' \
      "${name}" "${pid}" "${cores}" "${state}" "${logfile}"
    printf '  progress: %s\n' "$(last_progress "${logfile}")"
  done < "${PID_FILE}"

  echo
  echo "Live process snapshot for recorded sessions:"
  local any=0
  while IFS=$'\t' read -r name pid cores run_script logfile launcher_log status_file; do
    [[ -z "${pid:-}" ]] && continue
    if is_alive "${pid}" && is_our_session "${pid}"; then
      ps -o pid,ppid,sid,psr,%cpu,%mem,etime,rss,comm,args --sid "${pid}" || true
      any=1
    fi
  done < "${PID_FILE}"
  if (( any == 0 )); then
    echo "No recorded job session is currently alive."
  fi
}

monitor_all() {
  echo "Press Ctrl+C to leave this monitor; calculations will keep running."
  sleep 1
  while true; do
    clear || true
    echo "CASSCF monitor -- $(date '+%F %T%z')"
    echo
    status_all
    echo
    echo "Refresh: 10 s"
    sleep 10
  done
}

stop_all() {
  if [[ ! -s "${PID_FILE}" ]]; then
    echo "No recorded CASSCF jobs."
    return 0
  fi

  echo "Sending SIGTERM to recorded CASSCF process groups..."
  local name pid cores run_script logfile launcher_log status_file
  while IFS=$'\t' read -r name pid cores run_script logfile launcher_log status_file; do
    [[ -z "${pid:-}" ]] && continue
    if is_alive "${pid}" && is_our_session "${pid}"; then
      echo "  stopping ${name}: PID/SID=${pid}"
      kill -TERM -- "-${pid}" 2>/dev/null || true
    elif is_alive "${pid}"; then
      echo "  skip ${name}: PID ${pid} exists but session validation failed"
    else
      echo "  ${name}: already stopped"
    fi
  done < "${PID_FILE}"

  sleep 3
  echo
  status_all
}

show_logs() {
  if [[ ! -s "${PID_FILE}" ]]; then
    echo "No recorded CASSCF jobs."
    return 0
  fi

  local name pid cores run_script logfile launcher_log status_file
  while IFS=$'\t' read -r name pid cores run_script logfile launcher_log status_file; do
    [[ -z "${name:-}" ]] && continue
    echo "===== ${name}: calculation log ====="
    if [[ -f "${logfile}" ]]; then
      tail -n 20 "${logfile}"
    else
      echo "(not created yet)"
    fi
    echo "===== ${name}: launcher log ====="
    if [[ -f "${launcher_log}" ]]; then
      tail -n 20 "${launcher_log}"
    else
      echo "(not created yet)"
    fi
    echo
  done < "${PID_FILE}"
}

cmd=${1:-start}
case "${cmd}" in
  preflight) preflight_all ;;
  start) start_all ;;
  status) status_all ;;
  monitor) monitor_all ;;
  logs) show_logs ;;
  stop) stop_all ;;
  *)
    echo "Usage: $0 {preflight|start|status|monitor|logs|stop}" >&2
    exit 2
    ;;
esac
