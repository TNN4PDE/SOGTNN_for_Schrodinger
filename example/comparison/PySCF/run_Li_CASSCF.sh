#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-${SCRIPT_DIR}/Li_CASSCF_timed.py}

BASIS=${BASIS:-aug-cc-pCV5Z}
BASIS_SOURCE=${BASIS_SOURCE:-file}
BASIS_FILE=${BASIS_FILE:-${SCRIPT_DIR}/../../../data/aug-cc-pCV5Z.gbs}
THREADS=${THREADS:-8}
MEMORY_MB=${MEMORY_MB:-64000}
MAX_MACRO=${MAX_MACRO:-120}
MAX_MICRO=${MAX_MICRO:-4}
MAX_STEPSIZE=${MAX_STEPSIZE:-0.02}
CONV_TOL=${CONV_TOL:-1e-12}
CONV_TOL_GRAD=${CONV_TOL_GRAD:-1e-7}
FCI_CONV_TOL=${FCI_CONV_TOL:-1e-10}
RUN_SCOPE=${RUN_SCOPE:-remaining}

# The 2026-08-31 run completed 16..56 and was interrupted inside ncas=64.
if [[ -n "${NCAS_VALUES:-}" ]]; then
  read -r -a NCAS <<< "${NCAS_VALUES}"
elif [[ "${RUN_SCOPE}" == "remaining" ]]; then
  NCAS=(64 72 80 88 96)
elif [[ "${RUN_SCOPE}" == "full" ]]; then
  NCAS=(16 24 32 40 48 56 64 72 80 88 96)
else
  echo "ERROR: RUN_SCOPE must be 'remaining' or 'full' (got: ${RUN_SCOPE})" >&2
  exit 2
fi

if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: Python script not found: ${SCRIPT}" >&2
  exit 2
fi
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON}" >&2
  exit 2
fi
if [[ "${BASIS_SOURCE}" == "file" && ! -f "${BASIS_FILE}" ]]; then
  echo "ERROR: basis file not found: ${BASIS_FILE}" >&2
  echo "Place aug-cc-pCV5Z.gbs next to this script or set BASIS_FILE=/full/path/to/file.gbs" >&2
  exit 2
fi

STAMP=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RESULT_DIR=${RESULT_DIR:-${SCRIPT_DIR}/Results/${STAMP}}
mkdir -p "${RESULT_DIR}"
LOG=${LOG:-${RESULT_DIR}/Li_${STAMP}.log}
PREFIX=${PREFIX:-${RESULT_DIR}/Li_CASSCF_${STAMP}}

echo "[JOB] Li start=$(date '+%F %T%z') pid=$$ ncas=${NCAS[*]} threads=${THREADS}"
echo "[JOB] log=${LOG}"

"${PYTHON}" -u "${SCRIPT}" \
  --basis "${BASIS}" \
  --basis-source "${BASIS_SOURCE}" \
  --basis-file "${BASIS_FILE}" \
  --ncas "${NCAS[@]}" \
  --threads "${THREADS}" \
  --memory-mb "${MEMORY_MB}" \
  --conv-tol "${CONV_TOL}" \
  --conv-tol-grad "${CONV_TOL_GRAD}" \
  --fci-conv-tol "${FCI_CONV_TOL}" \
  --max-cycle-macro "${MAX_MACRO}" \
  --max-cycle-micro "${MAX_MICRO}" \
  --max-stepsize "${MAX_STEPSIZE}" \
  --prefix "${PREFIX}" \
  --log-file "${LOG}"

echo "[JOB] Li completed=$(date '+%F %T%z')"
