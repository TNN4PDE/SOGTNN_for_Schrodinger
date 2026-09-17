#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-${SCRIPT_DIR}/Be_CASSCF_timed.py}

BASIS=${BASIS:-aug-cc-pCVQZ}
BASIS_SOURCE=${BASIS_SOURCE:-bse}
THREADS=${THREADS:-8}
MEMORY_MB=${MEMORY_MB:-64000}
MAX_MACRO=${MAX_MACRO:-120}
MAX_MICRO=${MAX_MICRO:-4}
MAX_STEPSIZE=${MAX_STEPSIZE:-0.02}
CONV_TOL=${CONV_TOL:-1e-12}
CONV_TOL_GRAD=${CONV_TOL_GRAD:-1e-7}
FCI_CONV_TOL=${FCI_CONV_TOL:-1e-10}
RUN_SCOPE=${RUN_SCOPE:-remaining}

# The 2026-08-31 run completed 8..40 and was interrupted inside ncas=48.
if [[ -n "${NCAS_VALUES:-}" ]]; then
  read -r -a NCAS <<< "${NCAS_VALUES}"
elif [[ "${RUN_SCOPE}" == "remaining" ]]; then
  NCAS=(48 56 64 72 80 88)
elif [[ "${RUN_SCOPE}" == "full" ]]; then
  NCAS=(8 16 24 32 40 48 56 64 72 80 88)
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

STAMP=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
RESULT_DIR=${RESULT_DIR:-${SCRIPT_DIR}/Results/${STAMP}}
mkdir -p "${RESULT_DIR}"
LOG=${LOG:-${RESULT_DIR}/Be_${STAMP}.log}
PREFIX=${PREFIX:-${RESULT_DIR}/Be_CASSCF_${STAMP}}

echo "[JOB] Be start=$(date '+%F %T%z') pid=$$ ncas=${NCAS[*]} threads=${THREADS}"
echo "[JOB] log=${LOG}"

"${PYTHON}" -u "${SCRIPT}" \
  --basis "${BASIS}" \
  --basis-source "${BASIS_SOURCE}" \
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

echo "[JOB] Be completed=$(date '+%F %T%z')"
