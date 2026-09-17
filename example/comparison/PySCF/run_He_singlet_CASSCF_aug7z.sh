#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

PYTHON=${PYTHON:-python}
SCRIPT=${SCRIPT:-${SCRIPT_DIR}/He_singlet_CASSCF_aug7z.py}
BASIS=${BASIS:-aug-cc-pV7Z}
BASIS_SOURCE=file
BASIS_FILE=${BASIS_FILE:-${SCRIPT_DIR}/../../../data/He_aug-cc-pV7Z.nwchem}

THREADS=${THREADS:-8}
MEMORY_MB=${MEMORY_MB:-64000}
MAX_MACRO=${MAX_MACRO:-150}
MAX_MICRO=${MAX_MICRO:-4}
MAX_STEPSIZE=${MAX_STEPSIZE:-0.02}
CONV_TOL=${CONV_TOL:-1e-12}
CONV_TOL_GRAD=${CONV_TOL_GRAD:-1e-7}
FCI_CONV_TOL=${FCI_CONV_TOL:-1e-10}
EXPECTED_NAO=${EXPECTED_NAO:-189}
MIN_OVERLAP_EIG=${MIN_OVERLAP_EIG:-1e-8}

# Includes overlap with the 6Z run, then extends to the full 189-orbital FCI limit.
read -r -a NCAS <<< "${NCAS_LIST:-80 96 112 128}"

[[ -f "${SCRIPT}" ]] || { echo "ERROR: Python script not found: ${SCRIPT}" >&2; exit 2; }
[[ -f "${BASIS_FILE}" ]] || { echo "ERROR: basis file not found: ${BASIS_FILE}" >&2; exit 2; }

STAMP=$(date +%Y%m%d_%H%M%S)
LOG=${LOG:-He_singlet_aug7z_${STAMP}.log}
PREFIX=${PREFIX:-He_singlet_CASSCF_aug7z}

exec "${PYTHON}" -u "${SCRIPT}" \
  --basis "${BASIS}" \
  --basis-source "${BASIS_SOURCE}" \
  --basis-file "${BASIS_FILE}" \
  --expected-nao "${EXPECTED_NAO}" \
  --min-overlap-eig "${MIN_OVERLAP_EIG}" \
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
