#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
SCRATCH="${SCRATCH:-${ROOT}/scratch/eval}"
GPU_ID="${GPU_ID:-2}"

# PRR-style fixed-theta evaluation:
# EVAL_STEPS=10000 and MCMC_STEPS=10 gives O(1e5) MCMC steps.
EVAL_STEPS="${EVAL_STEPS:-10000}"
MCMC_STEPS="${MCMC_STEPS:-10}"
STATS_FREQ="${STATS_FREQ:-1}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
SAVE_FREQ="${SAVE_FREQ:-1000000.0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=64
export JAX_PLATFORM_NAME=cuda
export JAX_PLATFORMS=cuda

# Replace this path with the exact checkpoint to evaluate,
# or override it from the command line:
#   CKPT_FILE=/path/to/qmcjax_ckpt_XXXXXX.npz bash run_eval_Be_singlet_prr_style.sh
CKPT_FILE="${CKPT_FILE:-${ROOT}/checkpoints_ferminet_x64_atoms/Be_singlet/qmcjax_ckpt_203979.npz}"

ATOM="Be"
SPIN_POL="0"
LABEL="Be_singlet_eval_prr_style"

if [[ "${CKPT_FILE}" == *"XXXXXX"* ]]; then
  echo "ERROR: Please edit CKPT_FILE in run_eval_Be_singlet_prr_style.sh or pass CKPT_FILE=/path/to/qmcjax_ckpt_XXXXXX.npz" >&2
  exit 1
fi

if [[ ! -f "${CKPT_FILE}" ]]; then
  echo "ERROR: CKPT_FILE does not exist: ${CKPT_FILE}" >&2
  exit 1
fi

CKPT_BASENAME="$(basename "${CKPT_FILE}")"
if [[ "${CKPT_BASENAME}" != qmcjax_ckpt_*.npz ]]; then
  echo "ERROR: checkpoint filename should look like qmcjax_ckpt_XXXXXX.npz" >&2
  exit 1
fi

RESTORE_DIR="${SCRATCH}/selected_restore/${LABEL}"
EVAL_DIR="${SCRATCH}/eval_outputs/${LABEL}"
LOGDIR="${SCRATCH}/logs"
mkdir -p "${RESTORE_DIR}" "${EVAL_DIR}" "${LOGDIR}"

# Anti-pollution: restore_path contains only the exact selected checkpoint.
rm -f "${RESTORE_DIR}"/qmcjax_ckpt_*.npz
ln -s "${CKPT_FILE}" "${RESTORE_DIR}/${CKPT_BASENAME}"

# Avoid mixing with an old eval run.
if [[ -f "${EVAL_DIR}/train_stats.csv" ]]; then
  BACKUP="${EVAL_DIR}_old_$(date +%Y%m%d_%H%M%S)"
  echo "Existing eval directory found. Moving it to ${BACKUP}"
  mv "${EVAL_DIR}" "${BACKUP}"
  mkdir -p "${EVAL_DIR}"
fi

echo "============================================================"
echo "Fixed-theta FermiNet VMC evaluation: ${LABEL}"
echo "ROOT        = ${ROOT}"
echo "GPU_ID      = ${GPU_ID}"
echo "CKPT_FILE   = ${CKPT_FILE}"
echo "RESTORE_DIR = ${RESTORE_DIR}"
echo "EVAL_DIR    = ${EVAL_DIR}"
echo "ATOM        = ${ATOM}"
echo "SPIN_POL    = ${SPIN_POL}"
echo "BATCH_SIZE  = ${BATCH_SIZE}"
echo "EVAL_STEPS  = ${EVAL_STEPS}"
echo "MCMC_STEPS  = ${MCMC_STEPS}"
echo "STATS_FREQ  = ${STATS_FREQ}"
echo "SAVE_FREQ   = ${SAVE_FREQ}"
echo "============================================================"
nvidia-smi -i "${GPU_ID}" || true

cd "${ROOT}"

nohup python -u -m ferminet.main \
  --config ferminet/configs/atom.py \
  --config.system.atom "${ATOM}" \
  --config.system.charge 0 \
  --config.system.spin_polarisation "${SPIN_POL}" \
  --config.batch_size "${BATCH_SIZE}" \
  --config.network.network_type ferminet \
  --config.network.determinants 16 \
  --config.pretrain.method hf \
  --config.pretrain.iterations 0 \
  --config.pretrain.basis ccpvdz \
  --config.pretrain.scf_fraction 1.0 \
  --config.mcmc.steps "${MCMC_STEPS}" \
  --config.mcmc.burn_in 0 \
  --config.mcmc.move_width 0.02 \
  --config.mcmc.adapt_frequency 100 \
  --config.optim.optimizer none \
  --config.optim.iterations "${EVAL_STEPS}" \
  --config.optim.clip_local_energy 5.0 \
  --config.log.restore_path "${RESTORE_DIR}" \
  --config.log.save_path "${EVAL_DIR}" \
  --config.log.stats_frequency "${STATS_FREQ}" \
  --config.log.timing_frequency 10000 \
  --config.log.save_frequency "${SAVE_FREQ}" \
  > "${LOGDIR}/eval_${LABEL}.log" 2>&1 &

echo "Launched fixed-theta evaluation: ${LABEL}"
echo "Log:      ${LOGDIR}/eval_${LABEL}.log"
echo "Stats:    ${EVAL_DIR}/train_stats.csv"
echo "Restore:  ${RESTORE_DIR}/${CKPT_BASENAME}"
echo "Eval dir: ${EVAL_DIR}"
