#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${SCRIPT_DIR}}"
GPU_ID="${GPU_ID:-5}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=64
export JAX_PLATFORM_NAME=cuda
export JAX_PLATFORMS=cuda

mkdir -p "${ROOT}/logs_ferminet_x64_atoms"
mkdir -p "${ROOT}/checkpoints_ferminet_x64_atoms"

echo "============================================================"
echo "ROOT=${ROOT}"
echo "GPU_ID=${GPU_ID}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "JAX_ENABLE_X64=${JAX_ENABLE_X64}"
echo "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "Checking GPU visibility before launching FermiNet..."
nvidia-smi -i "${GPU_ID}" || true
echo "============================================================"

cd "${ROOT}"

LOGDIR="${ROOT}/logs_ferminet_x64_atoms"
CKPTDIR="${ROOT}/checkpoints_ferminet_x64_atoms/He_singlet"
mkdir -p "${LOGDIR}" "${CKPTDIR}"

nohup python -u -m ferminet.main \
  --config ferminet/configs/atom.py \
  --config.system.atom He \
  --config.system.charge 0 \
  --config.system.spin_polarisation 0 \
  --config.batch_size 4096 \
  --config.network.network_type ferminet \
  --config.network.determinants 16 \
  --config.pretrain.method hf \
  --config.pretrain.iterations 1000 \
  --config.pretrain.basis ccpvdz \
  --config.pretrain.scf_fraction 1.0 \
  --config.mcmc.steps 10 \
  --config.mcmc.burn_in 100 \
  --config.mcmc.move_width 0.02 \
  --config.mcmc.adapt_frequency 100 \
  --config.optim.optimizer kfac \
  --config.optim.iterations 1000000 \
  --config.optim.lr.rate 0.05 \
  --config.optim.lr.delay 10000.0 \
  --config.optim.lr.decay 1.0 \
  --config.optim.clip_local_energy 5.0 \
  --config.optim.kfac.damping 0.001 \
  --config.optim.kfac.norm_constraint 0.001 \
  --config.optim.kfac.momentum 0.0 \
  --config.optim.kfac.cov_update_every 1 \
  --config.optim.kfac.invert_every 1 \
  --config.log.stats_frequency 1 \
  --config.log.timing_frequency 10000 \
  --config.log.save_frequency 100.0 \
  --config.log.save_path "${CKPTDIR}" \
  > "${LOGDIR}/train_He_singlet.log" 2>&1 &

echo "Launched He_singlet on physical GPU ${GPU_ID}."
echo "Log: ${LOGDIR}/train_He_singlet.log"
echo "Checkpoint: ${CKPTDIR}"
