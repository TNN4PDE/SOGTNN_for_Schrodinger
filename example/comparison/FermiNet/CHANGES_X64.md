# FermiNet x64/timing/logging patch

Modified files are listed in the assistant response. Key additions:

- `x64_config.py`: early JAX x64/runtime configuration helper.
- `sitecustomize.py`: optional project-root startup hook so even `python -c "import jax"` sees x64 if run from this directory or with this directory on `PYTHONPATH`.
- `run_ferminet_x64.sh`: shell wrapper exporting the required variables before launching `python -m ferminet.main`.
- `verify_x64.py`: quick check for `jax_enable_x64` and default dtype.

For a bare Python check, either run from a directory containing `sitecustomize.py`, set `PYTHONPATH` to that directory, or export:

```bash
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=True
export JAX_DEFAULT_DTYPE_BITS=64
python - << 'EOF'
import jax
import jax.numpy as jnp
print(jax.config.read("jax_enable_x64"))
print(jnp.ones(()).dtype)
EOF
```
