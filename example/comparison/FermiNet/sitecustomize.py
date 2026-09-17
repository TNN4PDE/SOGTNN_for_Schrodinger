"""Project-local Python startup hook for FermiNet x64 runs.

If this file is on PYTHONPATH or in the current working directory, Python imports
it before user code. This ensures that `import jax` sees the x64 and CUDA env
settings early enough.
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_DEFAULT_DTYPE_BITS", "64")
