# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Early runtime configuration for double-precision single-GPU FermiNet runs."""

import os

# These must be set before JAX is imported for the first time.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("JAX_DEFAULT_DTYPE_BITS", "64")


def enable_x64() -> None:
  """Enable JAX x64 mode if JAX has not already been locked to x32."""
  import jax  # pylint: disable=import-outside-toplevel
  jax.config.update("jax_enable_x64", True)
