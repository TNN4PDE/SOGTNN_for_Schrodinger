#!/usr/bin/env python3
import jax
import jax.numpy as jnp
print("jax_enable_x64 =", jax.config.read("jax_enable_x64"))
print("default dtype   =", jnp.ones(()).dtype)
print("visible devices =", jax.devices())
