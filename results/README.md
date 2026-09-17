# Numerical results

This directory contains the original numerical outputs and analysis records for the SOG-TNN examples and comparison methods.

## Execution modes

The three-dimensional Coulomb calculations reported in the manuscript were performed with `torch.compile` enabled. The original compiled-run logs are provided alongside the results of the default eager implementation.

[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) is an optional PyTorch compiler facility that can reduce execution time and memory use for tensor computations. Its effects depend on the workload and compiler settings. Compiler optimizations, such as operation fusion, can change tensor execution order and consequently alter the optimization trajectory ([PyTorch discussion of numerical changes](https://pytorch.org/blog/training-production-ai-models/)).

For reproduction, the supplied scripts use eager execution by default, without compilation or compiler autotuning. They can be run directly under the same environment and configuration as provided in [documented environment](../environment/sog-tnn/environment.txt) to reproduce the numerical results comparable to that reported in the manuscript. Both execution modes use the same source code. To enable compilation, uncomment the existing compilation configuration before `criterion`.

The following table summarizes the final relative errors and total elapsed times of the eager and compiled SOG-TNN runs for the four three-dimensional Coulomb systems. Each entry links to its original txt.

| System | Execution mode and log | Final relative Error | Total Time (s) |
| --- | --- | ---: | ---: |
| He | [Eager](coulomb_3d/He_c3d.txt) | 4.173774586396e-08 | 4566.74 |
| He | [Compiled](coulomb_3d/He_c3d_compile.txt) | 4.019714332839e-08 | 4279.24 |
| Triplet He | [Eager](coulomb_3d/He_t_c3d.txt) | 5.877529773091e-08 | 5692.18 |
| Triplet He | [Compiled](coulomb_3d/He_t_c3d_compile.txt) | 5.966159019411e-08 | 5994.52 |
| Li | [Eager](coulomb_3d/Li_c3d.txt) | 3.224893815916e-08 | 35131.99 |
| Li | [Compiled](coulomb_3d/Li_c3d_compile.txt) | 2.253551271057e-08 | 25392.55 |
| Be | [Eager](coulomb_3d/Be_c3d.txt) | 1.365670360860e-06 | 59642.76 |
| Be | [Compiled](coulomb_3d/Be_c3d_compile.txt) | 1.216495871564e-06 | 43736.15 |

## Comparison methods

The following table summarizes the final relative errors and total elapsed times of the CASSCF and FermiNet calculations for the same four systems. Each entry links to the corresponding numerical record.

| System | Method and numerical record | Final Best Error | Total Time (s) |
| --- | --- | ---: | ---: |
| He | [CASSCF](comparison/PySCF/casscf/He_singlet_CASSCF_aug7z_aug-cc-pV7Z_ncas128_macro.csv) | 6.816399434645e-05 | 34982.62 |
| He | [FermiNet](comparison/FermiNet/train_He_singlet.log) | 6.535751565851e-07 | 83747.50 |
| Triplet He | [CASSCF](comparison/PySCF/casscf/He_triplet_CASSCF_20260902_005021_aug-cc-pV6Z_ncas112_macro.csv) | 8.215332854504e-04 | 16982.41 |
| Triplet He | [FermiNet](comparison/FermiNet/train_He_triplet.log) | 1.046339439271e-04 | 58168.99 |
| Li | [CASSCF](comparison/PySCF/casscf/Li_CASSCF_20260902_005021_aug-cc-pCV5Z_ncas072_macro.csv) | 8.816493815864e-05 | 95828.08 |
| Li | [FermiNet](comparison/FermiNet/train_Li_doublet.log) | 1.876422721876e-05 | 148902.98 |
| Be | [CASSCF](comparison/PySCF/casscf/Be_CASSCF_ncas56_aug-cc-pCVQZ_ncas056_macro.csv) | 1.660073286944e-04 | 331562.92 |
| Be | [FermiNet](comparison/FermiNet/train_Be_singlet.log) | 4.727157375345e-06 | 223577.51 |