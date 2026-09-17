# SOG-TNN for Schrödinger

Sum-of-Gaussians tensor neural networks for high-dimensional Schrödinger equations.

This repository accompanies the SOG-TNN papers below. It contains the research code, precomputed kernel data, computational environment records, and numerical outputs used to study SOG-TNN for many-electron systems.

## Papers

- Qi Zhou, Teng Wu, Jianghao Liu, Qingyuan Sun, Hehu Xie, and Zhenli Xu. **Sum-of-Gaussians tensor neural networks for high-dimensional Schrödinger equation.** [arXiv:2508.10454](https://arxiv.org/abs/2508.10454).
- Teng Wu, Qi Zhou, Huangjie Zheng, Hehu Xie, and Zhenli Xu. **Spectral convergence of sum-of-Gaussians tensor neural networks for many-electron Schrödinger equation.** The Journal of Chemical Physics 164, 244103 (2026). [DOI: 10.1063/5.0335566](https://doi.org/10.1063/5.0335566).

Please cite the relevant paper when using the code or numerical data.

## Highlights

- Learnable tensor-product basis functions that adapt the Galerkin subspace with nonlinear neural-network parameterization.
- Direct SOG Coulomb tensorization with error-controlled range splitting, and exact Slater-determinant antisymmetry with efficient matrix-element assembly via Löwdin’s rules and biorthogonalization.
- A fully deterministic Galerkin framework combining neural-network approximation with high-order Gauss-type quadrature for accurate variational energy evaluation.

## Computational workflow

The offline stage constructs kernel approximations in MATLAB using the SOG, WBT, and range-splitting (RS) strategies. The MATLAB source code is provided in [src](src), and the precomputed data are supplied in [data](data).

The online stage uses these precomputed data to assemble the variational problem and optimize the tensor neural networks in Python. Shared utilities for neural networks, quadrature, and integration are provided in [sogtnn](sogtnn). The supplied data can be used directly without repeating the MATLAB preprocessing.

To run SOG-TNN, first configure the [documented environment](environment/README.md), then run the following commands from the repository root, replacing the placeholders with the appropriate names:

```bash
conda activate <environment_name>
cd example/<system_directory>
bash <script_name>.sh
```

## Repository layout

| Directory | Contents |
| --- | --- |
| [data](data) | Precomputed MATLAB kernel matrices and basis-set files for PySCF calculations. |
| [environment](environment) | Recorded software versions and computational environments for SOG-TNN, PySCF, FermiNet, and MATLAB. |
| [example](example) | Numerical examples organized by physical system, together with comparison methods. |
| [results](results) | Original calculation logs, text outputs, and result-analysis files. |
| [sogtnn](sogtnn) | Shared SOG-TNN routines. |
| [src](src) | MATLAB implementations of the offline kernel approximations and their numerical checks. |

## Numerical examples and results

| Calculation | Code | Records |
| --- | --- | --- |
| One-dimensional Poisson system | [example/poisson_1d](example/poisson_1d) | [results/poisson_1d](results/poisson_1d) |
| One-dimensional soft-Coulomb systems | [example/soft_coulomb_1d](example/soft_coulomb_1d) | [results/soft_coulomb_1d](results/soft_coulomb_1d) |
| Three-dimensional Coulomb systems | [example/coulomb_3d](example/coulomb_3d) | [results/coulomb_3d](results/coulomb_3d) |
| SG, adaptive MGF, PySCF, and FermiNet comparisons | [example/comparison](example/comparison) | [results/comparison](results/comparison) |

The example scripts contain the calculation settings; accompanying shell scripts are provided alongside the Python files.

## Computational environments

The following files record the environments used for the respective calculations, including library versions and numerical backends:

| Calculation | Environment record |
| --- | --- |
| SOG-TNN  | [PyTorch environment](environment/sog-tnn/environment.txt) |
| PySCF comparisons | [PySCF environment](environment/pyscf/environment.txt) |
| FermiNet comparisons | [FermiNet environment](environment/ferminet/environment.txt) |
| Offline MATLAB calculations | [MATLAB, toolboxes, and numerical libraries](environment/matlab/matlab-environment.txt) |

## License

The SOG-TNN code is distributed under the [MIT License](LICENSE). 

## Citation

This repository contains the code and data associated with the following
two papers. If you use SOG-TNN in your research, please cite **both papers**
and identify the software release or commit used.

1. Qi Zhou, Teng Wu, Jianghao Liu, Qingyuan Sun, Hehu Xie, and Zhenli Xu.
   “Sum-of-Gaussians tensor neural networks for high-dimensional
   Schrödinger equation.”
   arXiv preprint arXiv:2508.10454 (2025).
   https://doi.org/10.48550/arXiv.2508.10454

2. Teng Wu, Qi Zhou, Huangjie Zheng, Hehu Xie, and Zhenli Xu.
   “Spectral convergence of sum-of-Gaussians tensor neural networks
   for many-electron Schrödinger equation.”
   The Journal of Chemical Physics 164(24), 244103 (2026).
   https://doi.org/10.1063/5.0335566
