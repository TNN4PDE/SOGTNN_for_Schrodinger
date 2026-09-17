#!/usr/bin/env python3
"""
He(1S) CASCI/CASSCF benchmark with per-macro-iteration wall timing.

This script is intended for controlled comparison with SOG-TNN.

What is optimized?
------------------
CASCI:
    CI/configuration coefficients only; orbitals are fixed at the SCF orbitals.

CASSCF:
    CI/configuration coefficients + non-redundant orbital rotations.
    PySCF parameterizes orbital rotations on the orthonormal MO manifold.
    The orbitals remain inside the finite AO basis span; Gaussian exponents
    and primitive basis functions themselves are NOT optimized.

Timing
------
For each active-space size, the script writes a *_macro.csv file in real time.
The first CASSCF AO->MO transformation + initial CASCI solve is reported as
"initialization".  Each subsequent callback corresponds to one completed
macro iteration.  Thus macro_step_wall_s is the wall time of that macro step
(excluding the initial CASSCF setup), and cumulative_casscf_wall_s includes
the initialization.

Use --max-cycle-macro to choose how many macro iterations are allowed.
For error-vs-time plots, use the per-macro CSV rather than relying only on
PySCF's final ``converged`` flag.
"""

import argparse
import csv
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

SYSTEM_LABEL = 'He(1S)'
ATOM_SPEC = 'He 0 0 0'
ELEMENT = 'He'
CHARGE = 0
SPIN = 0          # PySCF convention: N_alpha - N_beta = 2S
NELECAS = (1, 1) # all electrons active in the default benchmark
E_REF = -2.9037243770341146
SCF_KIND = 'RHF'


def parse_args():
    p = argparse.ArgumentParser(description=f"{SYSTEM_LABEL} CASCI/CASSCF benchmark.")
    p.add_argument("--basis", default='aug-cc-pV6Z')
    p.add_argument(
        "--basis-source", choices=["auto", "pyscf", "bse", "file"], default="bse",
        help="Use BSE for large Dunning bases not bundled with PySCF."
    )
    p.add_argument("--basis-file", default=None,
                   help="Local NWChem-format basis file with --basis-source file.")
    p.add_argument("--ncas", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory-mb", type=int, default=64000)
    p.add_argument("--conv-tol", type=float, default=1e-12)
    p.add_argument("--conv-tol-grad", type=float, default=1e-7)
    p.add_argument("--fci-conv-tol", type=float, default=1e-10)
    p.add_argument("--max-cycle-macro", type=int, default=100)
    p.add_argument("--max-cycle-micro", type=int, default=4)
    p.add_argument("--max-stepsize", type=float, default=0.02)
    p.add_argument("--prefix", default='He_singlet_CASSCF')
    p.add_argument("--eref", type=float, default=E_REF)
    p.add_argument("--log-file", default=None)
    return p.parse_args()


args = parse_args()

# Redirect before importing numerical libraries so all PySCF output is captured.
_log_handle = None
if args.log_file:
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    _log_handle = open(args.log_file, "a", buffering=1, encoding="utf-8")
    sys.stdout = _log_handle
    sys.stderr = _log_handle

# Limit threaded BLAS/OpenMP before NumPy/PySCF imports.
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = str(args.threads)

import numpy as np
import pyscf
from pyscf import gto, scf, mcscf, lib


def peak_rss_mb():
    # Linux ru_maxrss is KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def rel_error(e):
    return abs(float(e) - args.eref) / abs(args.eref)


def n_det(ncas):
    na, nb = NELECAS
    return math.comb(ncas, na) * math.comb(ncas, nb)


def safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def load_basis():
    src = args.basis_source
    if src == "file":
        if not args.basis_file:
            raise ValueError("--basis-file is required with --basis-source file")
        txt = (Path(__file__).resolve().parents[3] / 'data' / Path(args.basis_file).expanduser()).read_text(encoding="utf-8", errors="ignore")
        return {ELEMENT: gto.basis.parse(txt, symb=ELEMENT)}

    if src == "bse":
        try:
            import basis_set_exchange as bse
        except ImportError as exc:
            raise RuntimeError(
                "basis_set_exchange is required. Install once with:\n"
                "  python -m pip install --user basis_set_exchange"
            ) from exc
        nwchem = bse.get_basis(args.basis, elements=[ELEMENT], fmt="nwchem")
        return {ELEMENT: gto.basis.parse(nwchem, symb=ELEMENT)}

    if src == "pyscf":
        return {ELEMENT: gto.basis.load(args.basis, ELEMENT)}

    return args.basis


def make_scf(mol):
    if SCF_KIND == "RHF":
        mf = scf.RHF(mol)
    elif SCF_KIND == "ROHF":
        mf = scf.ROHF(mol)
    else:
        raise ValueError(f"Unknown SCF_KIND={SCF_KIND}")
    mf.max_memory = args.memory_mb
    mf.conv_tol = min(args.conv_tol, 1e-10)
    mf.max_cycle = 100
    return mf


def make_casscf(mf, ncas):
    # One-step is PySCF's standard CASSCF implementation.
    mc = mcscf.CASSCF(mf, ncas, NELECAS)
    mc.max_memory = args.memory_mb
    mc.conv_tol = args.conv_tol
    mc.conv_tol_grad = args.conv_tol_grad
    mc.max_cycle_macro = args.max_cycle_macro
    mc.max_cycle_micro = args.max_cycle_micro
    mc.max_stepsize = args.max_stepsize
    mc.fcisolver.conv_tol = args.fci_conv_tol
    mc.fcisolver.spin = SPIN
    return mc


def exact_nonredundant_rotation_count(mc, nmo):
    # This is the exact Boolean mask used internally by PySCF for unique
    # orbital-rotation variables (core-active/core-virtual/active-virtual;
    # redundant within-subspace rotations excluded by default).
    mask = mc.uniq_var_indices(nmo, mc.ncore, mc.ncas, mc.frozen)
    return int(np.count_nonzero(mask))


def main():
    lib.num_threads(args.threads)
    basis_spec = load_basis()

    mol = gto.M(
        atom=ATOM_SPEC,
        unit="Bohr",
        charge=CHARGE,
        spin=SPIN,
        basis=basis_spec,
        symmetry=False,
        verbose=4,
        max_memory=args.memory_mb,
    )

    print("=" * 92, flush=True)
    print(f"{SYSTEM_LABEL} PySCF CASCI/CASSCF benchmark", flush=True)
    print(f"PySCF version       : {pyscf.__version__}", flush=True)
    print(f"Basis requested     : {args.basis}", flush=True)
    print(f"Basis source        : {args.basis_source}", flush=True)
    print(f"SCF reference       : {SCF_KIND}", flush=True)
    print(f"System spin (2S)    : {SPIN}", flush=True)
    print(f"Active electrons    : alpha/beta = {NELECAS}", flush=True)
    print(f"Threads             : {args.threads}", flush=True)
    print(f"Memory cap (MB)     : {args.memory_mb}", flush=True)
    print(f"conv_tol            : {args.conv_tol:.3e}", flush=True)
    print(f"conv_tol_grad       : {args.conv_tol_grad:.3e}", flush=True)
    print(f"FCI conv_tol        : {args.fci_conv_tol:.3e}", flush=True)
    print(f"max_cycle_macro     : {args.max_cycle_macro}", flush=True)
    print(f"max_cycle_micro     : {args.max_cycle_micro}", flush=True)
    print(f"max_stepsize        : {args.max_stepsize}", flush=True)
    print(f"Reference energy    : {args.eref:.16f} Ha", flush=True)
    print(f"Host                : {platform.node()}", flush=True)
    print("=" * 92, flush=True)

    mf = make_scf(mol)
    t0 = time.perf_counter()
    e_scf = mf.kernel()
    t_scf = time.perf_counter() - t0
    if not mf.converged:
        raise RuntimeError(f"{SCF_KIND} did not converge.")

    nmo = int(mf.mo_coeff.shape[1])
    nao = int(mol.nao_nr())

    print(f"\n{SCF_KIND} energy         : {e_scf:.16f} Ha", flush=True)
    print(f"{SCF_KIND} relative error : {rel_error(e_scf):.6e}", flush=True)
    print(f"{SCF_KIND} wall time      : {t_scf:.6f} s", flush=True)
    print(f"NAO / NMO                  : {nao} / {nmo}", flush=True)
    print(f"Peak RSS after SCF         : {peak_rss_mb():.2f} MB", flush=True)

    results = {
        "metadata": {
            "system": SYSTEM_LABEL,
            "reference_energy_hartree": args.eref,
            "basis": args.basis,
            "basis_source": args.basis_source,
            "threads": args.threads,
            "memory_cap_mb": args.memory_mb,
            "pyscf_version": pyscf.__version__,
            "host": platform.node(),
            "nao": nao,
            "nmo": nmo,
            "spin_2S": SPIN,
            "nelecas": list(NELECAS),
            "scf_kind": SCF_KIND,
            "scf_energy_hartree": float(e_scf),
            "scf_relative_error": rel_error(e_scf),
            "scf_wall_seconds": t_scf,
        },
        "active_space_results": [],
    }

    min_ncas = max(NELECAS)
    valid_ncas = sorted(set(n for n in args.ncas if min_ncas <= n <= nmo))
    skipped = sorted(set(args.ncas) - set(valid_ncas))
    if skipped:
        print(f"Skipping invalid ncas values (allowed {min_ncas}..{nmo}): {skipped}", flush=True)

    safe_basis = args.basis.replace("*", "star").replace("/", "_").replace(" ", "_")

    for ncas in valid_ncas:
        ndet = n_det(ncas)

        print("\n" + "-" * 92, flush=True)
        print(f"Active space: CAS({sum(NELECAS)}e,{ncas}o), alpha/beta={NELECAS}", flush=True)
        print(f"Determinant dimension N_det : {ndet}", flush=True)

        # CASCI with fixed SCF orbitals
        casci = mcscf.CASCI(mf, ncas, NELECAS)
        casci.max_memory = args.memory_mb
        casci.fcisolver.conv_tol = args.fci_conv_tol
        casci.fcisolver.spin = SPIN

        t0 = time.perf_counter()
        casci_out = casci.kernel(mf.mo_coeff)
        t_casci = time.perf_counter() - t0
        e_casci = float(casci_out[0])

        print(f"CASCI energy             : {e_casci:.16f} Ha", flush=True)
        print(f"CASCI relative error     : {rel_error(e_casci):.6e}", flush=True)
        print(f"CASCI wall time          : {t_casci:.6f} s", flush=True)

        # CASSCF with orbital optimization
        mc = make_casscf(mf, ncas)
        nrot = exact_nonredundant_rotation_count(mc, nmo)
        ncore = int(mc.ncore)
        nvir = int(nmo - ncore - ncas)
        full_active = (ncore == 0 and ncas == nmo)

        print(f"ncore / nactive / nvirtual: {ncore} / {ncas} / {nvir}", flush=True)
        print(f"Exact PySCF nonredundant orbital-rotation variables: {nrot}", flush=True)
        if full_active:
            print(
                "NOTE: all MOs are active. CASCI is FCI in this finite AO space and "
                "there are no nonredundant orbital rotations; CASSCF cannot lower "
                "the energy further except for numerical noise.",
                flush=True,
            )

        macro_path = Path(f"{args.prefix}_{safe_basis}_ncas{ncas:03d}_macro.csv")
        macro_file = macro_path.open("w", newline="", encoding="utf-8", buffering=1)
        macro_fields = [
            "macro", "energy_hartree", "relative_error", "dE",
            "grad_orb", "grad_ci", "ddm", "max_rot",
            "jk_this_macro", "micro_this_macro",
            "total_jk", "total_micro",
            "macro_step_wall_s", "cumulative_macro_wall_s",
            "cumulative_casscf_wall_s", "cumulative_peak_rss_mb",
        ]
        writer = csv.DictWriter(macro_file, fieldnames=macro_fields)
        writer.writeheader()
        macro_file.flush()

        tracker = {
            "kernel_start": None,
            "initial_end": None,
            "prev_callback": None,
            "macro_rows": [],
        }

        # Wrap the CASSCF object's CASCI method only to identify the end of
        # CASSCF initialization. The first call is the initial CASCI solve
        # before macro iteration 1. This leaves the numerical algorithm unchanged.
        original_casci = mc.casci

        def timed_casci(*a, **kw):
            out = original_casci(*a, **kw)
            if tracker["initial_end"] is None:
                now = time.perf_counter()
                tracker["initial_end"] = now
                tracker["prev_callback"] = now
                init_s = now - tracker["kernel_start"]
                print(f"[TIMING] CASSCF initialization = {init_s:.6f} s", flush=True)
            return out

        mc.casci = timed_casci

        def callback(envs):
            now = time.perf_counter()
            prev = tracker["prev_callback"]
            if prev is None:
                prev = tracker["kernel_start"]
            macro_step = now - prev
            tracker["prev_callback"] = now

            initial_end = tracker["initial_end"]
            cum_macro = None if initial_end is None else now - initial_end
            cum_casscf = now - tracker["kernel_start"]

            imacro = int(envs.get("imacro", len(tracker["macro_rows"]) + 1))
            e_tot = safe_float(envs.get("e_tot"))
            row = {
                "macro": imacro,
                "energy_hartree": e_tot,
                "relative_error": None if e_tot is None else rel_error(e_tot),
                "dE": safe_float(envs.get("de")),
                "grad_orb": safe_float(envs.get("norm_gorb0", envs.get("norm_gorb"))),
                "grad_ci": safe_float(envs.get("norm_gci")),
                "ddm": safe_float(envs.get("norm_ddm")),
                "max_rot": safe_float(envs.get("max_offdiag_u")),
                "jk_this_macro": int(envs["njk"]) if envs.get("njk") is not None else None,
                "micro_this_macro": int(envs["imicro"]) if envs.get("imicro") is not None else None,
                "total_jk": int(envs["totinner"]) if envs.get("totinner") is not None else None,
                "total_micro": int(envs["totmicro"]) if envs.get("totmicro") is not None else None,
                "macro_step_wall_s": macro_step,
                "cumulative_macro_wall_s": cum_macro,
                "cumulative_casscf_wall_s": cum_casscf,
                "cumulative_peak_rss_mb": peak_rss_mb(),
            }
            writer.writerow(row)
            macro_file.flush()
            tracker["macro_rows"].append(row)

            print(
                "[MACRO_TIMING] "
                f"ncas={ncas:3d} macro={imacro:3d} "
                f"step={macro_step:10.6f} s "
                f"cum_CASSCF={cum_casscf:10.6f} s "
                f"E={e_tot:.15f} "
                f"Er={row['relative_error']:.6e}",
                flush=True,
            )

        mc.callback = callback

        tracker["kernel_start"] = time.perf_counter()
        casscf_out = mc.kernel(mf.mo_coeff)
        t_casscf = time.perf_counter() - tracker["kernel_start"]
        e_casscf = float(casscf_out[0])
        macro_file.flush()
        macro_file.close()

        init_s = None
        if tracker["initial_end"] is not None:
            init_s = tracker["initial_end"] - tracker["kernel_start"]

        print(f"CASSCF energy            : {e_casscf:.16f} Ha", flush=True)
        print(f"CASSCF relative error    : {rel_error(e_casscf):.6e}", flush=True)
        print(f"CASSCF total wall time   : {t_casscf:.6f} s", flush=True)
        print(f"CASSCF initialization    : {init_s} s", flush=True)
        print(f"CASSCF converged flag    : {bool(mc.converged)}", flush=True)
        print(f"CASCI-CASSCF energy gain : {e_casci - e_casscf:.6e} Ha", flush=True)
        print(f"Macro timing CSV         : {macro_path}", flush=True)

        results["active_space_results"].append({
            "ncas": ncas,
            "nelecas_alpha": NELECAS[0],
            "nelecas_beta": NELECAS[1],
            "n_det": ndet,
            "ncore": ncore,
            "nvirtual": nvir,
            "exact_nonredundant_orbital_rotations": nrot,
            "full_active_space": full_active,
            "casci_energy_hartree": e_casci,
            "casci_relative_error": rel_error(e_casci),
            "casci_wall_seconds": t_casci,
            "casscf_energy_hartree": e_casscf,
            "casscf_relative_error": rel_error(e_casscf),
            "casscf_wall_seconds": t_casscf,
            "casscf_initialization_seconds": init_s,
            "casscf_converged": bool(mc.converged),
            "n_macro_recorded": len(tracker["macro_rows"]),
            "macro_timing_csv": str(macro_path),
            "peak_rss_mb": peak_rss_mb(),
            "orbital_relaxation_gain_hartree": e_casci - e_casscf,
        })

    json_path = Path(f"{args.prefix}_{safe_basis}.json")
    csv_path = Path(f"{args.prefix}_{safe_basis}.csv")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    summary_fields = [
        "ncas", "nelecas_alpha", "nelecas_beta", "n_det",
        "ncore", "nvirtual", "exact_nonredundant_orbital_rotations",
        "full_active_space", "casci_energy_hartree", "casci_relative_error",
        "casci_wall_seconds", "casscf_energy_hartree", "casscf_relative_error",
        "casscf_wall_seconds", "casscf_initialization_seconds",
        "casscf_converged", "n_macro_recorded", "macro_timing_csv",
        "peak_rss_mb", "orbital_relaxation_gain_hartree",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(results["active_space_results"])

    print("\n" + "=" * 92, flush=True)
    print(f"Saved summary JSON: {json_path}", flush=True)
    print(f"Saved summary CSV : {csv_path}", flush=True)
    print("=" * 92, flush=True)

    if _log_handle is not None:
        _log_handle.flush()


if __name__ == "__main__":
    main()
