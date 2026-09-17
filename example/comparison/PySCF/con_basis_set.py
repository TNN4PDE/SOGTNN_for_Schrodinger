#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PySCF classified Dunning benchmark v6

Systems:
  He_singlet
  He_triplet
  Li_doublet
  Be_singlet

Basis policy:
  He: cc-pVXZ / aug-cc-pVXZ up to 6Z
  Li: cc-pVXZ / cc-pCVXZ / cc-pwCVXZ and aug variants up to QZ
  Be: cc-pVXZ / cc-pCVXZ / cc-pwCVXZ and aug variants up to TZ

Methods:
  SCF
  FCI
  CISD
  CCSD
  CCSD(T)

Major fixes in v6:
  1. Open-shell UCCSD/UCCSD(T) bug workaround:
       Some PySCF/lib.einsum combinations may fail in UCCSD with
       ValueError: not enough values to unpack ...
       This script patches pyscf.lib.einsum to numpy.einsum by default.
       Disable with --no-einsum-patch if your PySCF build is fine.

  2. He_triplet special case:
       Neutral He triplet has N_alpha=2, N_beta=0.
       UCCSD can fail because the beta occupied block is empty.
       For this two-electron high-spin sector, in a finite basis:
           CISD = CCSD = FCI, and (T) = 0.

  3. Size metrics carefully distinguish:
       basis_nao / basis_nbas / basis_lmax
       FCI_dim
       CISD_dim with same-spin antisymmetry
       CCSD amplitude dimension with same-spin antisymmetry
       CCSD(T) triples excitation dimension with same-spin antisymmetry
       optional RHF/ROHF spatial compressed proxy
       naive tensor-storage proxy

Important size definitions:
  FCI_dim:
    C(norb, n_alpha) * C(norb, n_beta)

  CISD_dim_antisym:
    1 + singles + antisymmetrized doubles

  CCSD_amp_dim_antisym:
    T1 independent spin-resolved amplitudes + antisymmetrized T2 amplitudes

  CCSD_T_triples_dim_antisym:
    independent triple-excitation manifold size with same-spin antisymmetry

Same-spin antisymmetry:
  aa doubles: C(nocc_a,2) * C(nvir_a,2)
  bb doubles: C(nocc_b,2) * C(nvir_b,2)

  aaa triples: C(nocc_a,3) * C(nvir_a,3)
  aab triples: C(nocc_a,2) * nocc_b * C(nvir_a,2) * nvir_b
  abb triples: nocc_a * C(nocc_b,2) * nvir_a * C(nvir_b,2)
  bbb triples: C(nocc_b,3) * C(nvir_b,3)
"""

import argparse
import time
import math
import traceback

import numpy as np
import pandas as pd

from pyscf import gto, scf, fci, ci, cc, lib


# ============================================================
# Systems
# ============================================================

SYSTEMS = {
    # "He_singlet": {
    #     "atom": "He 0 0 0",
    #     "charge": 0,
    #     "spin": 0,
    #     "unit": "Bohr",
    #     "scf_fci": "RHF",
    #     "scf_cisd": "RHF",
    #     "scf_cc": "RHF",
    #     "description": "He singlet ground-state sector",
    # },

    # "He_triplet": {
    #     "atom": "He 0 0 0",
    #     "charge": 0,
    #     "spin": 2,   # N_alpha - N_beta = 2, so neutral He has (2,0)
    #     "unit": "Bohr",
    #     "scf_fci": "ROHF",
    #     "scf_cisd": "ROHF",
    #     "scf_cc": "ROHF",
    #     "cc_special": "two_electron_highspin_fci_equivalent",
    #     "description": "He high-spin triplet sector, approximately 1s2s 3S",
    # },

    # "Li_doublet": {
    #     "atom": "Li 0 0 0",
    #     "charge": 0,
    #     "spin": 1,   # N_alpha - N_beta = 1, neutral Li has (2,1)
    #     "unit": "Bohr",
    #     "scf_fci": "ROHF",
    #     "scf_cisd": "UHF",
    #     "scf_cc": "UHF",
    #     "description": "Li doublet ground-state sector",
    # },

    "Be_singlet": {
        "atom": "Be 0 0 0",
        "charge": 0,
        "spin": 0,
        "unit": "Bohr",
        "scf_fci": "RHF",
        "scf_cisd": "RHF",
        "scf_cc": "RHF",
        "description": "Be singlet ground-state sector",
    },
}


# ============================================================
# Basis policy
# ============================================================

BASIS_POLICY = {
    # "He_singlet": {
    #     "families": ["cc-pv"],
    #     "aug_prefixes": ["", "aug-"],
    #     "zetas": ["dz", "tz", "qz", "5z", "6z"],
    # },

    # "He_triplet": {
    #     "families": ["cc-pv"],
    #     "aug_prefixes": ["", "aug-"],
    #     "zetas": ["dz", "tz", "qz", "5z", "6z"],
    # },

    # "Li_doublet": {
    #     "families": ["cc-pv", "cc-pcv"],
    #     "aug_prefixes": ["", "aug-"],
    #     "zetas": ["dz", "tz", "qz", "5z"],
    # },

    "Be_singlet": {
        "families": ["cc-pv", "cc-pcv"],
        "aug_prefixes": ["", "aug-"],
        "zetas": ["qz"],
    },
}


def make_basis_list_for_system(system_name):
    policy = BASIS_POLICY[system_name]
    out = []

    for aug in policy["aug_prefixes"]:
        for fam in policy["families"]:
            for zeta in policy["zetas"]:
                out.append(f"{aug}{fam}{zeta}".lower())

    seen = set()
    final = []
    for b in out:
        if b not in seen:
            final.append(b)
            seen.add(b)

    return final


def basis_family_label(basis_name):
    b = basis_name.lower()
    aug_level = "none"
    core = b

    for p in ["t-aug-", "d-aug-", "aug-"]:
        if b.startswith(p):
            aug_level = p[:-1]
            core = b[len(p):]
            break

    if core.startswith("cc-pwcv"):
        family = "cc-pwCVXZ"
    elif core.startswith("cc-pcv"):
        family = "cc-pCVXZ"
    elif core.startswith("cc-pv"):
        family = "cc-pVXZ"
    else:
        family = "other"

    zeta = None
    for z in ["dz", "tz", "qz", "5z", "6z"]:
        if core.endswith(z):
            zeta = z.upper()
            break

    return family, aug_level, zeta


# ============================================================
# PySCF compatibility patch
# ============================================================

def patch_pyscf_einsum_to_numpy():
    """
    Workaround for some PySCF builds where pyscf.lib.einsum can fail in UCCSD
    for contractions with more than two operands, e.g.
        lib.einsum('ia,jb,iajb', ...)

    numpy.einsum is slower but robust for the small atoms considered here.
    """
    lib.einsum = np.einsum
    try:
        import pyscf.lib.numpy_helper as numpy_helper
        numpy_helper.einsum = np.einsum
    except Exception:
        pass


# ============================================================
# Size utilities
# ============================================================

def comb_safe(n, k):
    n = int(n)
    k = int(k)
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def estimate_fci_dim(norb, nelec_tuple):
    nalpha, nbeta = nelec_tuple
    return comb_safe(norb, nalpha) * comb_safe(norb, nbeta)


def inspect_mol(mol):
    nbas = mol.nbas
    nao = mol.nao_nr()
    lmax = max(mol.bas_angular(i) for i in range(nbas)) if nbas > 0 else None

    shell_count = {}
    for i in range(nbas):
        l = mol.bas_angular(i)
        shell_count[l] = shell_count.get(l, 0) + 1

    return nao, nbas, lmax, shell_count


def get_orbital_counts(mf, mol):
    """
    Return alpha/beta MO, occupied, and virtual counts.

    For UHF:
      mo_coeff = (Ca, Cb)
      mo_occ   = (occ_a, occ_b)

    For RHF/ROHF:
      one spatial MO basis is used for both spin sectors.
      nocc_alpha, nocc_beta are taken from mol.nelec.
    """
    nelec_a, nelec_b = mol.nelec

    if isinstance(mf.mo_coeff, (tuple, list)):
        nmo_a = int(mf.mo_coeff[0].shape[1])
        nmo_b = int(mf.mo_coeff[1].shape[1])

        occ_a = np.asarray(mf.mo_occ[0])
        occ_b = np.asarray(mf.mo_occ[1])

        nocc_a = int(np.count_nonzero(occ_a > 1e-8))
        nocc_b = int(np.count_nonzero(occ_b > 1e-8))
    else:
        nmo_a = int(mf.mo_coeff.shape[1])
        nmo_b = int(mf.mo_coeff.shape[1])
        nocc_a = int(nelec_a)
        nocc_b = int(nelec_b)

    nvir_a = nmo_a - nocc_a
    nvir_b = nmo_b - nocc_b

    return {
        "nmo_alpha": nmo_a,
        "nmo_beta": nmo_b,
        "nocc_alpha": nocc_a,
        "nocc_beta": nocc_b,
        "nvir_alpha": nvir_a,
        "nvir_beta": nvir_b,
        "nocc_spinorb": nocc_a + nocc_b,
        "nvir_spinorb": nvir_a + nvir_b,
        "is_unrestricted": isinstance(mf.mo_coeff, (tuple, list)),
    }


def excitation_space_sizes_antisym(counts):
    """
    Independent spin-resolved excitation-space sizes with same-spin
    antisymmetry. This is the main theoretical size metric for CISD/CCSD.
    """
    oa = counts["nocc_alpha"]
    ob = counts["nocc_beta"]
    va = counts["nvir_alpha"]
    vb = counts["nvir_beta"]

    # Singles
    s_a = oa * va
    s_b = ob * vb
    singles = s_a + s_b

    # Doubles: same-spin blocks use combinations, unlike-spin block is direct product.
    d_aa = comb_safe(oa, 2) * comb_safe(va, 2)
    d_ab = oa * ob * va * vb
    d_bb = comb_safe(ob, 2) * comb_safe(vb, 2)
    doubles = d_aa + d_ab + d_bb

    # Triples: same-spin subsets use combinations.
    t_aaa = comb_safe(oa, 3) * comb_safe(va, 3)
    t_aab = comb_safe(oa, 2) * ob * comb_safe(va, 2) * vb
    t_abb = oa * comb_safe(ob, 2) * va * comb_safe(vb, 2)
    t_bbb = comb_safe(ob, 3) * comb_safe(vb, 3)
    triples = t_aaa + t_aab + t_abb + t_bbb

    return {
        "singles_alpha_dim": int(s_a),
        "singles_beta_dim": int(s_b),
        "singles_dim": int(singles),

        "doubles_aa_dim_antisym": int(d_aa),
        "doubles_ab_dim": int(d_ab),
        "doubles_bb_dim_antisym": int(d_bb),
        "doubles_dim_antisym": int(doubles),

        "triples_aaa_dim_antisym": int(t_aaa),
        "triples_aab_dim_antisym": int(t_aab),
        "triples_abb_dim_antisym": int(t_abb),
        "triples_bbb_dim_antisym": int(t_bbb),
        "triples_dim_antisym": int(triples),
    }


def excitation_space_sizes_tensor_proxy(counts):
    """
    Naive tensor-storage proxy, without same-spin permutation compression.
    Useful as an implementation/storage proxy, not as the independent excitation
    dimension.
    """
    oa = counts["nocc_alpha"]
    ob = counts["nocc_beta"]
    va = counts["nvir_alpha"]
    vb = counts["nvir_beta"]

    t1_tensor = oa * va + ob * vb

    t2_tensor = (
        oa * oa * va * va
        + oa * ob * va * vb
        + ob * ob * vb * vb
    )

    nocc_so = oa + ob
    nvir_so = va + vb
    t3_tensor = nocc_so**3 * nvir_so**3

    return {
        "t1_size_tensor_proxy": int(t1_tensor),
        "t2_size_tensor_proxy": int(t2_tensor),
        "ccsd_amp_size_tensor_proxy": int(t1_tensor + t2_tensor),
        "t3_size_tensor_proxy": int(t3_tensor),
    }


def rhf_rohf_spatial_proxy(mf, mol, counts):
    """
    Optional spatial-orbital compressed proxy for RHF/ROHF-like references.
    This is closer to restricted/spin-adapted storage, but it is not the same
    as spin-resolved determinant excitation dimension.
    """
    if isinstance(mf.mo_coeff, (tuple, list)):
        return {}

    nmo = int(mf.mo_coeff.shape[1])
    # closed shell: nocc spatial = nelec/2; ROHF high-spin: use max alpha/beta occ
    nocc_spatial = int(max(counts["nocc_alpha"], counts["nocc_beta"]))
    nvir_spatial = nmo - nocc_spatial

    t1_spatial = nocc_spatial * nvir_spatial
    t2_spatial = nocc_spatial * nocc_spatial * nvir_spatial * nvir_spatial

    if mol.nelectron < 3:
        t3_spatial = 0
    else:
        t3_spatial = nocc_spatial**3 * nvir_spatial**3

    return {
        "spatial_proxy_nmo": int(nmo),
        "spatial_proxy_nocc": int(nocc_spatial),
        "spatial_proxy_nvir": int(nvir_spatial),
        "cc_t1_dim_spatial_proxy": int(t1_spatial),
        "cc_t2_dim_spatial_proxy": int(t2_spatial),
        "ccsd_amp_dim_spatial_proxy": int(t1_spatial + t2_spatial),
        "ccsd_t_triples_dim_spatial_proxy": int(t3_spatial),
        "ccsd_t_total_dim_spatial_proxy": int(t1_spatial + t2_spatial + t3_spatial),
    }


def get_cisd_size_info(mf, mol):
    counts = get_orbital_counts(mf, mol)
    ex = excitation_space_sizes_antisym(counts)
    tensor = excitation_space_sizes_tensor_proxy(counts)
    spatial = rhf_rohf_spatial_proxy(mf, mol, counts)

    cisd_dim = 1 + ex["singles_dim"] + ex["doubles_dim_antisym"]

    out = {
        "cisd_nmo_alpha": counts["nmo_alpha"],
        "cisd_nmo_beta": counts["nmo_beta"],
        "cisd_nocc_alpha": counts["nocc_alpha"],
        "cisd_nocc_beta": counts["nocc_beta"],
        "cisd_nvir_alpha": counts["nvir_alpha"],
        "cisd_nvir_beta": counts["nvir_beta"],
        "cisd_nocc_spinorb": counts["nocc_spinorb"],
        "cisd_nvir_spinorb": counts["nvir_spinorb"],

        "CISD_dim_ref": 1,
        "CISD_dim_singles": ex["singles_dim"],
        "CISD_dim_doubles_antisym": ex["doubles_dim_antisym"],
        "CISD_dim": int(cisd_dim),

        "CISD_dim_s_alpha": ex["singles_alpha_dim"],
        "CISD_dim_s_beta": ex["singles_beta_dim"],
        "CISD_dim_d_aa_antisym": ex["doubles_aa_dim_antisym"],
        "CISD_dim_d_ab": ex["doubles_ab_dim"],
        "CISD_dim_d_bb_antisym": ex["doubles_bb_dim_antisym"],

        "CISD_tensor_proxy": int(1 + tensor["t1_size_tensor_proxy"] + tensor["t2_size_tensor_proxy"]),
    }
    out.update({"cisd_" + k: v for k, v in spatial.items()})
    return out


def get_cc_size_info(mf, mol):
    counts = get_orbital_counts(mf, mol)
    ex = excitation_space_sizes_antisym(counts)
    tensor = excitation_space_sizes_tensor_proxy(counts)
    spatial = rhf_rohf_spatial_proxy(mf, mol, counts)

    ccsd_amp_dim = ex["singles_dim"] + ex["doubles_dim_antisym"]
    ccsdt_total_excitation_dim = ccsd_amp_dim + ex["triples_dim_antisym"]

    out = {
        "cc_nmo_alpha": counts["nmo_alpha"],
        "cc_nmo_beta": counts["nmo_beta"],
        "cc_nmo_total_spatial_sum": counts["nmo_alpha"] + counts["nmo_beta"],
        "cc_nocc_alpha": counts["nocc_alpha"],
        "cc_nocc_beta": counts["nocc_beta"],
        "cc_nvir_alpha": counts["nvir_alpha"],
        "cc_nvir_beta": counts["nvir_beta"],
        "cc_nocc_spinorb": counts["nocc_spinorb"],
        "cc_nvir_spinorb": counts["nvir_spinorb"],

        # Independent spin-resolved antisymmetrized CCSD amplitude dimensions
        "cc_t1_dim": ex["singles_dim"],
        "cc_t1_alpha_dim": ex["singles_alpha_dim"],
        "cc_t1_beta_dim": ex["singles_beta_dim"],
        "cc_t2_aa_dim_antisym": ex["doubles_aa_dim_antisym"],
        "cc_t2_ab_dim": ex["doubles_ab_dim"],
        "cc_t2_bb_dim_antisym": ex["doubles_bb_dim_antisym"],
        "cc_t2_dim_antisym": ex["doubles_dim_antisym"],
        "ccsd_amp_dim_antisym": int(ccsd_amp_dim),

        # Independent spin-resolved antisymmetrized triples dimensions for CCSD(T)
        "ccsd_t_triples_aaa_dim_antisym": ex["triples_aaa_dim_antisym"],
        "ccsd_t_triples_aab_dim_antisym": ex["triples_aab_dim_antisym"],
        "ccsd_t_triples_abb_dim_antisym": ex["triples_abb_dim_antisym"],
        "ccsd_t_triples_bbb_dim_antisym": ex["triples_bbb_dim_antisym"],
        "ccsd_t_triples_dim_antisym": ex["triples_dim_antisym"],
        "ccsd_t_total_excitation_dim_antisym": int(ccsdt_total_excitation_dim),

        # Tensor-storage proxies
        "cc_t1_size_tensor_proxy": tensor["t1_size_tensor_proxy"],
        "cc_t2_size_tensor_proxy": tensor["t2_size_tensor_proxy"],
        "ccsd_amp_size_tensor_proxy": tensor["ccsd_amp_size_tensor_proxy"],
        "ccsd_t_t3_size_tensor_proxy": tensor["t3_size_tensor_proxy"],
    }
    out.update(spatial)
    return out


# ============================================================
# PySCF wrappers
# ============================================================

def build_mol(system, basis_name, max_memory_mb):
    mol = gto.Mole()
    mol.atom = system["atom"]
    mol.basis = basis_name
    mol.charge = system["charge"]
    mol.spin = system["spin"]
    mol.unit = system["unit"]
    mol.verbose = 0

    if max_memory_mb and max_memory_mb > 0:
        mol.max_memory = max_memory_mb

    mol.build()
    return mol


def make_scf(mol, kind, conv_tol):
    kind = kind.upper()

    if kind == "RHF":
        mf = scf.RHF(mol)
    elif kind == "ROHF":
        mf = scf.ROHF(mol)
    elif kind == "UHF":
        mf = scf.UHF(mol)
    else:
        raise ValueError(f"Unknown SCF kind: {kind}")

    mf.conv_tol = conv_tol
    mf.max_cycle = 200
    mf.verbose = 0

    # Useful for diffuse/large bases; usually harmless for these atoms.
    try:
        mf = scf.addons.remove_linear_dep_(mf, threshold=1e-10)
    except Exception:
        pass

    return mf


def run_scf(mol, kind, conv_tol):
    mf = make_scf(mol, kind, conv_tol)

    t0 = time.time()
    e = mf.kernel()
    wall = time.time() - t0

    s2 = None
    mult = None
    try:
        ss = mf.spin_square()
        if isinstance(ss, tuple):
            s2, mult = ss
    except Exception:
        pass

    return mf, float(e), wall, s2, mult, bool(mf.converged)


def run_fci(mol, mf, max_fci_dim, run_fci_flag):
    t0 = time.time()

    if isinstance(mf.mo_coeff, (tuple, list)):
        return {
            "E_FCI": None,
            "wall_FCI_s": 0.0,
            "FCI_status": "skipped_tuple_mo_coeff",
            "err_msg_FCI": "FCI here expects RHF/ROHF spatial MO coefficients.",
        }

    norb = int(mf.mo_coeff.shape[1])
    nelec_tuple = mol.nelec
    fci_dim = estimate_fci_dim(norb, nelec_tuple)

    out = {
        "fci_norb": norb,
        "fci_nalpha": int(nelec_tuple[0]),
        "fci_nbeta": int(nelec_tuple[1]),
        "FCI_dim": int(fci_dim),
    }

    if not run_fci_flag:
        out.update({
            "E_FCI": None,
            "wall_FCI_s": time.time() - t0,
            "FCI_status": "dimension_only",
            "err_msg_FCI": None,
        })
        return out

    if max_fci_dim and max_fci_dim > 0 and fci_dim > max_fci_dim:
        out.update({
            "E_FCI": None,
            "wall_FCI_s": time.time() - t0,
            "FCI_status": "skipped_dim_cutoff",
            "err_msg_FCI": f"FCI_dim={fci_dim} exceeds max_fci_dim={max_fci_dim}",
        })
        return out

    try:
        cisolver = fci.FCI(mol, mf.mo_coeff)
        cisolver.conv_tol = 1e-12
        cisolver.max_cycle = 300
        cisolver.verbose = 0

        # Do not use fci.addons.fix_spin_ by default. It may call spin_op paths
        # that are limited for norb >= 64 in some PySCF versions.
        e_fci, fcivec = cisolver.kernel()

        out.update({
            "E_FCI": float(e_fci),
            "wall_FCI_s": time.time() - t0,
            "FCI_status": "ok",
            "err_msg_FCI": None,
        })
        return out

    except Exception:
        out.update({
            "E_FCI": None,
            "wall_FCI_s": time.time() - t0,
            "FCI_status": "failed",
            "err_msg_FCI": traceback.format_exc(),
        })
        return out


def run_cisd(mol, mf, max_cisd_dim, run_cisd_flag):
    t0 = time.time()

    size_info = get_cisd_size_info(mf, mol)
    out = dict(size_info)

    cisd_dim = size_info["CISD_dim"]

    if not run_cisd_flag:
        out.update({
            "E_CISD": None,
            "E_CISD_corr": None,
            "wall_CISD_s": time.time() - t0,
            "CISD_status": "dimension_only",
            "CISD_converged": None,
            "err_msg_CISD": None,
        })
        return out

    if max_cisd_dim and max_cisd_dim > 0 and cisd_dim > max_cisd_dim:
        out.update({
            "E_CISD": None,
            "E_CISD_corr": None,
            "wall_CISD_s": time.time() - t0,
            "CISD_status": "skipped_dim_cutoff",
            "CISD_converged": None,
            "err_msg_CISD": f"CISD_dim={cisd_dim} exceeds max_cisd_dim={max_cisd_dim}",
        })
        return out

    try:
        myci = ci.CISD(mf)
        myci.conv_tol = 1e-10
        myci.max_cycle = 200
        myci.verbose = 0

        kernel_out = myci.kernel()

        e_corr = None
        if isinstance(kernel_out, tuple) and len(kernel_out) >= 1:
            try:
                e_corr = float(kernel_out[0])
            except Exception:
                e_corr = None

        e_tot = getattr(myci, "e_tot", None)
        if e_tot is None:
            if e_corr is not None:
                e_tot = float(mf.e_tot + e_corr)
            else:
                raise RuntimeError("Cannot determine CISD total energy.")

        out.update({
            "E_CISD": float(e_tot),
            "E_CISD_corr": None if e_corr is None else float(e_corr),
            "wall_CISD_s": time.time() - t0,
            "CISD_status": "ok",
            "CISD_converged": bool(getattr(myci, "converged", True)),
            "err_msg_CISD": None,
        })
        return out

    except Exception:
        out.update({
            "E_CISD": None,
            "E_CISD_corr": None,
            "wall_CISD_s": time.time() - t0,
            "CISD_status": "failed",
            "CISD_converged": None,
            "err_msg_CISD": traceback.format_exc(),
        })
        return out


def run_ccsd_t(mol, mf, max_cc_nmo, run_ccsd_t_flag, retry_patch=True):
    t0 = time.time()

    size_info = get_cc_size_info(mf, mol)
    nmo_max = max(size_info["cc_nmo_alpha"], size_info["cc_nmo_beta"])

    out = dict(size_info)

    if not run_ccsd_t_flag:
        out.update({
            "E_CCSD": None,
            "E_CCSD_corr": None,
            "E_T": None,
            "E_CCSD_T": None,
            "wall_CCSD_T_s": time.time() - t0,
            "CCSD_status": "size_only",
            "CCSD_converged": None,
            "err_msg_CCSD": None,
            "err_msg_T": None,
        })
        return out

    if max_cc_nmo and max_cc_nmo > 0 and nmo_max > max_cc_nmo:
        out.update({
            "E_CCSD": None,
            "E_CCSD_corr": None,
            "E_T": None,
            "E_CCSD_T": None,
            "wall_CCSD_T_s": time.time() - t0,
            "CCSD_status": "skipped_nmo_cutoff",
            "CCSD_converged": None,
            "err_msg_CCSD": f"nmo_max={nmo_max} exceeds max_cc_nmo={max_cc_nmo}",
            "err_msg_T": None,
        })
        return out

    def _run_once():
        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1e-10
        mycc.max_cycle = 100
        mycc.verbose = 0

        ecc, t1, t2 = mycc.kernel()
        e_ccsd = mycc.e_tot

        e_t = None
        e_ccsd_t = None
        err_t = None
        try:
            e_t = mycc.ccsd_t()
            e_ccsd_t = e_ccsd + e_t
        except Exception:
            err_t = traceback.format_exc()

        return mycc, ecc, e_ccsd, e_t, e_ccsd_t, err_t

    try:
        try:
            mycc, ecc, e_ccsd, e_t, e_ccsd_t, err_t = _run_once()
        except ValueError as e:
            # Extra safety: if the einsum bug appears despite not patching at startup,
            # patch and retry once.
            if retry_patch and "not enough values to unpack" in str(e):
                patch_pyscf_einsum_to_numpy()
                mycc, ecc, e_ccsd, e_t, e_ccsd_t, err_t = _run_once()
            else:
                raise

        out.update({
            "E_CCSD": float(e_ccsd),
            "E_CCSD_corr": float(ecc),
            "E_T": None if e_t is None else float(e_t),
            "E_CCSD_T": None if e_ccsd_t is None else float(e_ccsd_t),
            "wall_CCSD_T_s": time.time() - t0,
            "CCSD_status": "ok",
            "CCSD_converged": bool(mycc.converged),
            "err_msg_CCSD": None,
            "err_msg_T": err_t,
        })
        return out

    except Exception:
        out.update({
            "E_CCSD": None,
            "E_CCSD_corr": None,
            "E_T": None,
            "E_CCSD_T": None,
            "wall_CCSD_T_s": time.time() - t0,
            "CCSD_status": "failed",
            "CCSD_converged": None,
            "err_msg_CCSD": traceback.format_exc(),
            "err_msg_T": None,
        })
        return out


# ============================================================
# Special cases
# ============================================================

def run_two_electron_highspin_cisd_from_fci(mol, mf_ref, row):
    t0 = time.time()

    size_info = get_cisd_size_info(mf_ref, mol)
    out = dict(size_info)

    e_fci = row.get("E_FCI", None)
    e_ref = row.get("E_SCF_FCI_REF", None)

    if e_fci is None:
        out.update({
            "E_CISD": None,
            "E_CISD_corr": None,
            "wall_CISD_s": time.time() - t0,
            "CISD_status": "skipped_need_fci",
            "CISD_converged": None,
            "err_msg_CISD": "He triplet special CISD uses FCI equivalence, but E_FCI is unavailable.",
        })
        return out

    out.update({
        "E_CISD": float(e_fci),
        "E_CISD_corr": None if e_ref is None else float(e_fci - e_ref),
        "wall_CISD_s": time.time() - t0,
        "CISD_status": "fci_equivalent_two_electron_highspin",
        "CISD_converged": True,
        "err_msg_CISD": None,
    })

    return out


def run_two_electron_highspin_cc_from_fci(mol, mf_ref, row):
    t0 = time.time()

    size_info = get_cc_size_info(mf_ref, mol)
    out = dict(size_info)

    e_fci = row.get("E_FCI", None)
    e_ref = row.get("E_SCF_FCI_REF", None)

    if e_fci is None:
        out.update({
            "E_CCSD": None,
            "E_CCSD_corr": None,
            "E_T": None,
            "E_CCSD_T": None,
            "wall_CCSD_T_s": time.time() - t0,
            "CCSD_status": "skipped_need_fci",
            "CCSD_converged": None,
            "err_msg_CCSD": "He triplet special CCSD uses FCI equivalence, but E_FCI is unavailable.",
            "err_msg_T": None,
        })
        return out

    out.update({
        "E_CCSD": float(e_fci),
        "E_CCSD_corr": None if e_ref is None else float(e_fci - e_ref),
        "E_T": 0.0,
        "E_CCSD_T": float(e_fci),
        "wall_CCSD_T_s": time.time() - t0,
        "CCSD_status": "fci_equivalent_two_electron_highspin",
        "CCSD_converged": True,
        "err_msg_CCSD": None,
        "err_msg_T": None,
    })

    return out


# ============================================================
# One benchmark task
# ============================================================

def run_one(system_name, basis_name, args):
    system = SYSTEMS[system_name]
    family, aug_level, zeta = basis_family_label(basis_name)

    print("\n" + "=" * 122)
    print(f"System = {system_name:12s} | Basis = {basis_name:18s}")
    print(f"Family = {family}, aug = {aug_level}, zeta = {zeta}")
    print("=" * 122)

    row = {
        "system": system_name,
        "description": system["description"],
        "basis": basis_name,
        "basis_family": family,
        "aug_level": aug_level,
        "zeta": zeta,
        "charge": system["charge"],
        "spin": system["spin"],
        "status": "ok",
    }

    try:
        mol = build_mol(system, basis_name, args.max_memory_mb)
    except Exception:
        print("  Build failed. Skip this basis.")
        row["status"] = "build_failed"
        row["err_msg_build"] = traceback.format_exc()
        return row

    nao, nbas, lmax, shell_count = inspect_mol(mol)

    row.update({
        "nelec": int(mol.nelectron),
        "nelec_alpha": int(mol.nelec[0]),
        "nelec_beta": int(mol.nelec[1]),

        "basis_nao": int(nao),
        "basis_nbas": int(nbas),
        "basis_lmax": None if lmax is None else int(lmax),
        "basis_shell_count": str(shell_count),

        "nao": int(nao),
        "nbas": int(nbas),
        "lmax": None if lmax is None else int(lmax),
        "shell_count": str(shell_count),
    })

    print(f"  nelec = {mol.nelectron}, alpha/beta = {mol.nelec}")
    print(f"  AO basis: nao = {nao}, nbas = {nbas}, lmax = {lmax}, shells = {shell_count}")

    # ------------------------------------------------------------
    # SCF for FCI
    # ------------------------------------------------------------
    try:
        mf_fci, e_scf_fci, wall_scf_fci, s2_fci, mult_fci, conv_fci = run_scf(
            mol, system["scf_fci"], args.scf_conv_tol
        )

        row.update({
            "SCF_FCI_REF": system["scf_fci"],
            "E_SCF_FCI_REF": e_scf_fci,
            "wall_SCF_FCI_REF_s": wall_scf_fci,
            "S2_SCF_FCI_REF": s2_fci,
            "mult_SCF_FCI_REF": mult_fci,
            "SCF_FCI_converged": conv_fci,
        })

        print(
            f"  {system['scf_fci']:<4s} for FCI: "
            f"E = {e_scf_fci: .15f} | conv = {conv_fci} | wall = {wall_scf_fci:.2f}s"
        )
        if s2_fci is not None:
            print(f"       <S^2> = {s2_fci:.8f}, 2S+1 = {mult_fci:.8f}")

    except Exception:
        print("  SCF for FCI failed.")
        row["status"] = "scf_fci_failed"
        row["err_msg_scf_fci"] = traceback.format_exc()
        return row

    # ------------------------------------------------------------
    # FCI
    # ------------------------------------------------------------
    fci_res = run_fci(
        mol,
        mf_fci,
        max_fci_dim=args.max_fci_dim,
        run_fci_flag=(not args.no_fci),
    )
    row.update(fci_res)

    print(
        f"  FCI size: norb = {row.get('fci_norb')}, "
        f"nalpha = {row.get('fci_nalpha')}, nbeta = {row.get('fci_nbeta')}, "
        f"FCI_dim = {row.get('FCI_dim')}"
    )

    if row.get("E_FCI") is not None:
        print(f"  FCI: E = {row['E_FCI']: .15f} | wall = {row['wall_FCI_s']:.2f}s")
    else:
        print(f"  FCI: {row.get('FCI_status')} | {row.get('err_msg_FCI')}")

    # ------------------------------------------------------------
    # Special case: He triplet
    # ------------------------------------------------------------
    if system.get("cc_special", None) == "two_electron_highspin_fci_equivalent":
        print("  Special treatment: two-electron high-spin sector.")
        print("  Use finite-basis identities: CISD = CCSD = FCI, and (T) = 0.")

        row.update({
            "SCF_CISD_REF": system["scf_fci"],
            "E_SCF_CISD_REF": row.get("E_SCF_FCI_REF"),
            "wall_SCF_CISD_REF_s": row.get("wall_SCF_FCI_REF_s"),
            "S2_SCF_CISD_REF": row.get("S2_SCF_FCI_REF"),
            "mult_SCF_CISD_REF": row.get("mult_SCF_FCI_REF"),
            "SCF_CISD_converged": row.get("SCF_FCI_converged"),

            "SCF_CC_REF": system["scf_fci"],
            "E_SCF_CC_REF": row.get("E_SCF_FCI_REF"),
            "wall_SCF_CC_REF_s": row.get("wall_SCF_FCI_REF_s"),
            "S2_SCF_CC_REF": row.get("S2_SCF_FCI_REF"),
            "mult_SCF_CC_REF": row.get("mult_SCF_FCI_REF"),
            "SCF_CC_converged": row.get("SCF_FCI_converged"),
        })

        cisd_res = run_two_electron_highspin_cisd_from_fci(mol, mf_fci, row)
        row.update(cisd_res)

        cc_res = run_two_electron_highspin_cc_from_fci(mol, mf_fci, row)
        row.update(cc_res)

        print(
            "  CISD independent antisym size: "
            f"CISD_dim = {row.get('CISD_dim')} "
            f"= 1 + singles({row.get('CISD_dim_singles')}) "
            f"+ doubles({row.get('CISD_dim_doubles_antisym')})"
        )
        print(f"  CISD: E = {row.get('E_CISD')} | status = {row.get('CISD_status')}")

        print(
            "  CCSD independent antisym size: "
            f"T1 = {row.get('cc_t1_dim')}, "
            f"T2 = {row.get('cc_t2_dim_antisym')}, "
            f"CCSD_amp = {row.get('ccsd_amp_dim_antisym')}"
        )
        print(
            "  CCSD(T) independent triples size: "
            f"T3 = {row.get('ccsd_t_triples_dim_antisym')}, "
            f"CCSD+T3 = {row.get('ccsd_t_total_excitation_dim_antisym')}"
        )
        print(
            "  Tensor proxy: "
            f"CCSD_amp_tensor = {row.get('ccsd_amp_size_tensor_proxy')}, "
            f"T3_tensor = {row.get('ccsd_t_t3_size_tensor_proxy')}"
        )
        print(f"  CCSD:    E = {row.get('E_CCSD')} | status = {row.get('CCSD_status')}")
        print(f"  (T):     E_T = {row.get('E_T')}")
        print(f"  CCSD(T): E = {row.get('E_CCSD_T')}")

        return row

    # ------------------------------------------------------------
    # CISD for normal systems
    # ------------------------------------------------------------
    try:
        mf_cisd, e_scf_cisd, wall_scf_cisd, s2_cisd, mult_cisd, conv_cisd = run_scf(
            mol, system["scf_cisd"], args.scf_conv_tol
        )

        row.update({
            "SCF_CISD_REF": system["scf_cisd"],
            "E_SCF_CISD_REF": e_scf_cisd,
            "wall_SCF_CISD_REF_s": wall_scf_cisd,
            "S2_SCF_CISD_REF": s2_cisd,
            "mult_SCF_CISD_REF": mult_cisd,
            "SCF_CISD_converged": conv_cisd,
        })

        print(
            f"  {system['scf_cisd']:<4s} for CISD: "
            f"E = {e_scf_cisd: .15f} | conv = {conv_cisd} | wall = {wall_scf_cisd:.2f}s"
        )
        if s2_cisd is not None:
            print(f"       <S^2> = {s2_cisd:.8f}, 2S+1 = {mult_cisd:.8f}")

    except Exception:
        print("  SCF for CISD failed.")
        row["status"] = "scf_cisd_failed"
        row["err_msg_scf_cisd"] = traceback.format_exc()
        return row

    cisd_res = run_cisd(
        mol,
        mf_cisd,
        max_cisd_dim=args.max_cisd_dim,
        run_cisd_flag=(not args.no_cisd),
    )
    row.update(cisd_res)

    print(
        "  CISD independent antisym size: "
        f"CISD_dim = {row.get('CISD_dim')} "
        f"= 1 + singles({row.get('CISD_dim_singles')}) "
        f"+ doubles({row.get('CISD_dim_doubles_antisym')})"
    )
    print(
        "       blocks: "
        f"S_a={row.get('CISD_dim_s_alpha')}, "
        f"S_b={row.get('CISD_dim_s_beta')}, "
        f"D_aa={row.get('CISD_dim_d_aa_antisym')}, "
        f"D_ab={row.get('CISD_dim_d_ab')}, "
        f"D_bb={row.get('CISD_dim_d_bb_antisym')}"
    )

    if row.get("E_CISD") is not None:
        print(f"  CISD: E = {row['E_CISD']: .15f} | wall = {row['wall_CISD_s']:.2f}s")
    else:
        print(f"  CISD: {row.get('CISD_status')} | {row.get('err_msg_CISD')}")

    # ------------------------------------------------------------
    # CCSD / CCSD(T)
    # ------------------------------------------------------------
    try:
        mf_cc, e_scf_cc, wall_scf_cc, s2_cc, mult_cc, conv_cc = run_scf(
            mol, system["scf_cc"], args.scf_conv_tol
        )

        row.update({
            "SCF_CC_REF": system["scf_cc"],
            "E_SCF_CC_REF": e_scf_cc,
            "wall_SCF_CC_REF_s": wall_scf_cc,
            "S2_SCF_CC_REF": s2_cc,
            "mult_SCF_CC_REF": mult_cc,
            "SCF_CC_converged": conv_cc,
        })

        print(
            f"  {system['scf_cc']:<4s} for CC : "
            f"E = {e_scf_cc: .15f} | conv = {conv_cc} | wall = {wall_scf_cc:.2f}s"
        )
        if s2_cc is not None:
            print(f"       <S^2> = {s2_cc:.8f}, 2S+1 = {mult_cc:.8f}")

    except Exception:
        print("  SCF for CC failed.")
        row["status"] = "scf_cc_failed"
        row["err_msg_scf_cc"] = traceback.format_exc()
        return row

    cc_res = run_ccsd_t(
        mol,
        mf_cc,
        max_cc_nmo=args.max_cc_nmo,
        run_ccsd_t_flag=(not args.no_ccsdt),
        retry_patch=True,
    )
    row.update(cc_res)

    print(
        "  CCSD independent antisym size: "
        f"T1 = {row.get('cc_t1_dim')}, "
        f"T2 = {row.get('cc_t2_dim_antisym')}, "
        f"CCSD_amp = {row.get('ccsd_amp_dim_antisym')}"
    )
    print(
        "       T2 blocks: "
        f"aa={row.get('cc_t2_aa_dim_antisym')}, "
        f"ab={row.get('cc_t2_ab_dim')}, "
        f"bb={row.get('cc_t2_bb_dim_antisym')}"
    )
    print(
        "  CCSD(T) independent triples size: "
        f"T3 = {row.get('ccsd_t_triples_dim_antisym')}, "
        f"CCSD+T3 = {row.get('ccsd_t_total_excitation_dim_antisym')}"
    )
    print(
        "       T3 blocks: "
        f"aaa={row.get('ccsd_t_triples_aaa_dim_antisym')}, "
        f"aab={row.get('ccsd_t_triples_aab_dim_antisym')}, "
        f"abb={row.get('ccsd_t_triples_abb_dim_antisym')}, "
        f"bbb={row.get('ccsd_t_triples_bbb_dim_antisym')}"
    )
    if row.get("ccsd_amp_dim_spatial_proxy") is not None:
        print(
            "  Spatial compressed proxy: "
            f"CCSD_amp_spatial = {row.get('ccsd_amp_dim_spatial_proxy')}, "
            f"T3_spatial = {row.get('ccsd_t_triples_dim_spatial_proxy')}"
        )
    print(
        "  Tensor proxy: "
        f"CCSD_amp_tensor = {row.get('ccsd_amp_size_tensor_proxy')}, "
        f"T3_tensor = {row.get('ccsd_t_t3_size_tensor_proxy')}"
    )

    if row.get("E_CCSD") is not None:
        print(f"  CCSD:    E = {row['E_CCSD']: .15f} | conv = {row['CCSD_converged']}")
        if row.get("E_CCSD_T") is not None:
            print(
                f"  (T):     E_T = {row['E_T']: .15e}\n"
                f"  CCSD(T): E = {row['E_CCSD_T']: .15f} | wall = {row['wall_CCSD_T_s']:.2f}s"
            )
        else:
            print(f"  CCSD(T): triples failed/skipped | {row.get('err_msg_T')}")
    else:
        print(f"  CCSD: {row.get('CCSD_status')} | {row.get('err_msg_CCSD')}")

    return row


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system",
        type=str,
        default="all",
        choices=["He_singlet", "He_triplet", "Li_doublet", "Be_singlet", "all"],
    )

    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-memory-mb", type=int, default=0)

    # 0 means unlimited
    parser.add_argument("--max-fci-dim", type=int, default=0)
    parser.add_argument("--max-cisd-dim", type=int, default=0)
    parser.add_argument("--max-cc-nmo", type=int, default=0)

    parser.add_argument("--no-fci", action="store_true")
    parser.add_argument("--no-cisd", action="store_true")
    parser.add_argument("--no-ccsdt", action="store_true")

    parser.add_argument(
        "--no-einsum-patch",
        action="store_true",
        help="Disable numpy.einsum patch for PySCF UCCSD. Default: patch is enabled.",
    )

    parser.add_argument("--scf-conv-tol", type=float, default=1e-12)

    parser.add_argument(
        "--output",
        type=str,
        default="atoms_dunning_classified_benchmark_v6.csv",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    lib.num_threads(args.threads)

    if not args.no_einsum_patch:
        patch_pyscf_einsum_to_numpy()
        einsum_patch_status = "enabled: pyscf.lib.einsum -> numpy.einsum"
    else:
        einsum_patch_status = "disabled"

    print("=" * 122)
    print("PySCF classified Dunning benchmark v6")
    print(f"threads = {args.threads}")
    print(f"max_memory_mb = {args.max_memory_mb}  (0 means PySCF default)")
    print(f"max_fci_dim = {args.max_fci_dim}  (0 means unlimited)")
    print(f"max_cisd_dim = {args.max_cisd_dim}  (0 means unlimited)")
    print(f"max_cc_nmo = {args.max_cc_nmo}  (0 means unlimited)")
    print(f"run_fci = {not args.no_fci}, run_cisd = {not args.no_cisd}, run_ccsd_t = {not args.no_ccsdt}")
    print(f"einsum patch = {einsum_patch_status}")
    print("Size metrics include same-spin antisymmetrized excitation dimensions.")
    print("He_triplet special case: CISD = CCSD = FCI, (T) = 0.")
    print("=" * 122)

    if args.system == "all":
        systems_to_run = ["Be_singlet"]
    else:
        systems_to_run = [args.system]

    rows = []
    t_all = time.time()

    partial_file = args.output.replace(".csv", "_partial.csv")

    for system_name in systems_to_run:
        basis_list = make_basis_list_for_system(system_name)

        print("\n" + "#" * 122)
        print(f"Basis list for {system_name}:")
        for b in basis_list:
            print(" ", b)
        print("#" * 122)

        for basis_name in basis_list:
            row = run_one(system_name, basis_name, args)
            rows.append(row)
            pd.DataFrame(rows).to_csv(partial_file, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print("\n" + "=" * 122)
    print("Compact summary")
    print("=" * 122)

    summary_cols = [
        "system", "basis", "status",
        "basis_nao", "basis_nbas", "basis_lmax",

        "fci_norb", "fci_nalpha", "fci_nbeta", "FCI_dim", "FCI_status",
        "E_FCI", "wall_FCI_s",

        "cisd_nmo_alpha", "cisd_nmo_beta",
        "cisd_nocc_alpha", "cisd_nocc_beta",
        "cisd_nvir_alpha", "cisd_nvir_beta",
        "CISD_dim", "CISD_dim_singles", "CISD_dim_doubles_antisym",
        "CISD_dim_d_aa_antisym", "CISD_dim_d_ab", "CISD_dim_d_bb_antisym",
        "CISD_status", "E_CISD", "wall_CISD_s",

        "SCF_CC_REF",
        "cc_nmo_alpha", "cc_nmo_beta",
        "cc_nocc_alpha", "cc_nocc_beta",
        "cc_nvir_alpha", "cc_nvir_beta",

        "cc_t1_dim",
        "cc_t2_dim_antisym",
        "cc_t2_aa_dim_antisym",
        "cc_t2_ab_dim",
        "cc_t2_bb_dim_antisym",
        "ccsd_amp_dim_antisym",

        "ccsd_t_triples_dim_antisym",
        "ccsd_t_triples_aaa_dim_antisym",
        "ccsd_t_triples_aab_dim_antisym",
        "ccsd_t_triples_abb_dim_antisym",
        "ccsd_t_triples_bbb_dim_antisym",
        "ccsd_t_total_excitation_dim_antisym",

        "ccsd_amp_dim_spatial_proxy",
        "ccsd_t_triples_dim_spatial_proxy",
        "ccsd_amp_size_tensor_proxy",
        "ccsd_t_t3_size_tensor_proxy",

        "CCSD_status", "E_CCSD", "E_T", "E_CCSD_T", "wall_CCSD_T_s",
    ]

    summary_cols = [c for c in summary_cols if c in df.columns]
    print(df[summary_cols].to_string(index=False))

    print(f"\nSaved full CSV:    {args.output}")
    print(f"Saved partial CSV: {partial_file}")
    print(f"Total wall time = {time.time() - t_all:.2f}s")


if __name__ == "__main__":
    main()
