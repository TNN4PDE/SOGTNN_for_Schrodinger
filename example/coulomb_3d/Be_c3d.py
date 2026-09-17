import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import copy
import itertools
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from scipy.io import loadmat
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from sogtnn.quadrature import *
from sogtnn.integration import *
from sogtnn.tnn import *
torch.set_printoptions(precision=16)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
pi = 3.14159265358979323846
# ********** choose data type and device **********
dtype = torch.double
device = 'cuda:0'
# ********** generate data points **********
a_x, b_x = -1.0, 1.0
a_y, b_y = -1.0, 1.0
a_z, b_z = -1.0, 1.0
scale = 12.0
Z = 4.0  
# number of quad points
quad = [10, 6, 10]
N = [8, 25, 8]
ratio = [1, 2, 1]
# quad points and quad weights.
point_x, w_x = composite_quadrature_custom_diff_points(quad, a_x, b_x, ratio, N, device=device, dtype=dtype)
point_y, w_y = composite_quadrature_custom_diff_points(quad, a_y, b_y, ratio, N, device=device, dtype=dtype)
point_z, w_z = composite_quadrature_custom_diff_points(quad, a_z, b_z, ratio, N, device=device, dtype=dtype)
point_short, w_short = composite_quadrature_custom_diff_points(quad, a_x, b_x, ratio, N, device=device, dtype=dtype)
N_x, N_y, N_z = len(point_x), len(point_y), len(point_z)
X, Y = torch.meshgrid(point_x, point_x, indexing='ij')
# SOG and range-spiltting parameters
def _load_mat_first(path, keys):
    mat = loadmat(path)
    for k in keys:
        if k in mat:
            return mat[k]
    raise KeyError(f'None of the keys {keys} found in {path}; available keys: {list(mat.keys())}')
SQRT_2PI = torch.sqrt(torch.tensor(2.0 * pi, dtype=dtype, device=device))
# rank and truncation for 1/r
truncation_number_r_positive = 80
truncation_number_r_negative = 24
truncation_number_r = truncation_number_r_positive + truncation_number_r_negative + 1
truncation_number_r_s_from = -120
truncation_number_r_s_to = -25
truncation_number_s = truncation_number_r_s_to - truncation_number_r_s_from + 1
# rank and truncation for 1/rij
truncation_number_l1_from = 7
truncation_number_l1_to = 100
truncation_number_l1 = truncation_number_l1_to - truncation_number_l1_from + 1
truncation_number_l2_from = -1
truncation_number_l2_to = 6
truncation_number_l2 = truncation_number_l2_to - truncation_number_l2_from + 1
truncation_number_l3_from = -5
truncation_number_l3_to = -2
truncation_number_l3 = truncation_number_l3_to - truncation_number_l3_from + 1
truncation_number_lm1_from = -7
truncation_number_lm1_to = -6
p_lm1 = 42
truncation_number_lm2_from = -9
truncation_number_lm2_to = -8
p_lm2 = 80
truncation_number_lm3_from = -11
truncation_number_lm3_to = -10
p_lm3 = 150
truncation_number_ss_from = -130
truncation_number_ss_to = -12
truncation_number_ss = truncation_number_ss_to - truncation_number_ss_from + 1
# ********** SOG tensor generate **********
b0 = torch.tensor(1.4, dtype=dtype, device=device)
coeff_b0 = 2.0 * torch.log(b0) / SQRT_2PI
ewald = torch.zeros((truncation_number_r, N_x), dtype=dtype, device=device)
ewald1 = torch.zeros((truncation_number_r, N_x), dtype=dtype, device=device)
for l in range(1, truncation_number_r_negative + 1):
    idx = truncation_number_r_negative - l
    fac = b0 ** l
    expo = torch.exp(-0.5 * (fac ** 2) * (point_x ** 2))
    ewald[idx] = coeff_b0 * fac * expo
    ewald1[idx] = expo
for l in range(0, truncation_number_r_positive + 1):
    idx = truncation_number_r_negative + l
    fac = b0 ** l
    expo = torch.exp(-0.5 * (point_x ** 2) / (fac ** 2))
    ewald[idx] = coeff_b0 / fac * expo
    ewald1[idx] = expo
point0 = torch.zeros(1, dtype=dtype, device=device)
w_0 = torch.ones_like(point0)
point_sx = torch.zeros(truncation_number_s, dtype=dtype, device=device)
point_sy = torch.zeros(truncation_number_s, dtype=dtype, device=device)
for l in range(truncation_number_r_s_from, truncation_number_r_s_to + 1):
    idx = l - truncation_number_r_s_from
    point_sx[idx] = 2.0 * torch.log(b0)
    point_sy[idx] = SQRT_2PI * (b0 ** l)
b1 = torch.tensor(1.4, dtype=dtype, device=device)
coeff_b1 = 2.0 * torch.log(b1) / SQRT_2PI
max_cheb_degree = 50
cheb_poly_cache = torch.zeros((max_cheb_degree + 1, N_x), dtype=dtype, device=device)
cheb_poly_cache[0] = 1.0
cheb_poly_cache[1] = point_x
for i in range(2, max_cheb_degree + 1):
    cheb_poly_cache[i] = 2.0 * point_x * cheb_poly_cache[i - 1] - cheb_poly_cache[i - 2]
def load_kl_index(path, possible_keys):
    data = _load_mat_first(path, possible_keys)
    return torch.from_numpy(data).to(device=device, dtype=torch.int64).squeeze()
def build_alpha_cheb(l_from, l_to, num_of_cheb_expansion):
    alpha_chev1 = torch.zeros((l_to - l_from + 1, num_of_cheb_expansion), dtype=dtype, device=device)
    co_l = torch.zeros((l_to - l_from + 1,), dtype=dtype, device=device)
    for idx, l in enumerate(range(l_from, l_to + 1)):
        co_l[idx] = coeff_b1 / (b1 ** l)
        mat_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cheb_mat_data_Be', f'vectorE_{l}.mat'))
        alpha_chev1[idx] = torch.tensor(mat_data['E_vector'], device=device, dtype=dtype).t()
    alpha_chev = alpha_chev1 * co_l[:, None]
    return alpha_chev, alpha_chev1
kl_index_long_1 = load_kl_index(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cheb_mat_data_Be', 'kl_index_long_3.mat'), ['kl_index_long_2', 'kl_index_long_3', 'kl_index_long_1'])
kl_index_long_2 = load_kl_index(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cheb_mat_data_Be', 'kl_index_long_2.mat'), ['kl_index_long_2', 'kl_index_long_1', 'kl_index_long_3'])
kl_index_long_3 = load_kl_index(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cheb_mat_data_Be', 'kl_index_long_1.mat'), ['kl_index_long_1', 'kl_index_long_2', 'kl_index_long_3'])
alpha_l1_chev, alpha_l1_chev1 = build_alpha_cheb(truncation_number_l1_from, truncation_number_l1_to, kl_index_long_1.shape[0])
alpha_l2_chev, alpha_l2_chev1 = build_alpha_cheb(truncation_number_l2_from, truncation_number_l2_to, kl_index_long_2.shape[0])
alpha_l3_chev, alpha_l3_chev1 = build_alpha_cheb(truncation_number_l3_from, truncation_number_l3_to, kl_index_long_3.shape[0])
def build_middle_tensors(l_from, l_to, rank_p):
    tnum = l_to - l_from + 1
    psi_x0 = torch.zeros((tnum, rank_p, N_x), dtype=dtype, device=device)
    psi_x1 = torch.zeros((tnum, rank_p, N_x), dtype=dtype, device=device)
    psi_y0 = torch.zeros((tnum, rank_p, N_y), dtype=dtype, device=device)
    psi_y1 = torch.zeros((tnum, rank_p, N_y), dtype=dtype, device=device)
    alpha_x = torch.zeros((tnum, rank_p), dtype=dtype, device=device)
    alpha_y = torch.zeros((tnum, rank_p), dtype=dtype, device=device)
    for idx, l in enumerate(range(l_from, l_to + 1)):
        sv_vectors, sv_values = low_rank_svd_approximation(b1, l, rank_p, X, Y)
        psi_x0[idx] = sv_vectors[0]
        psi_x1[idx] = sv_vectors[1]
        psi_y0[idx] = sv_vectors[0]
        psi_y1[idx] = sv_vectors[1]
        alpha_x[idx] = coeff_b1 * (b1 ** (-l)) * sv_values
        alpha_y[idx] = sv_values
    return psi_x0, psi_x1, psi_y0, psi_y1, alpha_x, alpha_y
psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1, alpha_psi_m1_x, alpha_psi_m1_y = build_middle_tensors(truncation_number_lm1_from, truncation_number_lm1_to, p_lm1)
psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1, alpha_psi_m2_x, alpha_psi_m2_y = build_middle_tensors(truncation_number_lm2_from, truncation_number_lm2_to, p_lm2)
psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1, alpha_psi_m3_x, alpha_psi_m3_y = build_middle_tensors(truncation_number_lm3_from, truncation_number_lm3_to, p_lm3)
point_ssx = torch.zeros(truncation_number_ss, dtype=dtype, device=device)
point_ssy = torch.zeros(truncation_number_ss, dtype=dtype, device=device)
for l in range(truncation_number_ss_from, truncation_number_ss_to + 1):
    idx = l - truncation_number_ss_from
    point_ssx[idx] = 2.0 * torch.log(b1)
    point_ssy[idx] = SQRT_2PI * (b1 ** l)
# ********** Main Model **********
p = 60
sizes = [1, 50, 50, p]
activation = TNN_Sin
def build_orbital_triplet():
    mx = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
    my = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
    mz = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
    return mx, my, mz
model_x0, model_y0, model_z0 = build_orbital_triplet()
model_x1, model_y1, model_z1 = build_orbital_triplet()
model_x2, model_y2, model_z2 = build_orbital_triplet()
model_x3, model_y3, model_z3 = build_orbital_triplet()
# ********** Loss Function Core **********
def overlap_1d_li_style(w, a, b):
    # a,b: [1,p,n] -> [p,p]
    aw = a * w.view(1, 1, -1)
    return torch.prod(aw @ b.transpose(-1, -2), dim=0)
def overlap_1d_cheb_li_style(w, cheb, a, b):
    # a,b: [1,p,n] -> [k,p,p]
    aw_cheb = a.unsqueeze(0) * (w * cheb).view(cheb.shape[0], 1, 1, cheb.shape[1])
    res = aw_cheb @ b.unsqueeze(0).transpose(-1, -2)
    return torch.prod(res, dim=1)
def calc_V_element_li_style(phi_xL, phi_xR, phi_yL, phi_yR, phi_zL, phi_zR,phi_sxL, phi_sxR, phi_syL, phi_syR, phi_szL, phi_szR,w_x, w_y, w_z, w_0, ewald, ewald1, point_sx, point_sy):
    def op_long(w, v, L, R):
        vw = v * w.unsqueeze(0)
        L_vw = L.unsqueeze(0) * vw.view(vw.shape[0], 1, 1, vw.shape[1])
        res = L_vw @ R.unsqueeze(0).transpose(-1, -2)
        return torch.prod(res, dim=1)
    def op_short(w, v, L, R):
        vw = v * w
        L_vw = L.unsqueeze(0) * vw.view(-1, 1, 1, 1)
        res = L_vw @ R.unsqueeze(0).transpose(-1, -2)
        return torch.prod(res, dim=1)
    term_long = torch.sum(op_long(w_x, ewald, phi_xL, phi_xR)* op_long(w_y, ewald1, phi_yL, phi_yR)* op_long(w_z, ewald1, phi_zL, phi_zR),dim=0)
    term_short = torch.sum(op_short(w_0, point_sx, phi_sxL, phi_sxR)* op_short(w_0, point_sy, phi_syL, phi_syR)* op_short(w_0, point_sy, phi_szL, phi_szR),dim=0)
    out = term_long + term_short
    del term_long, term_short
    return out
def calc_rr_long_li_style(ix_ab, ix_cd, iy_ab, iy_cd, iz_ab, iz_cd,kl_index_long_1, kl_index_long_2, kl_index_long_3,alpha_l1_chev, alpha_l1_chev1, alpha_l2_chev, alpha_l2_chev1, alpha_l3_chev, alpha_l3_chev1):
    def one_block(kl_index, alpha_x, alpha_y, Xab, Xcd, Yab, Ycd, Zab, Zcd):
        ix_term = Xab[kl_index[:, 0]] * Xcd[kl_index[:, 1]]
        iy_term = Yab[kl_index[:, 0]] * Ycd[kl_index[:, 1]]
        iz_term = Zab[kl_index[:, 0]] * Zcd[kl_index[:, 1]]
        P, Q = ix_term.shape[1], ix_term.shape[2]
        term_x = (alpha_x @ ix_term.reshape(ix_term.shape[0], -1)).reshape(-1, P, Q)
        term_y = (alpha_y @ iy_term.reshape(iy_term.shape[0], -1)).reshape(-1, P, Q)
        term_z = (alpha_y @ iz_term.reshape(iz_term.shape[0], -1)).reshape(-1, P, Q)
        out = torch.sum(term_x * term_y * term_z, dim=0)
        del ix_term, iy_term, iz_term, term_x, term_y, term_z
        return out
    out = one_block(kl_index_long_1, alpha_l1_chev, alpha_l1_chev1, ix_ab, ix_cd, iy_ab, iy_cd, iz_ab, iz_cd)
    tmp = one_block(kl_index_long_2, alpha_l2_chev, alpha_l2_chev1, ix_ab, ix_cd, iy_ab, iy_cd, iz_ab, iz_cd)
    out = out + tmp
    del tmp
    tmp = one_block(kl_index_long_3, alpha_l3_chev, alpha_l3_chev1, ix_ab, ix_cd, iy_ab, iy_cd, iz_ab, iz_cd)
    out = out + tmp
    del tmp
    return out
def calc_middle_interaction_li_style(px_L1, px_R1, py_L1, py_R1, pz_L1, pz_R1,px_L2, px_R2, py_L2, py_R2, pz_L2, pz_R2,w_x, w_y, w_z,psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1, alpha_psi_m1_x, alpha_psi_m1_y,psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1, alpha_psi_m2_x, alpha_psi_m2_y,psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1, alpha_psi_m3_x, alpha_psi_m3_y):
    px_L1_0 = px_L1[0]
    px_R1_0 = px_R1[0]
    py_L1_0 = py_L1[0]
    py_R1_0 = py_R1[0]
    pz_L1_0 = pz_L1[0]
    pz_R1_0 = pz_R1[0]
    px_L2_0 = px_L2[0]
    px_R2_0 = px_R2[0]
    py_L2_0 = py_L2[0]
    py_R2_0 = py_R2[0]
    pz_L2_0 = pz_L2[0]
    pz_R2_0 = pz_R2[0]
    def calc_single_mode(alpha_x, alpha_y, psi_x0, psi_x1, psi_y0, psi_y1):
        weighted_x0 = (alpha_x[:, :, None] * psi_x0) * w_x[None, None, :]
        pure_x1 = psi_x1 * w_x[None, None, :]
        term_x = torch.sum(torch.einsum('trn,pn,qn->trpq', weighted_x0, px_L1_0, px_R1_0)* torch.einsum('trn,pn,qn->trpq', pure_x1, px_L2_0, px_R2_0),dim=1)
        weighted_y0 = (alpha_y[:, :, None] * psi_y0) * w_y[None, None, :]
        pure_y1 = psi_y1 * w_y[None, None, :]
        term_y = torch.sum(torch.einsum('trn,pn,qn->trpq', weighted_y0, py_L1_0, py_R1_0)* torch.einsum('trn,pn,qn->trpq', pure_y1, py_L2_0, py_R2_0),dim=1)
        weighted_z0 = (alpha_y[:, :, None] * psi_y0) * w_z[None, None, :]
        pure_z1 = psi_y1 * w_z[None, None, :]
        term_z = torch.sum(torch.einsum('trn,pn,qn->trpq', weighted_z0, pz_L1_0, pz_R1_0)* torch.einsum('trn,pn,qn->trpq', pure_z1, pz_L2_0, pz_R2_0),dim=1)
        out = torch.sum(term_x * term_y * term_z, dim=0)
        del weighted_x0, pure_x1, weighted_y0, pure_y1, weighted_z0, pure_z1
        del term_x, term_y, term_z
        return out
    out = calc_single_mode(alpha_psi_m1_x, alpha_psi_m1_y, psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1)
    tmp = calc_single_mode(alpha_psi_m2_x, alpha_psi_m2_y, psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1)
    out = out + tmp
    del tmp
    tmp = calc_single_mode(alpha_psi_m3_x, alpha_psi_m3_y, psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1)
    out = out + tmp
    del tmp
    return out
def calc_short_from_pairs_li_style(pair_x_A, pair_y_A, pair_z_A, pair_x_B, pair_y_B, pair_z_B, w_short, point_ssx, point_ssy):
    core_x = (pair_x_A * w_short.view(1, 1, -1)) @ pair_x_B.transpose(-1, -2)
    core_y = (pair_y_A * w_short.view(1, 1, -1)) @ pair_y_B.transpose(-1, -2)
    core_z = (pair_z_A * w_short.view(1, 1, -1)) @ pair_z_B.transpose(-1, -2)
    term_x = point_ssx.view(-1, 1, 1, 1) * core_x.unsqueeze(0)
    term_y = point_ssy.view(-1, 1, 1, 1) * core_y.unsqueeze(0)
    term_z = point_ssy.view(-1, 1, 1, 1) * core_z.unsqueeze(0)
    out = torch.sum(torch.prod(term_x * term_y * term_z, dim=1), dim=0)
    del core_x, core_y, core_z, term_x, term_y, term_z
    return out
def build_matrices_logic_be_singlegraph(
    phi_x0, phi_x1, phi_x2, phi_x3,
    phi_y0, phi_y1, phi_y2, phi_y3,
    phi_z0, phi_z1, phi_z2, phi_z3,
    grad_phi_x0, grad_phi_x1, grad_phi_x2, grad_phi_x3,
    grad_phi_y0, grad_phi_y1, grad_phi_y2, grad_phi_y3,
    grad_phi_z0, grad_phi_z1, grad_phi_z2, grad_phi_z3,
    phi_short_x0, phi_short_x1, phi_short_x2, phi_short_x3,
    phi_short_y0, phi_short_y1, phi_short_y2, phi_short_y3,
    phi_short_z0, phi_short_z1, phi_short_z2, phi_short_z3,
    phi_s_x0, phi_s_x1, phi_s_x2, phi_s_x3,
    phi_s_y0, phi_s_y1, phi_s_y2, phi_s_y3,
    phi_s_z0, phi_s_z1, phi_s_z2, phi_s_z3,
    w_x, w_y, w_z, w_short, w_0,
    ewald, ewald1, point_sx, point_sy, point_ssx, point_ssy,
    cheb_poly_cache, kl_index_long_1, kl_index_long_2, kl_index_long_3,
    alpha_l1_chev, alpha_l1_chev1, alpha_l2_chev, alpha_l2_chev1, alpha_l3_chev, alpha_l3_chev1,
    psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1, alpha_psi_m1_x, alpha_psi_m1_y,
    psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1, alpha_psi_m2_x, alpha_psi_m2_y,
    psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1, alpha_psi_m3_x, alpha_psi_m3_y,
    scale):
    ix00 = overlap_1d_li_style(w_x, phi_x0, phi_x0)
    ix11 = overlap_1d_li_style(w_x, phi_x1, phi_x1)
    ix22 = overlap_1d_li_style(w_x, phi_x2, phi_x2)
    ix33 = overlap_1d_li_style(w_x, phi_x3, phi_x3)
    ix01 = overlap_1d_li_style(w_x, phi_x0, phi_x1)
    ix23 = overlap_1d_li_style(w_x, phi_x2, phi_x3)
    ix10 = ix01.transpose(0, 1)
    ix32 = ix23.transpose(0, 1)
    iy00 = overlap_1d_li_style(w_y, phi_y0, phi_y0)
    iy11 = overlap_1d_li_style(w_y, phi_y1, phi_y1)
    iy22 = overlap_1d_li_style(w_y, phi_y2, phi_y2)
    iy33 = overlap_1d_li_style(w_y, phi_y3, phi_y3)
    iy01 = overlap_1d_li_style(w_y, phi_y0, phi_y1)
    iy23 = overlap_1d_li_style(w_y, phi_y2, phi_y3)
    iy10 = iy01.transpose(0, 1)
    iy32 = iy23.transpose(0, 1)
    iz00 = overlap_1d_li_style(w_z, phi_z0, phi_z0)
    iz11 = overlap_1d_li_style(w_z, phi_z1, phi_z1)
    iz22 = overlap_1d_li_style(w_z, phi_z2, phi_z2)
    iz33 = overlap_1d_li_style(w_z, phi_z3, phi_z3)
    iz01 = overlap_1d_li_style(w_z, phi_z0, phi_z1)
    iz23 = overlap_1d_li_style(w_z, phi_z2, phi_z3)
    iz10 = iz01.transpose(0, 1)
    iz32 = iz23.transpose(0, 1)
    S00 = ix00 * iy00 * iz00
    S11 = ix11 * iy11 * iz11
    S22 = ix22 * iy22 * iz22
    S33 = ix33 * iy33 * iz33
    S01 = ix01 * iy01 * iz01
    S10 = ix10 * iy10 * iz10
    S23 = ix23 * iy23 * iz23
    S32 = ix32 * iy32 * iz32
    M_up = S00 * S11 - S01 * S10
    M_dn = S22 * S33 - S23 * S32
    partdown = M_up * M_dn
    gx00 = overlap_1d_li_style(w_x, grad_phi_x0, grad_phi_x0)
    gx11 = overlap_1d_li_style(w_x, grad_phi_x1, grad_phi_x1)
    gx22 = overlap_1d_li_style(w_x, grad_phi_x2, grad_phi_x2)
    gx33 = overlap_1d_li_style(w_x, grad_phi_x3, grad_phi_x3)
    gx01 = overlap_1d_li_style(w_x, grad_phi_x0, grad_phi_x1)
    gx23 = overlap_1d_li_style(w_x, grad_phi_x2, grad_phi_x3)
    gx10 = gx01.transpose(0, 1)
    gx32 = gx23.transpose(0, 1)
    gy00 = overlap_1d_li_style(w_y, grad_phi_y0, grad_phi_y0)
    gy11 = overlap_1d_li_style(w_y, grad_phi_y1, grad_phi_y1)
    gy22 = overlap_1d_li_style(w_y, grad_phi_y2, grad_phi_y2)
    gy33 = overlap_1d_li_style(w_y, grad_phi_y3, grad_phi_y3)
    gy01 = overlap_1d_li_style(w_y, grad_phi_y0, grad_phi_y1)
    gy23 = overlap_1d_li_style(w_y, grad_phi_y2, grad_phi_y3)
    gy10 = gy01.transpose(0, 1)
    gy32 = gy23.transpose(0, 1)
    gz00 = overlap_1d_li_style(w_z, grad_phi_z0, grad_phi_z0)
    gz11 = overlap_1d_li_style(w_z, grad_phi_z1, grad_phi_z1)
    gz22 = overlap_1d_li_style(w_z, grad_phi_z2, grad_phi_z2)
    gz33 = overlap_1d_li_style(w_z, grad_phi_z3, grad_phi_z3)
    gz01 = overlap_1d_li_style(w_z, grad_phi_z0, grad_phi_z1)
    gz23 = overlap_1d_li_style(w_z, grad_phi_z2, grad_phi_z3)
    gz10 = gz01.transpose(0, 1)
    gz32 = gz23.transpose(0, 1)
    T00 = 0.5 * (gx00 * iy00 * iz00 + ix00 * gy00 * iz00 + ix00 * iy00 * gz00)
    T11 = 0.5 * (gx11 * iy11 * iz11 + ix11 * gy11 * iz11 + ix11 * iy11 * gz11)
    T22 = 0.5 * (gx22 * iy22 * iz22 + ix22 * gy22 * iz22 + ix22 * iy22 * gz22)
    T33 = 0.5 * (gx33 * iy33 * iz33 + ix33 * gy33 * iz33 + ix33 * iy33 * gz33)
    T01 = 0.5 * (gx01 * iy01 * iz01 + ix01 * gy01 * iz01 + ix01 * iy01 * gz01)
    T10 = 0.5 * (gx10 * iy10 * iz10 + ix10 * gy10 * iz10 + ix10 * iy10 * gz10)
    T23 = 0.5 * (gx23 * iy23 * iz23 + ix23 * gy23 * iz23 + ix23 * iy23 * gz23)
    T32 = 0.5 * (gx32 * iy32 * iz32 + ix32 * gy32 * iz32 + ix32 * iy32 * gz32)
    T_up = T00 * S11 + S00 * T11 - T01 * S10 - S01 * T10
    T_dn = T22 * S33 + S22 * T33 - T23 * S32 - S23 * T32
    partgrad = T_up * M_dn + M_up * T_dn
    del gx00, gx11, gx22, gx33, gx01, gx10, gx23, gx32
    del gy00, gy11, gy22, gy33, gy01, gy10, gy23, gy32
    del gz00, gz11, gz22, gz33, gz01, gz10, gz23, gz32
    del T00, T11, T22, T33, T01, T10, T23, T32, T_up, T_dn
    V00 = calc_V_element_li_style(phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0, phi_s_x0, phi_s_x0, phi_s_y0, phi_s_y0, phi_s_z0, phi_s_z0,w_x, w_y, w_z, w_0, ewald, ewald1, point_sx, point_sy)
    V11 = calc_V_element_li_style(phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1, phi_s_x1, phi_s_x1, phi_s_y1, phi_s_y1, phi_s_z1, phi_s_z1,w_x, w_y, w_z, w_0, ewald, ewald1, point_sx, point_sy)
    V22 = calc_V_element_li_style(phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2, phi_s_x2, phi_s_x2, phi_s_y2, phi_s_y2, phi_s_z2, phi_s_z2,w_x, w_y, w_z, w_0, ewald, ewald1, point_sx, point_sy)
    V33 = calc_V_element_li_style(phi_x3, phi_x3, phi_y3, phi_y3, phi_z3, phi_z3, phi_s_x3, phi_s_x3, phi_s_y3, phi_s_y3, phi_s_z3, phi_s_z3,w_x, w_y, w_z, w_0, ewald, ewald1, point_sx, point_sy)
    V01 = calc_V_element_li_style(phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1, phi_s_x0, phi_s_x1, phi_s_y0, phi_s_y1, phi_s_z0, phi_s_z1,w_x, w_y, w_z, w_0, ewald, ewald1, point_sx, point_sy)
    V23 = calc_V_element_li_style(phi_x2, phi_x3, phi_y2, phi_y3, phi_z2, phi_z3, phi_s_x2, phi_s_x3, phi_s_y2, phi_s_y3, phi_s_z2, phi_s_z3,w_x, w_y, w_z, w_0, ewald, ewald1, point_sx, point_sy)
    V10 = V01.transpose(0, 1)
    V32 = V23.transpose(0, 1)
    V00 = -Z * V00
    V11 = -Z * V11
    V22 = -Z * V22
    V33 = -Z * V33
    V01 = -Z * V01
    V10 = -Z * V10
    V23 = -Z * V23
    V32 = -Z * V32
    Vn_up = V00 * S11 + S00 * V11 - V01 * S10 - S01 * V10
    Vn_dn = V22 * S33 + S22 * V33 - V23 * S32 - S23 * V32
    partr = Vn_up * M_dn + M_up * Vn_dn
    del V00, V11, V22, V33, V01, V10, V23, V32, Vn_up, Vn_dn
    Cx00 = overlap_1d_cheb_li_style(w_x, cheb_poly_cache, phi_x0, phi_x0)
    Cx11 = overlap_1d_cheb_li_style(w_x, cheb_poly_cache, phi_x1, phi_x1)
    Cx22 = overlap_1d_cheb_li_style(w_x, cheb_poly_cache, phi_x2, phi_x2)
    Cx33 = overlap_1d_cheb_li_style(w_x, cheb_poly_cache, phi_x3, phi_x3)
    Cx01 = overlap_1d_cheb_li_style(w_x, cheb_poly_cache, phi_x0, phi_x1)
    Cx23 = overlap_1d_cheb_li_style(w_x, cheb_poly_cache, phi_x2, phi_x3)
    Cx10 = Cx01.transpose(1, 2)
    Cx32 = Cx23.transpose(1, 2)
    Cy00 = overlap_1d_cheb_li_style(w_y, cheb_poly_cache, phi_y0, phi_y0)
    Cy11 = overlap_1d_cheb_li_style(w_y, cheb_poly_cache, phi_y1, phi_y1)
    Cy22 = overlap_1d_cheb_li_style(w_y, cheb_poly_cache, phi_y2, phi_y2)
    Cy33 = overlap_1d_cheb_li_style(w_y, cheb_poly_cache, phi_y3, phi_y3)
    Cy01 = overlap_1d_cheb_li_style(w_y, cheb_poly_cache, phi_y0, phi_y1)
    Cy23 = overlap_1d_cheb_li_style(w_y, cheb_poly_cache, phi_y2, phi_y3)
    Cy10 = Cy01.transpose(1, 2)
    Cy32 = Cy23.transpose(1, 2)
    Cz00 = overlap_1d_cheb_li_style(w_z, cheb_poly_cache, phi_z0, phi_z0)
    Cz11 = overlap_1d_cheb_li_style(w_z, cheb_poly_cache, phi_z1, phi_z1)
    Cz22 = overlap_1d_cheb_li_style(w_z, cheb_poly_cache, phi_z2, phi_z2)
    Cz33 = overlap_1d_cheb_li_style(w_z, cheb_poly_cache, phi_z3, phi_z3)
    Cz01 = overlap_1d_cheb_li_style(w_z, cheb_poly_cache, phi_z0, phi_z1)
    Cz23 = overlap_1d_cheb_li_style(w_z, cheb_poly_cache, phi_z2, phi_z3)
    Cz10 = Cz01.transpose(1, 2)
    Cz32 = Cz23.transpose(1, 2)
    short_x00 = phi_short_x0 * phi_short_x0
    short_x11 = phi_short_x1 * phi_short_x1
    short_x22 = phi_short_x2 * phi_short_x2
    short_x33 = phi_short_x3 * phi_short_x3
    short_x01 = phi_short_x0 * phi_short_x1
    short_x23 = phi_short_x2 * phi_short_x3
    short_x02 = phi_short_x0 * phi_short_x2
    short_x03 = phi_short_x0 * phi_short_x3
    short_x12 = phi_short_x1 * phi_short_x2
    short_x13 = phi_short_x1 * phi_short_x3
    short_y00 = phi_short_y0 * phi_short_y0
    short_y11 = phi_short_y1 * phi_short_y1
    short_y22 = phi_short_y2 * phi_short_y2
    short_y33 = phi_short_y3 * phi_short_y3
    short_y01 = phi_short_y0 * phi_short_y1
    short_y23 = phi_short_y2 * phi_short_y3
    short_y02 = phi_short_y0 * phi_short_y2
    short_y03 = phi_short_y0 * phi_short_y3
    short_y12 = phi_short_y1 * phi_short_y2
    short_y13 = phi_short_y1 * phi_short_y3
    short_z00 = phi_short_z0 * phi_short_z0
    short_z11 = phi_short_z1 * phi_short_z1
    short_z22 = phi_short_z2 * phi_short_z2
    short_z33 = phi_short_z3 * phi_short_z3
    short_z01 = phi_short_z0 * phi_short_z1
    short_z23 = phi_short_z2 * phi_short_z3
    short_z02 = phi_short_z0 * phi_short_z2
    short_z03 = phi_short_z0 * phi_short_z3
    short_z12 = phi_short_z1 * phi_short_z2
    short_z13 = phi_short_z1 * phi_short_z3
    def calc_J_explicit(Xab, Xcd, Yab, Ycd, Zab, Zcd,px_i, px_j, py_i, py_j, pz_i, pz_j,px_k, px_l, py_k, py_l, pz_k, pz_l,short_xik, short_yik, short_zik,short_xjl, short_yjl, short_zjl):
        out = calc_rr_long_li_style(Xab, Xcd, Yab, Ycd, Zab, Zcd,kl_index_long_1, kl_index_long_2, kl_index_long_3,alpha_l1_chev, alpha_l1_chev1, alpha_l2_chev, alpha_l2_chev1, alpha_l3_chev, alpha_l3_chev1)
        tmp = calc_middle_interaction_li_style(px_i, px_j, py_i, py_j, pz_i, pz_j,px_k, px_l, py_k, py_l, pz_k, pz_l,w_x, w_y, w_z,psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1, alpha_psi_m1_x, alpha_psi_m1_y,psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1, alpha_psi_m2_x, alpha_psi_m2_y,psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1, alpha_psi_m3_x, alpha_psi_m3_y)
        out = out + tmp
        del tmp
        tmp = calc_short_from_pairs_li_style(short_xik, short_yik, short_zik, short_xjl, short_yjl, short_zjl, w_short, point_ssx, point_ssy)
        out = out + tmp
        del tmp
        return out
    partrr = torch.zeros_like(partdown)
    tmp = calc_J_explicit(Cx00, Cx11, Cy00, Cy11, Cz00, Cz11,phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0,phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1,short_x01, short_y01, short_z01,short_x01, short_y01, short_z01)
    partrr = partrr + tmp * M_dn
    del tmp
    tmp = calc_J_explicit(Cx01, Cx10, Cy01, Cy10, Cz01, Cz10,phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1,phi_x1, phi_x0, phi_y1, phi_y0, phi_z1, phi_z0,short_x01, short_y01, short_z01,short_x01, short_y01, short_z01)
    partrr = partrr - tmp * M_dn
    del tmp
    tmp = calc_J_explicit(Cx22, Cx33, Cy22, Cy33, Cz22, Cz33,phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2,phi_x3, phi_x3, phi_y3, phi_y3, phi_z3, phi_z3,short_x23, short_y23, short_z23,short_x23, short_y23, short_z23)
    partrr = partrr + tmp * M_up
    del tmp
    tmp = calc_J_explicit(Cx23, Cx32, Cy23, Cy32, Cz23, Cz32,phi_x2, phi_x3, phi_y2, phi_y3, phi_z2, phi_z3,phi_x3, phi_x2, phi_y3, phi_y2, phi_z3, phi_z2,short_x23, short_y23, short_z23,short_x23, short_y23, short_z23)
    partrr = partrr - tmp * M_up
    del tmp
    tmp = calc_J_explicit(Cx00, Cx22, Cy00, Cy22, Cz00, Cz22,phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0,phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2,short_x02, short_y02, short_z02,short_x02, short_y02, short_z02)
    partrr = partrr + S11 * S33 * tmp
    del tmp
    tmp = calc_J_explicit(Cx00, Cx33, Cy00, Cy33, Cz00, Cz33,phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0,phi_x3, phi_x3, phi_y3, phi_y3, phi_z3, phi_z3,short_x03, short_y03, short_z03,short_x03, short_y03, short_z03)
    partrr = partrr + S11 * S22 * tmp
    del tmp
    tmp = calc_J_explicit(Cx00, Cx23, Cy00, Cy23, Cz00, Cz23,phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0,phi_x2, phi_x3, phi_y2, phi_y3, phi_z2, phi_z3,short_x02, short_y02, short_z02,short_x03, short_y03, short_z03)
    partrr = partrr - S11 * S32 * tmp
    del tmp
    tmp = calc_J_explicit(Cx00, Cx32, Cy00, Cy32, Cz00, Cz32,phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0,phi_x3, phi_x2, phi_y3, phi_y2, phi_z3, phi_z2,short_x03, short_y03, short_z03,short_x02, short_y02, short_z02)
    partrr = partrr - S11 * S23 * tmp
    del tmp
    tmp = calc_J_explicit(Cx11, Cx22, Cy11, Cy22, Cz11, Cz22,phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1,phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2,short_x12, short_y12, short_z12,short_x12, short_y12, short_z12)
    partrr = partrr + S00 * S33 * tmp
    del tmp
    tmp = calc_J_explicit(Cx11, Cx33, Cy11, Cy33, Cz11, Cz33,phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1,phi_x3, phi_x3, phi_y3, phi_y3, phi_z3, phi_z3,short_x13, short_y13, short_z13,short_x13, short_y13, short_z13)
    partrr = partrr + S00 * S22 * tmp
    del tmp
    tmp = calc_J_explicit(Cx11, Cx23, Cy11, Cy23, Cz11, Cz23,phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1,phi_x2, phi_x3, phi_y2, phi_y3, phi_z2, phi_z3,short_x12, short_y12, short_z12,short_x13, short_y13, short_z13)
    partrr = partrr - S00 * S32 * tmp
    del tmp
    tmp = calc_J_explicit(Cx11, Cx32, Cy11, Cy32, Cz11, Cz32,phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1,phi_x3, phi_x2, phi_y3, phi_y2, phi_z3, phi_z2,short_x13, short_y13, short_z13,short_x12, short_y12, short_z12)
    partrr = partrr - S00 * S23 * tmp
    del tmp
    tmp = calc_J_explicit(Cx01, Cx22, Cy01, Cy22, Cz01, Cz22,phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1,phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2,short_x02, short_y02, short_z02,short_x12, short_y12, short_z12)
    partrr = partrr - S10 * S33 * tmp
    del tmp
    tmp = calc_J_explicit(Cx01, Cx33, Cy01, Cy33, Cz01, Cz33,phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1,phi_x3, phi_x3, phi_y3, phi_y3, phi_z3, phi_z3,short_x03, short_y03, short_z03,short_x13, short_y13, short_z13)
    partrr = partrr - S10 * S22 * tmp
    del tmp
    tmp = calc_J_explicit(Cx01, Cx23, Cy01, Cy23, Cz01, Cz23,phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1,phi_x2, phi_x3, phi_y2, phi_y3, phi_z2, phi_z3,short_x02, short_y02, short_z02,short_x13, short_y13, short_z13)
    partrr = partrr + S10 * S32 * tmp
    del tmp
    tmp = calc_J_explicit(Cx01, Cx32, Cy01, Cy32, Cz01, Cz32,phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1,phi_x3, phi_x2, phi_y3, phi_y2, phi_z3, phi_z2,short_x03, short_y03, short_z03,short_x12, short_y12, short_z12)
    partrr = partrr + S10 * S23 * tmp
    del tmp
    tmp = calc_J_explicit(Cx10, Cx22, Cy10, Cy22, Cz10, Cz22,phi_x1, phi_x0, phi_y1, phi_y0, phi_z1, phi_z0,phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2,short_x12, short_y12, short_z12,short_x02, short_y02, short_z02)
    partrr = partrr - S01 * S33 * tmp
    del tmp
    tmp = calc_J_explicit(Cx10, Cx33, Cy10, Cy33, Cz10, Cz33,phi_x1, phi_x0, phi_y1, phi_y0, phi_z1, phi_z0,phi_x3, phi_x3, phi_y3, phi_y3, phi_z3, phi_z3,short_x13, short_y13, short_z13,short_x03, short_y03, short_z03)
    partrr = partrr - S01 * S22 * tmp
    del tmp
    tmp = calc_J_explicit(Cx10, Cx23, Cy10, Cy23, Cz10, Cz23,phi_x1, phi_x0, phi_y1, phi_y0, phi_z1, phi_z0,phi_x2, phi_x3, phi_y2, phi_y3, phi_z2, phi_z3,short_x12, short_y12, short_z12,short_x03, short_y03, short_z03)
    partrr = partrr + S01 * S32 * tmp
    del tmp
    tmp = calc_J_explicit(Cx10, Cx32, Cy10, Cy32, Cz10, Cz32,phi_x1, phi_x0, phi_y1, phi_y0, phi_z1, phi_z0,phi_x3, phi_x2, phi_y3, phi_y2, phi_z3, phi_z2,short_x13, short_y13, short_z13,short_x02, short_y02, short_z02)
    partrr = partrr + S01 * S23 * tmp
    del tmp
    del Cx00, Cx11, Cx22, Cx33, Cx01, Cx10, Cx23, Cx32
    del Cy00, Cy11, Cy22, Cy33, Cy01, Cy10, Cy23, Cy32
    del Cz00, Cz11, Cz22, Cz33, Cz01, Cz10, Cz23, Cz32
    del short_x00, short_x11, short_x22, short_x33, short_x01, short_x23, short_x02, short_x03, short_x12, short_x13
    del short_y00, short_y11, short_y22, short_y33, short_y01, short_y23, short_y02, short_y03, short_y12, short_y13
    del short_z00, short_z11, short_z22, short_z33, short_z01, short_z23, short_z02, short_z03, short_z12, short_z13
    return partdown, partgrad, partr, partrr

def solve_eigenvalue_problem(M, A):
    M = 0.5 * (M + M.transpose(0, 1))
    A = 0.5 * (A + A.transpose(0, 1))
    L = torch.linalg.cholesky(M)
    C = torch.linalg.solve(L, A.transpose(0, 1)).transpose(0, 1)
    D = torch.linalg.solve(L, C)
    E, U = torch.linalg.eigh(D)
    ind = torch.argmin(E)
    lam = E[ind]
    alpha = torch.linalg.solve(L.transpose(0, 1), U[:, ind])
    return lam, alpha

# # ********** Compile Configuration **********
# print("Compiling Matrix Construction Logic...")
# build_matrices_logic_be_singlegraph = torch.compile(
#     build_matrices_logic_be_singlegraph,
#     backend="inductor",
#     mode="max-autotune",
#     dynamic=False,
#     fullgraph=False,
# )

def criterion(
    model_x0, model_x1, model_x2, model_x3,
    model_y0, model_y1, model_y2, model_y3,
    model_z0, model_z1, model_z2, model_z3,
):
    phi_x0, grad_phi_x0 = model_x0(w_x, point_x, need_grad=1, normed=False)
    phi_x0_norm = torch.sqrt(torch.sum(w_x * phi_x0 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_x0 = phi_x0 / phi_x0_norm
    grad_phi_x0 = grad_phi_x0 / phi_x0_norm
    phi_x1, grad_phi_x1 = model_x1(w_x, point_x, need_grad=1, normed=False)
    phi_x1_norm = torch.sqrt(torch.sum(w_x * phi_x1 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_x1 = phi_x1 / phi_x1_norm
    grad_phi_x1 = grad_phi_x1 / phi_x1_norm
    phi_x2, grad_phi_x2 = model_x2(w_x, point_x, need_grad=1, normed=False)
    phi_x2_norm = torch.sqrt(torch.sum(w_x * phi_x2 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_x2 = phi_x2 / phi_x2_norm
    grad_phi_x2 = grad_phi_x2 / phi_x2_norm
    phi_x3, grad_phi_x3 = model_x3(w_x, point_x, need_grad=1, normed=False)
    phi_x3_norm = torch.sqrt(torch.sum(w_x * phi_x3 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_x3 = phi_x3 / phi_x3_norm
    grad_phi_x3 = grad_phi_x3 / phi_x3_norm
    phi_y0, grad_phi_y0 = model_y0(w_y, point_y, need_grad=1, normed=False)
    phi_y0_norm = torch.sqrt(torch.sum(w_y * phi_y0 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_y0 = phi_y0 / phi_y0_norm
    grad_phi_y0 = grad_phi_y0 / phi_y0_norm
    phi_y1, grad_phi_y1 = model_y1(w_y, point_y, need_grad=1, normed=False)
    phi_y1_norm = torch.sqrt(torch.sum(w_y * phi_y1 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_y1 = phi_y1 / phi_y1_norm
    grad_phi_y1 = grad_phi_y1 / phi_y1_norm
    phi_y2, grad_phi_y2 = model_y2(w_y, point_y, need_grad=1, normed=False)
    phi_y2_norm = torch.sqrt(torch.sum(w_y * phi_y2 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_y2 = phi_y2 / phi_y2_norm
    grad_phi_y2 = grad_phi_y2 / phi_y2_norm
    phi_y3, grad_phi_y3 = model_y3(w_y, point_y, need_grad=1, normed=False)
    phi_y3_norm = torch.sqrt(torch.sum(w_y * phi_y3 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_y3 = phi_y3 / phi_y3_norm
    grad_phi_y3 = grad_phi_y3 / phi_y3_norm
    phi_z0, grad_phi_z0 = model_z0(w_z, point_z, need_grad=1, normed=False)
    phi_z0_norm = torch.sqrt(torch.sum(w_z * phi_z0 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_z0 = phi_z0 / phi_z0_norm
    grad_phi_z0 = grad_phi_z0 / phi_z0_norm
    phi_z1, grad_phi_z1 = model_z1(w_z, point_z, need_grad=1, normed=False)
    phi_z1_norm = torch.sqrt(torch.sum(w_z * phi_z1 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_z1 = phi_z1 / phi_z1_norm
    grad_phi_z1 = grad_phi_z1 / phi_z1_norm
    phi_z2, grad_phi_z2 = model_z2(w_z, point_z, need_grad=1, normed=False)
    phi_z2_norm = torch.sqrt(torch.sum(w_z * phi_z2 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_z2 = phi_z2 / phi_z2_norm
    grad_phi_z2 = grad_phi_z2 / phi_z2_norm
    phi_z3, grad_phi_z3 = model_z3(w_z, point_z, need_grad=1, normed=False)
    phi_z3_norm = torch.sqrt(torch.sum(w_z * phi_z3 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_z3 = phi_z3 / phi_z3_norm
    grad_phi_z3 = grad_phi_z3 / phi_z3_norm
    phi_short_x0 = model_x0(w_short, point_short, need_grad=0, normed=True)
    phi_short_x1 = model_x1(w_short, point_short, need_grad=0, normed=True)
    phi_short_x2 = model_x2(w_short, point_short, need_grad=0, normed=True)
    phi_short_x3 = model_x3(w_short, point_short, need_grad=0, normed=True)
    phi_short_y0 = model_y0(w_short, point_short, need_grad=0, normed=True)
    phi_short_y1 = model_y1(w_short, point_short, need_grad=0, normed=True)
    phi_short_y2 = model_y2(w_short, point_short, need_grad=0, normed=True)
    phi_short_y3 = model_y3(w_short, point_short, need_grad=0, normed=True)
    phi_short_z0 = model_z0(w_short, point_short, need_grad=0, normed=True)
    phi_short_z1 = model_z1(w_short, point_short, need_grad=0, normed=True)
    phi_short_z2 = model_z2(w_short, point_short, need_grad=0, normed=True)
    phi_short_z3 = model_z3(w_short, point_short, need_grad=0, normed=True)
    phi_s_x0 = model_x0(w_0, point0, need_grad=0, normed=False) / phi_x0_norm
    phi_s_x1 = model_x1(w_0, point0, need_grad=0, normed=False) / phi_x1_norm
    phi_s_x2 = model_x2(w_0, point0, need_grad=0, normed=False) / phi_x2_norm
    phi_s_x3 = model_x3(w_0, point0, need_grad=0, normed=False) / phi_x3_norm
    phi_s_y0 = model_y0(w_0, point0, need_grad=0, normed=False) / phi_y0_norm
    phi_s_y1 = model_y1(w_0, point0, need_grad=0, normed=False) / phi_y1_norm
    phi_s_y2 = model_y2(w_0, point0, need_grad=0, normed=False) / phi_y2_norm
    phi_s_y3 = model_y3(w_0, point0, need_grad=0, normed=False) / phi_y3_norm
    phi_s_z0 = model_z0(w_0, point0, need_grad=0, normed=False) / phi_z0_norm
    phi_s_z1 = model_z1(w_0, point0, need_grad=0, normed=False) / phi_z1_norm
    phi_s_z2 = model_z2(w_0, point0, need_grad=0, normed=False) / phi_z2_norm
    phi_s_z3 = model_z3(w_0, point0, need_grad=0, normed=False) / phi_z3_norm
    partdown, partgrad, partr, partrr = build_matrices_logic_be_singlegraph(
        phi_x0, phi_x1, phi_x2, phi_x3,
        phi_y0, phi_y1, phi_y2, phi_y3,
        phi_z0, phi_z1, phi_z2, phi_z3,
        grad_phi_x0, grad_phi_x1, grad_phi_x2, grad_phi_x3,
        grad_phi_y0, grad_phi_y1, grad_phi_y2, grad_phi_y3,
        grad_phi_z0, grad_phi_z1, grad_phi_z2, grad_phi_z3,
        phi_short_x0, phi_short_x1, phi_short_x2, phi_short_x3,
        phi_short_y0, phi_short_y1, phi_short_y2, phi_short_y3,
        phi_short_z0, phi_short_z1, phi_short_z2, phi_short_z3,
        phi_s_x0, phi_s_x1, phi_s_x2, phi_s_x3,
        phi_s_y0, phi_s_y1, phi_s_y2, phi_s_y3,
        phi_s_z0, phi_s_z1, phi_s_z2, phi_s_z3,
        w_x, w_y, w_z, w_short, w_0,
        ewald, ewald1, point_sx, point_sy, point_ssx, point_ssy,
        cheb_poly_cache, kl_index_long_1, kl_index_long_2, kl_index_long_3,
        alpha_l1_chev, alpha_l1_chev1, alpha_l2_chev, alpha_l2_chev1, alpha_l3_chev, alpha_l3_chev1,
        psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1, alpha_psi_m1_x, alpha_psi_m1_y,
        psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1, alpha_psi_m2_x, alpha_psi_m2_y,
        psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1, alpha_psi_m3_x, alpha_psi_m3_y,
        scale)
    M = scale ** 2 * partdown
    A = partgrad + scale * partr + scale * partrr
    lam, alpha = solve_eigenvalue_problem(M, A)
    loss = lam
    with torch.no_grad():
        alpha_outer = torch.outer(alpha, alpha)
        partgrad_val = torch.sum(alpha_outer * partgrad)
        partr_val = torch.sum(alpha_outer * partr)
        partrr_val = torch.sum(alpha_outer * partrr)
        partdown_val = torch.sum(alpha_outer * partdown)
        lossE = (partgrad_val + scale * partr_val + scale * partrr_val) / (scale ** 2 * partdown_val)
    return loss, lossE, partgrad_val, partr_val, partrr_val, partdown_val, alpha


# ********** Multi-stage Training Adam **********
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
lr_list = [1e-2, 2e-5, 2e-6, 1e-7, 5e-9, 3e-10]
epoch_list = [27500, 15000, 12500, 16500, 21000, 17500]
lossbest, errorbest = [], []
current_min_error = float('inf')
print_every = 100
best_model_states = None
best_alpha = None
reference_energy = -14.66736
print(f'reference_energy={reference_energy:.17g} hartree')
def get_optimizer(lr):
    return optim.Adam(
        filter(lambda p: p.requires_grad,
               itertools.chain(
                   model_x0.parameters(), model_y0.parameters(), model_z0.parameters(),
                   model_x1.parameters(), model_y1.parameters(), model_z1.parameters(),
                   model_x2.parameters(), model_y2.parameters(), model_z2.parameters(),
                   model_x3.parameters(), model_y3.parameters(), model_z3.parameters()
               )),
        lr=lr
    )
starttime = time.time()
trainable_params = sum(p.numel()for p in itertools.chain(model_x0.parameters(),model_y0.parameters(),model_z0.parameters(),model_x1.parameters(),model_y1.parameters(),model_z1.parameters(),model_x2.parameters(),model_y2.parameters(),model_z2.parameters(), model_x3.parameters(), model_y3.parameters(), model_z3.parameters())if p.requires_grad)
for stage, (learning_rate, epochs) in enumerate(zip(lr_list, epoch_list), 1):
    if best_model_states is not None:
        model_x0.load_state_dict(best_model_states['x0'])
        model_x1.load_state_dict(best_model_states['x1'])
        model_x2.load_state_dict(best_model_states['x2'])
        model_x3.load_state_dict(best_model_states['x3'])
        model_y0.load_state_dict(best_model_states['y0'])
        model_y1.load_state_dict(best_model_states['y1'])
        model_y2.load_state_dict(best_model_states['y2'])
        model_y3.load_state_dict(best_model_states['y3'])
        model_z0.load_state_dict(best_model_states['z0'])
        model_z1.load_state_dict(best_model_states['z1'])
        model_z2.load_state_dict(best_model_states['z2'])
        model_z3.load_state_dict(best_model_states['z3'])
        alpha.data.copy_(best_alpha)
    optimizer = get_optimizer(learning_rate)
    for e in range(epochs):
        loss, lossE, partgrad, part2, part3, partdown, alpha = criterion(model_x0, model_x1,model_x2,model_x3,model_y0, model_y1,model_y2,model_y3,model_z0, model_z1,model_z2,model_z3)
        if e == 0:
            lossEv = lossE.item()
            errorv = abs(lossEv - reference_energy) / abs(reference_energy)
            print(f"stage={stage} "f"learning_rate={learning_rate:.0e} "f"epoch={e} "f"rel_error={errorv:.12e} "f"num_params={trainable_params:,}")
        if (e + 1) % 10 == 0:
            lossEv = lossE.item()
            errorv = abs(lossEv - reference_energy) / abs(reference_energy)
            if errorv <= current_min_error:
                current_min_error = errorv
                lossbest.append(lossEv)
                errorbest.append(errorv)
                best_model_states = {
                    'x0': copy.deepcopy(model_x0.state_dict()),
                    'x1': copy.deepcopy(model_x1.state_dict()),
                    'x2': copy.deepcopy(model_x2.state_dict()),
                    'x3': copy.deepcopy(model_x3.state_dict()),
                    'y0': copy.deepcopy(model_y0.state_dict()),
                    'y1': copy.deepcopy(model_y1.state_dict()),
                    'y2': copy.deepcopy(model_y2.state_dict()),
                    'y3': copy.deepcopy(model_y3.state_dict()),
                    'z0': copy.deepcopy(model_z0.state_dict()),
                    'z1': copy.deepcopy(model_z1.state_dict()),
                    'z2': copy.deepcopy(model_z2.state_dict()),
                    'z3': copy.deepcopy(model_z3.state_dict()),
                }
                best_alpha = alpha.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (e + 1) % print_every == 0:
            lossEv = lossE.item()
            errorv = abs(lossEv - reference_energy) / abs(reference_energy)
            torch.cuda.synchronize()
            print(f"stage={stage} "f"learning_rate={learning_rate:.0e} "f"epoch={e+1} "f"rel_error={errorv:.12e} "f"best_rel_error={current_min_error:.2e} "f"num_params={trainable_params:,}")
endtime = time.time()
print('*' * 60)
print('All Training Done!')
print(f'Final Best Error = {current_min_error:.12e}')
print(f'Total Time: {endtime-starttime:.2f}s')
