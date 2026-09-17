import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import copy
import itertools
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
# ********** choose type and device **********
dtype = torch.double
device = 'cuda:0'
# ********** generate data points **********
a_x, b_x = -1., 1.
a_y, b_y = -1., 1.
a_z, b_z = -1., 1.
scale = 15.
# number of quad points
quad = [8, 6, 8]
N = [10, 30, 10]
ratio = [2, 1, 2]
# quad points and quad weights.
point_x, w_x = composite_quadrature_custom_diff_points(quad, a_x, b_x, ratio, N, device=device, dtype=dtype)
point_y, w_y = composite_quadrature_custom_diff_points(quad, a_y, b_y, ratio, N, device=device, dtype=dtype)
point_z, w_z = composite_quadrature_custom_diff_points(quad, a_z, b_z, ratio, N, device=device, dtype=dtype)
point_short, w_short = composite_quadrature_custom_diff_points(quad, a_x, b_x, ratio, N, device=device, dtype=dtype)
N_x, N_y, N_z = len(point_x), len(point_y), len(point_z)
X, Y = torch.meshgrid(point_x, point_x, indexing='ij')
# SOG and range-spiltting parameters
truncation_number_r_positive = 80
truncation_number_r_negative = 24
truncation_number_r = truncation_number_r_positive + truncation_number_r_negative + 1
truncation_number_r_s_from = -150
truncation_number_r_s_to = -25
truncation_number_s = truncation_number_r_s_to - truncation_number_r_s_from + 1
truncation_number_l1_from, truncation_number_l1_to = 15, 80
truncation_number_l1 = truncation_number_l1_to - truncation_number_l1_from + 1
truncation_number_l2_from, truncation_number_l2_to = 0, 14
truncation_number_l2 = truncation_number_l2_to - truncation_number_l2_from + 1
truncation_number_l3_from, truncation_number_l3_to = -5, -1
truncation_number_l3 = truncation_number_l3_to - truncation_number_l3_from + 1
truncation_number_lm1_from, truncation_number_lm1_to, p_lm1 = -10, -6, 60
truncation_number_lm1 = truncation_number_lm1_to - truncation_number_lm1_from + 1
truncation_number_lm2_from, truncation_number_lm2_to, p_lm2 = -14, -11, 160
truncation_number_lm2 = truncation_number_lm2_to - truncation_number_lm2_from + 1
truncation_number_lm3_from, truncation_number_lm3_to, p_lm3 = -17, -15, 310
truncation_number_lm3 = truncation_number_lm3_to - truncation_number_lm3_from + 1
truncation_number_ss_from, truncation_number_ss_to = -150, -18
truncation_number_ss = truncation_number_ss_to - truncation_number_ss_from + 1
p = 60
sizes = [1, 50, 50, p]
# ********** SOG tensor generate **********
b0 = torch.tensor(1.3, device=device, dtype=dtype)
ewald = torch.zeros((truncation_number_r, N_x), dtype=dtype, device=device)
ewald1 = torch.zeros((truncation_number_r, N_x), dtype=dtype, device=device)
for l in range(1, truncation_number_r_negative + 1):
    ewald[truncation_number_r_negative - l, :] = (2.0 * torch.log(b0) / torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) * (b0 ** l) * torch.exp(-0.5 * (b0 ** (2 * l)) * (point_x ** 2))
    ewald1[truncation_number_r_negative - l, :] = torch.exp(-0.5 * (b0 ** (2 * l)) * (point_x ** 2))
for l in range(0, truncation_number_r_positive + 1):
    ewald[truncation_number_r_negative + l, :] = (2.0 * torch.log(b0) / torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) / (b0 ** l) * torch.exp(-0.5 / (b0 ** (2 * l)) * (point_x ** 2))
    ewald1[truncation_number_r_negative + l, :] = torch.exp(-0.5 / (b0 ** (2 * l)) * (point_x ** 2))
point0 = torch.zeros(1, dtype=dtype, device=device)
w_0 = torch.ones_like(point0)
point_sx = torch.zeros(truncation_number_s, dtype=dtype, device=device)
point_sy = torch.zeros(truncation_number_s, dtype=dtype, device=device)
for l in range(truncation_number_r_s_from, truncation_number_r_s_to + 1):
    point_sx[l - truncation_number_r_s_from] = 2.0 * torch.log(b0)
    point_sy[l - truncation_number_r_s_from] = (torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) * (b0 ** l)
b1 = torch.tensor(1.3, dtype=dtype, device=device)
max_cheb_degree = 50
cheb_poly_cache = torch.zeros((max_cheb_degree + 1, N_x), dtype=dtype, device=device)
for i in range(max_cheb_degree + 1):
    if i == 0: cheb_poly_cache[i, :] = torch.ones_like(point_x)
    elif i == 1: cheb_poly_cache[i, :] = point_x
    elif i == 2: cheb_poly_cache[i, :] = 2 * (point_x ** 2) - 1
    else: cheb_poly_cache[i, :] = 2 * point_x * cheb_poly_cache[i - 1, :] - cheb_poly_cache[i - 2, :]
def load_long_term_data(term_idx, trunc_from, trunc_to, trunc_num):
    index_mat_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cheb_mat_data_He_triplet', f'kl_index_long_{term_idx}.mat'))
    kl_index = torch.from_numpy(index_mat_data[f'kl_index_long_{term_idx}']).to(device).to(torch.int).squeeze()
    num_exp = len(kl_index[:, 0])
    co = torch.zeros(trunc_num, dtype=dtype, device=device)
    alpha = torch.zeros((trunc_num, num_exp), dtype=dtype, device=device)
    alpha1 = torch.zeros((trunc_num, num_exp), dtype=dtype, device=device)
    for l in range(trunc_from, trunc_to + 1):
        co[l - trunc_from] = (2.0 * torch.log(b1) / torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) / (b1 ** l)
        mat_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cheb_mat_data_He_triplet', f'vectorE_{l}.mat'))
        alpha1[l - trunc_from, :] = torch.tensor(mat_data['E_vector'], device=device, dtype=dtype).t()
    for l in range(num_exp):
        alpha[:, l] = alpha1[:, l] * co
    return kl_index, alpha, alpha1
kl_index_long_1, alpha_l1_chev, alpha_l1_chev1 = load_long_term_data(1, truncation_number_l1_from, truncation_number_l1_to, truncation_number_l1)
kl_index_long_2, alpha_l2_chev, alpha_l2_chev1 = load_long_term_data(2, truncation_number_l2_from, truncation_number_l2_to, truncation_number_l2)
kl_index_long_3, alpha_l3_chev, alpha_l3_chev1 = load_long_term_data(3, truncation_number_l3_from, truncation_number_l3_to, truncation_number_l3)
def gen_middle_term(trunc_from, trunc_to, trunc_num, p_lm):
    psi_x0, psi_x1, psi_y0, psi_y1 = [torch.zeros((trunc_num, 1, p_lm, N_x), dtype=dtype, device=device) for _ in range(4)]
    alpha_x, alpha_y = torch.zeros((trunc_num, p_lm), dtype=dtype, device=device), torch.zeros((trunc_num, p_lm), dtype=dtype, device=device)
    for l in range(trunc_from, trunc_to + 1):
        sv_vectors, sv_values = low_rank_svd_approximation(b1, l, p_lm, X, Y)
        psi_x0[l - trunc_from, 0] = sv_vectors[0, :, :]
        psi_x1[l - trunc_from, 0] = sv_vectors[1, :, :]
        psi_y0[l - trunc_from, 0] = sv_vectors[0, :, :]
        psi_y1[l - trunc_from, 0] = sv_vectors[1, :, :]
        alpha_x[l - trunc_from] = (2.0 * torch.log(b1) / torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) * (b1 ** (-l)) * sv_values
        alpha_y[l - trunc_from] = sv_values
    return psi_x0, psi_x1, psi_y0, psi_y1, alpha_x, alpha_y
psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1, alpha_psi_m1_x, alpha_psi_m1_y = gen_middle_term(truncation_number_lm1_from, truncation_number_lm1_to, truncation_number_lm1, p_lm1)
psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1, alpha_psi_m2_x, alpha_psi_m2_y = gen_middle_term(truncation_number_lm2_from, truncation_number_lm2_to, truncation_number_lm2, p_lm2)
psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1, alpha_psi_m3_x, alpha_psi_m3_y = gen_middle_term(truncation_number_lm3_from, truncation_number_lm3_to, truncation_number_lm3, p_lm3)
point_ssx = torch.zeros(truncation_number_ss, dtype=dtype, device=device)
point_ssy = torch.zeros(truncation_number_ss, dtype=dtype, device=device)
for l in range(truncation_number_ss_from, truncation_number_ss_to + 1):
    point_ssx[l - truncation_number_ss_from] = 2.0 * torch.log(b1)
    point_ssy[l - truncation_number_ss_from] = (torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) * (b1 ** l)
# Combine with quadrature weights to form weighted operators
w_ewald_x  = w_x * ewald
w_ewald1_y = w_y * ewald1
w_ewald1_z = w_z * ewald1
w_cheb_x = w_x * cheb_poly_cache
w_cheb_y = w_y * cheb_poly_cache
w_cheb_z = w_z * cheb_poly_cache
w_0_sx = w_0 * point_sx.unsqueeze(1)
w_0_sy = w_0 * point_sy.unsqueeze(1)
w_0_sz = w_0 * point_sy.unsqueeze(1) 
w_short_ssx = w_short * point_ssx.unsqueeze(1)
w_short_ssy = w_short * point_ssy.unsqueeze(1)
w_short_ssz = w_short * point_ssy.unsqueeze(1)
# ********** Main Model **********
activation = TNN_Sin
model_x0 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_x1 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_y0 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_y1 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_z0 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_z1 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)

def build_matrices_logic(
    phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1,
    grad_phi_x0, grad_phi_x1, grad_phi_y0, grad_phi_y1, grad_phi_z0, grad_phi_z1,
    phi_s_x0, phi_s_x1,  phi_s_y0, phi_s_y1,  phi_s_z0, phi_s_z1,
    phi_short_x0, phi_short_x1, phi_short_y0, phi_short_y1, phi_short_z0, phi_short_z1,
    w_x, w_y, w_z, w_short, w_0,
    w_ewald_x, w_ewald1_y, w_ewald1_z, w_0_sx, w_0_sy, w_0_sz,
    w_short_ssx, w_short_ssy, w_short_ssz,
    w_cheb_x, w_cheb_y, w_cheb_z,
    kl_index_long_1, kl_index_long_2, kl_index_long_3,
    alpha_l1_chev, alpha_l1_chev1, alpha_l2_chev, alpha_l2_chev1, alpha_l3_chev, alpha_l3_chev1,
    psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1, alpha_psi_m1_x, alpha_psi_m1_y,
    psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1, alpha_psi_m2_x, alpha_psi_m2_y,
    psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1, alpha_psi_m3_x, alpha_psi_m3_y,
    scale):
    def calc_inner(w, L, R):
        return torch.mm(w * L.squeeze(0), R.squeeze(0).T)
    def calc_weighted_inner(W_v, L, R):
        L_sq, R_sq = L.squeeze(0), R.squeeze(0)
        W_L = (W_v.unsqueeze(1) * L_sq.unsqueeze(0)).view(-1, W_v.shape[1]) 
        return torch.mm(W_L, R_sq.T).view(W_v.shape[0], L_sq.shape[0], R_sq.shape[0])
    def calc_short_term(W_v, A1, A2, B1, B2):
        A, B = A1.squeeze(0) * A2.squeeze(0), B1.squeeze(0) * B2.squeeze(0)
        W_A = (W_v.unsqueeze(1) * A.unsqueeze(0)).view(-1, W_v.shape[1]) 
        return torch.mm(W_A, B.T).view(W_v.shape[0], A.shape[0], B.shape[0])
    def get_proj(w, psi, pL, pR, alpha=None):
        q, r = pL.shape[1], pR.shape[1]
        pL_pR = (pL.squeeze(0).unsqueeze(1) * pR.squeeze(0).unsqueeze(0)).view(q*r, -1) 
        if alpha is not None:
            w_psi = (w.view(1, 1, -1) * alpha.unsqueeze(2) * psi.squeeze(1)).view(-1, pL_pR.shape[1]) 
        else:
            w_psi = (w.view(1, 1, -1) * psi.squeeze(1)).view(-1, pL_pR.shape[1]) 
        return torch.mm(w_psi, pL_pR.T).view(psi.shape[0], psi.shape[2], q*r)
    def calc_long_chev_term(alpha_x, alpha_y, alpha_z, idx, X0, X1, Y0, Y1, Z0, Z1, p, q):
        X_term = torch.mm(alpha_x, (X0[idx[:, 0]] * X1[idx[:, 1]]).view(idx.shape[0], -1))
        Y_term = torch.mm(alpha_y, (Y0[idx[:, 0]] * Y1[idx[:, 1]]).view(idx.shape[0], -1))
        Z_term = torch.mm(alpha_z, (Z0[idx[:, 0]] * Z1[idx[:, 1]]).view(idx.shape[0], -1))
        return torch.sum(X_term * Y_term * Z_term, dim=0).view(p, q)
    int_phi_x0_x0 = calc_inner(w_x, phi_x0, phi_x0)
    int_phi_x1_x1 = calc_inner(w_x, phi_x1, phi_x1)
    int_phi_x0_x1 = calc_inner(w_x, phi_x0, phi_x1)
    int_phi_x1_x0 = int_phi_x0_x1.t()
    int_phi_y0_y0 = calc_inner(w_y, phi_y0, phi_y0)
    int_phi_y1_y1 = calc_inner(w_y, phi_y1, phi_y1)
    int_phi_y0_y1 = calc_inner(w_y, phi_y0, phi_y1)
    int_phi_y1_y0 = int_phi_y0_y1.t()
    int_phi_z0_z0 = calc_inner(w_z, phi_z0, phi_z0)
    int_phi_z1_z1 = calc_inner(w_z, phi_z1, phi_z1)
    int_phi_z0_z1 = calc_inner(w_z, phi_z0, phi_z1)
    int_phi_z1_z0 = int_phi_z0_z1.t()
    partdown1 = int_phi_x0_x0 * int_phi_x1_x1 * int_phi_y0_y0 * int_phi_y1_y1 * int_phi_z0_z0 * int_phi_z1_z1 
    partdown2 = int_phi_x0_x1 * int_phi_x1_x0 * int_phi_y0_y1 * int_phi_y1_y0 * int_phi_z0_z1 * int_phi_z1_z0 
    partdown = 0.5 * (partdown1 - partdown2)
    int_grad_phi_x0_x0 = calc_inner(w_x, grad_phi_x0, grad_phi_x0)
    int_grad_phi_x1_x1 = calc_inner(w_x, grad_phi_x1, grad_phi_x1)
    int_grad_phi_x0_x1 = calc_inner(w_x, grad_phi_x0, grad_phi_x1)
    int_grad_phi_x1_x0 = int_grad_phi_x0_x1.t()
    int_grad_phi_y0_y0 = calc_inner(w_y, grad_phi_y0, grad_phi_y0)
    int_grad_phi_y1_y1 = calc_inner(w_y, grad_phi_y1, grad_phi_y1)
    int_grad_phi_y0_y1 = calc_inner(w_y, grad_phi_y0, grad_phi_y1)
    int_grad_phi_y1_y0 = int_grad_phi_y0_y1.t()
    int_grad_phi_z0_z0 = calc_inner(w_z, grad_phi_z0, grad_phi_z0)
    int_grad_phi_z1_z1 = calc_inner(w_z, grad_phi_z1, grad_phi_z1)
    int_grad_phi_z0_z1 = calc_inner(w_z, grad_phi_z0, grad_phi_z1)
    int_grad_phi_z1_z0 = int_grad_phi_z0_z1.t()
    partgrad_x = (int_grad_phi_x0_x0 * int_phi_x1_x1 + int_phi_x0_x0 * int_grad_phi_x1_x1) * int_phi_y0_y0 * int_phi_y1_y1  * int_phi_z0_z0 * int_phi_z1_z1 
    partgrad_x_ex = (int_grad_phi_x0_x1 * int_phi_x1_x0 + int_phi_x0_x1 * int_grad_phi_x1_x0) * int_phi_y0_y1 * int_phi_y1_y0  * int_phi_z0_z1 * int_phi_z1_z0
    partgrad_y = (int_grad_phi_y0_y0 * int_phi_y1_y1 + int_phi_y0_y0 * int_grad_phi_y1_y1) * int_phi_x0_x0 * int_phi_x1_x1  * int_phi_z0_z0 * int_phi_z1_z1 
    partgrad_y_ex = (int_grad_phi_y0_y1 * int_phi_y1_y0 + int_phi_y0_y1 * int_grad_phi_y1_y0) * int_phi_x0_x1 * int_phi_x1_x0  * int_phi_z0_z1 * int_phi_z1_z0
    partgrad_z = (int_grad_phi_z0_z0 * int_phi_z1_z1 + int_phi_z0_z0 * int_grad_phi_z1_z1) * int_phi_x0_x0 * int_phi_x1_x1  * int_phi_y0_y0 * int_phi_y1_y1 
    partgrad_z_ex = (int_grad_phi_z0_z1 * int_phi_z1_z0 + int_phi_z0_z1 * int_grad_phi_z1_z0) * int_phi_x0_x1 * int_phi_x1_x0  * int_phi_y0_y1 * int_phi_y1_y0
    partgrad1 = partgrad_x + partgrad_y + partgrad_z
    partgrad2 = partgrad_x_ex + partgrad_y_ex + partgrad_z_ex
    partgrad = (0.25 / (scale ** 2)) * (partgrad1 - partgrad2)
    def calc_V_element(phi_xL, phi_xR, phi_yL, phi_yR, phi_zL, phi_zR, phi_sxL, phi_sxR, phi_syL, phi_syR, phi_szL, phi_szR):
        term_long = torch.sum(calc_weighted_inner(w_ewald_x, phi_xL, phi_xR) * calc_weighted_inner(w_ewald1_y, phi_yL, phi_yR) * calc_weighted_inner(w_ewald1_z, phi_zL, phi_zR), dim=0)
        term_short = torch.sum(calc_weighted_inner(w_0_sx, phi_sxL, phi_sxR) * calc_weighted_inner(w_0_sy, phi_syL, phi_syR) * calc_weighted_inner(w_0_sz, phi_szL, phi_szR), dim=0)
        return term_long + term_short
    partr_r0_r0 = calc_V_element(phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0, phi_s_x0, phi_s_x0, phi_s_y0, phi_s_y0, phi_s_z0, phi_s_z0)
    partr_r1_r1 = calc_V_element(phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1, phi_s_x1, phi_s_x1, phi_s_y1, phi_s_y1, phi_s_z1, phi_s_z1)
    partr_r0_r1 = calc_V_element(phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1, phi_s_x0, phi_s_x1, phi_s_y0, phi_s_y1, phi_s_z0, phi_s_z1)
    partr_r1_r0 = partr_r0_r1.t()
    partr1 = -(2.0 / scale) * (partr_r0_r0 * int_phi_x1_x1 * int_phi_y1_y1 * int_phi_z1_z1 + partr_r1_r1 * int_phi_x0_x0 * int_phi_y0_y0 * int_phi_z0_z0) 
    partr2 = -(2.0 / scale) * (partr_r0_r1 * int_phi_x1_x0 * int_phi_y1_y0 * int_phi_z1_z0 + partr_r1_r0 * int_phi_x0_x1 * int_phi_y0_y1 * int_phi_z0_z1)
    partr = 0.5 * (partr1 - partr2)
    int_cheb_phi_x0_x0 = calc_weighted_inner(w_cheb_x, phi_x0, phi_x0)
    int_cheb_phi_x1_x1 = calc_weighted_inner(w_cheb_x, phi_x1, phi_x1)
    int_cheb_phi_x0_x1 = calc_weighted_inner(w_cheb_x, phi_x0, phi_x1)
    int_cheb_phi_x1_x0 = int_cheb_phi_x0_x1.transpose(1, 2)
    int_cheb_phi_y0_y0 = calc_weighted_inner(w_cheb_y, phi_y0, phi_y0)
    int_cheb_phi_y1_y1 = calc_weighted_inner(w_cheb_y, phi_y1, phi_y1)
    int_cheb_phi_y0_y1 = calc_weighted_inner(w_cheb_y, phi_y0, phi_y1)
    int_cheb_phi_y1_y0 = int_cheb_phi_y0_y1.transpose(1, 2)
    int_cheb_phi_z0_z0 = calc_weighted_inner(w_cheb_z, phi_z0, phi_z0)
    int_cheb_phi_z1_z1 = calc_weighted_inner(w_cheb_z, phi_z1, phi_z1)
    int_cheb_phi_z0_z1 = calc_weighted_inner(w_cheb_z, phi_z0, phi_z1)
    int_cheb_phi_z1_z0 = int_cheb_phi_z0_z1.transpose(1, 2)
    p_dim, q_dim = phi_x0.shape[1], phi_x0.shape[1]
    partrr01_long_1 = calc_long_chev_term(
        alpha_l1_chev, alpha_l1_chev1, alpha_l1_chev1, kl_index_long_1, 
        int_cheb_phi_x0_x0, int_cheb_phi_x1_x1, 
        int_cheb_phi_y0_y0, int_cheb_phi_y1_y1, 
        int_cheb_phi_z0_z0, int_cheb_phi_z1_z1, 
        p_dim, q_dim)
    partrr01_long_1_ex = calc_long_chev_term(
        alpha_l1_chev, alpha_l1_chev1, alpha_l1_chev1, kl_index_long_1, 
        int_cheb_phi_x0_x1, int_cheb_phi_x1_x0, 
        int_cheb_phi_y0_y1, int_cheb_phi_y1_y0, 
        int_cheb_phi_z0_z1, int_cheb_phi_z1_z0, 
        p_dim, q_dim)
    partrr01_long_2 = calc_long_chev_term(
        alpha_l2_chev, alpha_l2_chev1, alpha_l2_chev1, kl_index_long_2, 
        int_cheb_phi_x0_x0, int_cheb_phi_x1_x1, 
        int_cheb_phi_y0_y0, int_cheb_phi_y1_y1, 
        int_cheb_phi_z0_z0, int_cheb_phi_z1_z1, 
        p_dim, q_dim)
    partrr01_long_2_ex = calc_long_chev_term(
        alpha_l2_chev, alpha_l2_chev1, alpha_l2_chev1, kl_index_long_2, 
        int_cheb_phi_x0_x1, int_cheb_phi_x1_x0, 
        int_cheb_phi_y0_y1, int_cheb_phi_y1_y0, 
        int_cheb_phi_z0_z1, int_cheb_phi_z1_z0, 
        p_dim, q_dim)
    partrr01_long_3 = calc_long_chev_term(
        alpha_l3_chev, alpha_l3_chev1, alpha_l3_chev1, kl_index_long_3, 
        int_cheb_phi_x0_x0, int_cheb_phi_x1_x1, 
        int_cheb_phi_y0_y0, int_cheb_phi_y1_y1, 
        int_cheb_phi_z0_z0, int_cheb_phi_z1_z1, 
        p_dim, q_dim)
    partrr01_long_3_ex = calc_long_chev_term(
        alpha_l3_chev, alpha_l3_chev1, alpha_l3_chev1, kl_index_long_3, 
        int_cheb_phi_x0_x1, int_cheb_phi_x1_x0, 
        int_cheb_phi_y0_y1, int_cheb_phi_y1_y0, 
        int_cheb_phi_z0_z1, int_cheb_phi_z1_z0, 
        p_dim, q_dim)
    partrr01_long = partrr01_long_1 + partrr01_long_2 + partrr01_long_3
    partrr01_long_ex = partrr01_long_1_ex + partrr01_long_2_ex + partrr01_long_3_ex
    partrr_long = 0.5 * (partrr01_long - partrr01_long_ex)
    def calc_middle_interaction(px_L1, px_R1, py_L1, py_R1, pz_L1, pz_R1, px_L2, px_R2, py_L2, py_R2, pz_L2, pz_R2):
        def calc_single_mode_opt(alpha_x, alpha_y, psi_x0, psi_x1, psi_y0, psi_y1):
            A_x = get_proj(w_x, psi_x0, px_L1, px_R1, alpha_x) 
            B_x = get_proj(w_x, psi_x1, px_L2, px_R2)
            term_x = torch.sum(A_x * B_x, dim=1) 
            A_y = get_proj(w_y, psi_y0, py_L1, py_R1, alpha_y)
            B_y = get_proj(w_y, psi_y1, py_L2, py_R2)
            term_y = torch.sum(A_y * B_y, dim=1)
            A_z = get_proj(w_z, psi_y0, pz_L1, pz_R1, alpha_y)
            B_z = get_proj(w_z, psi_y1, pz_L2, pz_R2)
            term_z = torch.sum(A_z * B_z, dim=1)
            res_flat = torch.sum(term_x * term_y * term_z, dim=0) 
            return res_flat.view(px_L1.shape[1], px_R1.shape[1])
        res_m1 = calc_single_mode_opt(alpha_psi_m1_x, alpha_psi_m1_y, psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1)
        res_m2 = calc_single_mode_opt(alpha_psi_m2_x, alpha_psi_m2_y, psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1)
        res_m3 = calc_single_mode_opt(alpha_psi_m3_x, alpha_psi_m3_y, psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1)
        return res_m1 + res_m2 + res_m3
    mid_01_direct = calc_middle_interaction(phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0, phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1)
    mid_01_exchange = calc_middle_interaction(phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1, phi_x1, phi_x0, phi_y1, phi_y0, phi_z1, phi_z0)
    partrr_middle = 0.5 * (mid_01_direct - mid_01_exchange)
    def calc_short_interaction(px_L1, px_R1, py_L1, py_R1, pz_L1, pz_R1, px_L2, px_R2, py_L2, py_R2, pz_L2, pz_R2):
        term_x = calc_short_term(w_short_ssx, px_L1, px_R1, px_L2, px_R2)
        term_y = calc_short_term(w_short_ssy, py_L1, py_R1, py_L2, py_R2)
        term_z = calc_short_term(w_short_ssz, pz_L1, pz_R1, pz_L2, pz_R2)
        return torch.sum(term_x * term_y * term_z, dim=0)
    short_01_direct = calc_short_interaction(phi_short_x0, phi_short_x1, phi_short_y0, phi_short_y1, phi_short_z0, phi_short_z1, phi_short_x0, phi_short_x1, phi_short_y0, phi_short_y1, phi_short_z0, phi_short_z1)
    short_01_exchange = calc_short_interaction(phi_short_x0, phi_short_x1, phi_short_y0, phi_short_y1, phi_short_z0, phi_short_z1, phi_short_x1, phi_short_x0, phi_short_y1, phi_short_y0, phi_short_z1, phi_short_z0)
    partrr_short = 0.5 * (short_01_direct - short_01_exchange)
    partrr = (1/scale) * (partrr_long + partrr_middle + partrr_short)
    return partdown, partgrad, partr, partrr

def solve_eigenvalue_problem(M, A, scale):
    L = torch.linalg.cholesky(M)
    C = torch.linalg.solve(L, A.t()).t()
    D = torch.linalg.solve(L, C)
    E, U = torch.linalg.eigh(D)
    ind = torch.argmin(E)
    lam = E[ind]
    alpha = torch.linalg.solve(L.t(), U[:, ind])
    return lam, alpha
    
# # ********** Compile Configuration **********
# print("Compiling Matrix Construction Logic...")
# build_matrices_logic = torch.compile(build_matrices_logic, mode='max-autotune')

def criterion(model_x0, model_x1, model_y0, model_y1, model_z0, model_z1):
    phi_x0, grad_phi_x0 = model_x0(w_x, point_x * scale, need_grad=1, normed=False)
    phi_x0_norm = torch.sqrt(torch.sum(w_x * phi_x0 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_x0, grad_phi_x0 = phi_x0 / phi_x0_norm, grad_phi_x0 * scale / phi_x0_norm
    phi_x1, grad_phi_x1 = model_x1(w_x, point_x * scale, need_grad=1, normed=False)
    phi_x1_norm = torch.sqrt(torch.sum(w_x * phi_x1 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_x1, grad_phi_x1 = phi_x1 / phi_x1_norm, grad_phi_x1 * scale / phi_x1_norm
    phi_y0, grad_phi_y0 = model_y0(w_y, point_y * scale, need_grad=1, normed=False)
    phi_y0_norm = torch.sqrt(torch.sum(w_y * phi_y0 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_y0, grad_phi_y0 = phi_y0 / phi_y0_norm, grad_phi_y0 * scale / phi_y0_norm
    phi_y1, grad_phi_y1 = model_y1(w_y, point_y * scale, need_grad=1, normed=False)
    phi_y1_norm = torch.sqrt(torch.sum(w_y * phi_y1 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_y1, grad_phi_y1 = phi_y1 / phi_y1_norm, grad_phi_y1 * scale / phi_y1_norm
    phi_z0, grad_phi_z0 = model_z0(w_z, point_z * scale, need_grad=1, normed=False)
    phi_z0_norm = torch.sqrt(torch.sum(w_z * phi_z0 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_z0, grad_phi_z0 = phi_z0 / phi_z0_norm, grad_phi_z0 * scale / phi_z0_norm
    phi_z1, grad_phi_z1 = model_z1(w_z, point_z * scale, need_grad=1, normed=False)
    phi_z1_norm = torch.sqrt(torch.sum(w_z * phi_z1 ** 2, dim=-1)).unsqueeze(dim=-1)
    phi_z1, grad_phi_z1 = phi_z1 / phi_z1_norm, grad_phi_z1 * scale / phi_z1_norm
    phi_short_x0 = model_x0(w_short, point_short * scale, need_grad=0, normed=True)
    phi_short_x1 = model_x1(w_short, point_short * scale, need_grad=0, normed=True)
    phi_short_y0 = model_y0(w_short, point_short * scale, need_grad=0, normed=True)
    phi_short_y1 = model_y1(w_short, point_short * scale, need_grad=0, normed=True)
    phi_short_z0 = model_z0(w_short, point_short * scale, need_grad=0, normed=True)
    phi_short_z1 = model_z1(w_short, point_short * scale, need_grad=0, normed=True)
    phi_s_x0 = model_x0(w_0, point0 * scale, need_grad=0, normed=False) / phi_x0_norm
    phi_s_x1 = model_x1(w_0, point0 * scale, need_grad=0, normed=False) / phi_x1_norm
    phi_s_y0 = model_y0(w_0, point0 * scale, need_grad=0, normed=False) / phi_y0_norm
    phi_s_y1 = model_y1(w_0, point0 * scale, need_grad=0, normed=False) / phi_y1_norm
    phi_s_z0 = model_z0(w_0, point0 * scale, need_grad=0, normed=False) / phi_z0_norm
    phi_s_z1 = model_z1(w_0, point0 * scale, need_grad=0, normed=False) / phi_z1_norm
    partdown, partgrad, partr, partrr = build_matrices_logic(
        phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1,
        grad_phi_x0, grad_phi_x1, grad_phi_y0, grad_phi_y1, grad_phi_z0, grad_phi_z1,
        phi_s_x0, phi_s_x1,  phi_s_y0, phi_s_y1,  phi_s_z0, phi_s_z1,
        phi_short_x0, phi_short_x1, phi_short_y0, phi_short_y1, phi_short_z0, phi_short_z1,
        w_x, w_y, w_z, w_short, w_0,
        w_ewald_x, w_ewald1_y, w_ewald1_z, w_0_sx, w_0_sy, w_0_sz,
        w_short_ssx, w_short_ssy, w_short_ssz,
        w_cheb_x, w_cheb_y, w_cheb_z,
        kl_index_long_1, kl_index_long_2, kl_index_long_3,
        alpha_l1_chev, alpha_l1_chev1, alpha_l2_chev, alpha_l2_chev1, alpha_l3_chev, alpha_l3_chev1,
        psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1, alpha_psi_m1_x, alpha_psi_m1_y,
        psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1, alpha_psi_m2_x, alpha_psi_m2_y,
        psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1, alpha_psi_m3_x, alpha_psi_m3_y,
        scale)
    M = partdown
    A = partgrad + partr + partrr
    lam, alpha = solve_eigenvalue_problem(M, A, scale)
    loss = lam 
    with torch.no_grad():
        alpha_outer = torch.outer(alpha, alpha)
        partgrad_v = torch.sum(alpha_outer * partgrad)
        partr_v = torch.sum(alpha_outer * partr)
        partrr_v = torch.sum(alpha_outer * partrr)
        partdown_v = torch.sum(alpha_outer * partdown)
        lossE = (partgrad_v + partr_v + partrr_v) / partdown_v

    return loss, lossE, partgrad_v, partr_v, partrr_v, partdown_v, alpha

# ********** Multi-stage Training Adam **********
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
lr_list = [5e-3, 1e-5, 2e-7, 2e-8]
epoch_list = [13500, 8000, 8000, 8500]
lossbest, errorbest = [], []
current_min_error = float('inf')
print_every = 100
best_model_states = None
best_alpha = None
reference_energy = -2.175229378236791
print(f'reference_energy={reference_energy:.17g} hartree')
def get_optimizer(lr):
    return optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(model_x0.parameters(), model_y0.parameters(), model_z0.parameters(), model_x1.parameters(), model_y1.parameters(), model_z1.parameters())),lr=lr)
starttime = time.time()
trainable_params = sum(p.numel()for p in itertools.chain(model_x0.parameters(),model_y0.parameters(),model_z0.parameters(),model_x1.parameters(),model_y1.parameters(),model_z1.parameters())if p.requires_grad)
for stage, (learning_rate, epochs) in enumerate(zip(lr_list, epoch_list), 1):
    if best_model_states is not None:
        model_x0.load_state_dict(best_model_states['x0'])
        model_x1.load_state_dict(best_model_states['x1'])
        model_y0.load_state_dict(best_model_states['y0'])
        model_y1.load_state_dict(best_model_states['y1'])
        model_z0.load_state_dict(best_model_states['z0'])
        model_z1.load_state_dict(best_model_states['z1'])
        alpha.data.copy_(best_alpha)
    optimizer = get_optimizer(learning_rate)
    log_start_time = time.time()
    for e in range(epochs):
        loss, lossE, partgrad, part2, part3, partdown, alpha = criterion(model_x0, model_x1, model_y0, model_y1, model_z0, model_z1)
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
                    'y0': copy.deepcopy(model_y0.state_dict()),
                    'y1': copy.deepcopy(model_y1.state_dict()),
                    'z0': copy.deepcopy(model_z0.state_dict()),
                    'z1': copy.deepcopy(model_z1.state_dict())
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
print('All Training Done!')
print(f'Final Best Error = {current_min_error:.12e}')
print(f'Total Time: {endtime-starttime:.2f}s')