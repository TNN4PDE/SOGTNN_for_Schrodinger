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
a_x = -1.
b_x = 1.
a_y = -1.
b_y = 1.
a_z = -1.
b_z = 1.
scale = 12.
# number of quad points
quad = [8, 6, 8]
N = [10, 30, 10]
ratio = [2, 1, 2]
# quad points and quad weights.
point_x, w_x = composite_quadrature_custom_diff_points(quad, a_x, b_x, ratio, N, device=device, dtype=dtype)
point_y, w_y = composite_quadrature_custom_diff_points(quad, a_y, b_y, ratio, N, device=device, dtype=dtype)
point_z, w_z = composite_quadrature_custom_diff_points(quad, a_z, b_z, ratio, N, device=device, dtype=dtype)
N_x = len(point_x)
N_y = len(point_y)
N_z = len(point_z)
X, Y = torch.meshgrid(point_x, point_x, indexing='ij')
# SOG and range-spiltting parameters
truncation_number_r_positive = 80
truncation_number_r_negative = 24
truncation_number_r = truncation_number_r_positive + truncation_number_r_negative + 1
truncation_number_r_s_from = -150
truncation_number_r_s_to = -25
truncation_number_s = truncation_number_r_s_to - truncation_number_r_s_from + 1
truncation_number_l1_from = 15
truncation_number_l1_to = 80
truncation_number_l1 = truncation_number_l1_to - truncation_number_l1_from + 1
truncation_number_l2_from = 0
truncation_number_l2_to = 14
truncation_number_l2 = truncation_number_l2_to - truncation_number_l2_from + 1
truncation_number_l3_from = -5
truncation_number_l3_to = -1
truncation_number_l3 = truncation_number_l3_to - truncation_number_l3_from + 1
truncation_number_lm1_from = -10
truncation_number_lm1_to = -6
truncation_number_lm1 = truncation_number_lm1_to - truncation_number_lm1_from + 1
p_lm1 = 60
truncation_number_lm2_from = -14
truncation_number_lm2_to = -11
truncation_number_lm2 = truncation_number_lm2_to - truncation_number_lm2_from + 1
p_lm2 = 160
truncation_number_lm3_from = -17
truncation_number_lm3_to = -15
truncation_number_lm3 = truncation_number_lm3_to - truncation_number_lm3_from + 1
p_lm3 = 310
truncation_number_ss_from = -150
truncation_number_ss_to = -18
truncation_number_ss = truncation_number_ss_to - truncation_number_ss_from + 1
p = 60
sizes = [1, 50, 50, p]
# ********** SOG tensor generate **********
b0 = torch.tensor(1.3, device=device, dtype=dtype)
ewald = torch.zeros((truncation_number_r, N_x), dtype=dtype, device=device)
ewald1 = torch.zeros((truncation_number_r, N_x), dtype=dtype, device=device)
for l in range(1, truncation_number_r_negative + 1):
    ewald[truncation_number_r_negative - l, :] = (2.0 * torch.log(b0) / torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) * (
            b0 ** l) * torch.exp(-0.5 * (b0 ** (2 * l)) * (point_x ** 2))
    ewald1[truncation_number_r_negative - l, :] = torch.exp(-0.5 * (b0 ** (2 * l)) * (point_x ** 2))
for l in range(0, truncation_number_r_positive + 1):
    ewald[truncation_number_r_negative + l, :] = (2.0 * torch.log(b0) / torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) / (
            b0 ** l) * torch.exp(-0.5 / (b0 ** (2 * l)) * (point_x ** 2))
    ewald1[truncation_number_r_negative + l, :] = torch.exp(-0.5 / (b0 ** (2 * l)) * (point_x ** 2))
point0 = torch.zeros(1, dtype=dtype, device=device)
w_0 = torch.ones_like(point0)
point_sx = torch.zeros(truncation_number_s, dtype=dtype, device=device)
point_sy = torch.zeros(truncation_number_s, dtype=dtype, device=device)
for l in range(truncation_number_r_s_from, truncation_number_r_s_to + 1):
    tensorx = 2.0 * torch.log(b0)
    tensory = (torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) * (b0 ** l)
    point_sx[l - truncation_number_r_s_from] = tensorx
    point_sy[l - truncation_number_r_s_from] = tensory
b1 = torch.tensor(1.3, dtype=dtype, device=device)
max_cheb_degree = 50
cheb_poly_cache = torch.zeros((max_cheb_degree + 1, N_x), dtype=dtype, device=device)
for i in range(max_cheb_degree + 1):
    if i == 0:
        cheb_poly_cache[i, :] = torch.ones_like(point_x)
    elif i == 1:
        cheb_poly_cache[i, :] = point_x
    elif i == 2:
        cheb_poly_cache[i, :] = 2 * (point_x ** 2) - 1
    else:
        cheb_poly_cache[i, :] = 2 * point_x * cheb_poly_cache[i - 1, :] - cheb_poly_cache[i - 2, :]
index_mat_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cheb_mat_data_Li', f'kl_index_long_1.mat'))
kl_index_long_1 = torch.from_numpy(index_mat_data['kl_index_long_1']).to(device).to(torch.int).squeeze()
num_of_cheb_expansion_long_1 = len(kl_index_long_1[:, 0])
co_l1 = torch.zeros(truncation_number_l1, dtype=dtype, device=device)
for l in range(truncation_number_l1_from, truncation_number_l1_to + 1):
    co_l1[l - truncation_number_l1_from] = (2.0 * torch.log(b1) / torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) / (b1 ** l)
alpha_l1_chev = torch.zeros((truncation_number_l1, num_of_cheb_expansion_long_1), dtype=dtype, device=device)
alpha_l1_chev1 = torch.zeros((truncation_number_l1, num_of_cheb_expansion_long_1), dtype=dtype, device=device)
for l in range(truncation_number_l1_from, truncation_number_l1_to + 1):
    mat_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cheb_mat_data_Li', f'vectorE_{l}.mat'))
    alpha_l1_chev1[l - truncation_number_l1_from, :] = torch.tensor(mat_data['E_vector'], device=device, dtype=dtype).t()
for l in range(num_of_cheb_expansion_long_1):
    alpha_l1_chev[:, l] = alpha_l1_chev1[:, l] * co_l1
index_mat_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cheb_mat_data_Li', f'kl_index_long_2.mat'))
kl_index_long_2 = torch.from_numpy(index_mat_data['kl_index_long_2']).to(device).to(torch.int).squeeze()
num_of_cheb_expansion_long_2 = len(kl_index_long_2[:, 0])
co_l2 = torch.zeros(truncation_number_l2, dtype=dtype, device=device)
for l in range(truncation_number_l2_from, truncation_number_l2_to + 1):
    co_l2[l - truncation_number_l2_from] = (2.0 * torch.log(b1) / torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) / (b1 ** l)
alpha_l2_chev = torch.zeros((truncation_number_l2, num_of_cheb_expansion_long_2), dtype=dtype, device=device)
alpha_l2_chev1 = torch.zeros((truncation_number_l2, num_of_cheb_expansion_long_2), dtype=dtype, device=device)
for l in range(truncation_number_l2_from, truncation_number_l2_to + 1):
    mat_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cheb_mat_data_Li', f'vectorE_{l}.mat'))
    alpha_l2_chev1[l - truncation_number_l2_from, :] = torch.tensor(mat_data['E_vector'], device=device, dtype=dtype).t()
for l in range(num_of_cheb_expansion_long_2):
    alpha_l2_chev[:, l] = alpha_l2_chev1[:, l] * co_l2
index_mat_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cheb_mat_data_Li', f'kl_index_long_3.mat'))
kl_index_long_3 = torch.from_numpy(index_mat_data['kl_index_long_3']).to(device).to(torch.int).squeeze()
num_of_cheb_expansion_long_3 = len(kl_index_long_3[:, 0])
co_l3 = torch.zeros(truncation_number_l3, dtype=dtype, device=device)
for l in range(truncation_number_l3_from, truncation_number_l3_to + 1):
    co_l3[l - truncation_number_l3_from] = (2.0 * torch.log(b1) / torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) / (b1 ** l)
alpha_l3_chev = torch.zeros((truncation_number_l3, num_of_cheb_expansion_long_3), dtype=dtype, device=device)
alpha_l3_chev1 = torch.zeros((truncation_number_l3, num_of_cheb_expansion_long_3), dtype=dtype, device=device)
for l in range(truncation_number_l3_from, truncation_number_l3_to + 1):
    mat_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'cheb_mat_data_Li', f'vectorE_{l}.mat'))
    alpha_l3_chev1[l - truncation_number_l3_from, :] = torch.tensor(mat_data['E_vector'], device=device, dtype=dtype).t()
for l in range(num_of_cheb_expansion_long_3):
    alpha_l3_chev[:, l] = alpha_l3_chev1[:, l] * co_l3
def generate_middle_term(t_from, t_to, p_lm):
    trunc_len = t_to - t_from + 1
    psi_x0 = torch.zeros((trunc_len, 1, p_lm, N_x), dtype=dtype, device=device)
    psi_x1 = torch.zeros((trunc_len, 1, p_lm, N_x), dtype=dtype, device=device)
    psi_y0 = torch.zeros((trunc_len, 1, p_lm, N_y), dtype=dtype, device=device)
    psi_y1 = torch.zeros((trunc_len, 1, p_lm, N_y), dtype=dtype, device=device)
    alpha_x = torch.zeros((trunc_len, p_lm), dtype=dtype, device=device)
    alpha_y = torch.zeros((trunc_len, p_lm), dtype=dtype, device=device)
    for l in range(t_from, t_to + 1):
        sv_vectors, sv_values = low_rank_svd_approximation(b1, l, p_lm, X, Y)
        idx = l - t_from
        psi_x0[idx, 0] = sv_vectors[0, :, :]
        psi_x1[idx, 0] = sv_vectors[1, :, :]
        psi_y0[idx, 0] = sv_vectors[0, :, :]
        psi_y1[idx, 0] = sv_vectors[1, :, :]
        alpha_x[idx] = (2.0 * torch.log(b1) / torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) * (b1 ** (-l)) * sv_values
        alpha_y[idx] = sv_values
    return psi_x0, psi_x1, psi_y0, psi_y1, alpha_x, alpha_y
psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1, alpha_psi_m1_x, alpha_psi_m1_y = generate_middle_term(truncation_number_lm1_from, truncation_number_lm1_to, p_lm1)
psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1, alpha_psi_m2_x, alpha_psi_m2_y = generate_middle_term(truncation_number_lm2_from, truncation_number_lm2_to, p_lm2)
psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1, alpha_psi_m3_x, alpha_psi_m3_y = generate_middle_term(truncation_number_lm3_from, truncation_number_lm3_to, p_lm3)
point_ssx = torch.zeros(truncation_number_ss, dtype=dtype, device=device)
point_ssy = torch.zeros(truncation_number_ss, dtype=dtype, device=device)
for l in range(truncation_number_ss_from, truncation_number_ss_to + 1):
    tensorx = 2.0 * torch.log(b1)
    tensory = (torch.sqrt(torch.tensor(2.0 * pi, device=device, dtype=dtype))) * (b1 ** l)
    point_ssx[l - truncation_number_ss_from] = tensorx
    point_ssy[l - truncation_number_ss_from] = tensory
# ********** Main Model **********
def bd(x): return (x - a_x) * (b_x - x)
def grad_bd(x): return -2 * x + a_x + b_x
def grad_grad_bd(x): return -2 * torch.ones_like(x)
activation = TNN_Sin
model_x0 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_x1 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_y0 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_y1 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_z0 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_z1 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_x2 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_y2 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
model_z2 = TNN(1, sizes, activation, bd=None, grad_bd=None, grad_grad_bd=None, scaling=False).to(dtype).to(device)
# ********** Loss Function Core **********
def build_matrices_logic(
    phi_x0, phi_x1, phi_x2, phi_y0, phi_y1, phi_y2, phi_z0, phi_z1, phi_z2,
    grad_phi_x0, grad_phi_x1, grad_phi_x2, grad_phi_y0, grad_phi_y1, grad_phi_y2, grad_phi_z0, grad_phi_z1, grad_phi_z2,
    phi_s_x0, phi_s_x1, phi_s_x2, phi_s_y0, phi_s_y1, phi_s_y2, phi_s_z0, phi_s_z1, phi_s_z2,
    w_x, w_y, w_z, w_0,
    ewald, ewald1, point_sx, point_sy, point_ssx, point_ssy,
    cheb_poly_cache, kl_index_long_1, kl_index_long_2, kl_index_long_3,
    alpha_l1_chev, alpha_l1_chev1, alpha_l2_chev, alpha_l2_chev1, alpha_l3_chev, alpha_l3_chev1,
    psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1, alpha_psi_m1_x, alpha_psi_m1_y,
    psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1, alpha_psi_m2_x, alpha_psi_m2_y,
    psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1, alpha_psi_m3_x, alpha_psi_m3_y,
    scale):
    Z = 3.0
    def overlap_1d(w, a, b):
        aw = a * w.view(1, 1, -1)
        return torch.prod(aw @ b.transpose(-1, -2), dim=0)
    def overlap_1d_cheb(w, cheb, a, b):
        aw_cheb = a.unsqueeze(0) * (w * cheb).view(cheb.shape[0], 1, 1, cheb.shape[1]) 
        res = aw_cheb @ b.unsqueeze(0).transpose(-1, -2)
        return torch.prod(res, dim=1)
    int_phi_x0_x0 = overlap_1d(w_x, phi_x0, phi_x0)
    int_phi_x1_x1 = overlap_1d(w_x, phi_x1, phi_x1)
    int_phi_x2_x2 = overlap_1d(w_x, phi_x2, phi_x2)
    int_phi_x0_x1 = overlap_1d(w_x, phi_x0, phi_x1)
    int_phi_x1_x0 = int_phi_x0_x1.t()
    int_phi_y0_y0 = overlap_1d(w_y, phi_y0, phi_y0)
    int_phi_y1_y1 = overlap_1d(w_y, phi_y1, phi_y1)
    int_phi_y2_y2 = overlap_1d(w_y, phi_y2, phi_y2)
    int_phi_y0_y1 = overlap_1d(w_y, phi_y0, phi_y1)
    int_phi_y1_y0 = int_phi_y0_y1.t()
    int_phi_z0_z0 = overlap_1d(w_z, phi_z0, phi_z0)
    int_phi_z1_z1 = overlap_1d(w_z, phi_z1, phi_z1)
    int_phi_z2_z2 = overlap_1d(w_z, phi_z2, phi_z2)
    int_phi_z0_z1 = overlap_1d(w_z, phi_z0, phi_z1)
    int_phi_z1_z0 = int_phi_z0_z1.t()
    S00 = int_phi_x0_x0 * int_phi_y0_y0 * int_phi_z0_z0
    S11 = int_phi_x1_x1 * int_phi_y1_y1 * int_phi_z1_z1
    S22 = int_phi_x2_x2 * int_phi_y2_y2 * int_phi_z2_z2
    S01 = int_phi_x0_x1 * int_phi_y0_y1 * int_phi_z0_z1
    S10 = int_phi_x1_x0 * int_phi_y1_y0 * int_phi_z1_z0
    raw_norm12 = S00 * S11 - S01 * S10
    partdown = 0.5 * raw_norm12 * S22
    int_grad_phi_x0_x0 = overlap_1d(w_x, grad_phi_x0, grad_phi_x0)
    int_grad_phi_x1_x1 = overlap_1d(w_x, grad_phi_x1, grad_phi_x1)
    int_grad_phi_x2_x2 = overlap_1d(w_x, grad_phi_x2, grad_phi_x2)
    int_grad_phi_x0_x1 = overlap_1d(w_x, grad_phi_x0, grad_phi_x1)
    int_grad_phi_x1_x0 = int_grad_phi_x0_x1.t()
    int_grad_phi_y0_y0 = overlap_1d(w_y, grad_phi_y0, grad_phi_y0)
    int_grad_phi_y1_y1 = overlap_1d(w_y, grad_phi_y1, grad_phi_y1)
    int_grad_phi_y2_y2 = overlap_1d(w_y, grad_phi_y2, grad_phi_y2)
    int_grad_phi_y0_y1 = overlap_1d(w_y, grad_phi_y0, grad_phi_y1)
    int_grad_phi_y1_y0 = int_grad_phi_y0_y1.t()
    int_grad_phi_z0_z0 = overlap_1d(w_z, grad_phi_z0, grad_phi_z0)
    int_grad_phi_z1_z1 = overlap_1d(w_z, grad_phi_z1, grad_phi_z1)
    int_grad_phi_z2_z2 = overlap_1d(w_z, grad_phi_z2, grad_phi_z2)
    int_grad_phi_z0_z1 = overlap_1d(w_z, grad_phi_z0, grad_phi_z1)
    int_grad_phi_z1_z0 = int_grad_phi_z0_z1.t()
    raw_grad12_x = (int_grad_phi_x0_x0 * int_phi_x1_x1 + int_phi_x0_x0 * int_grad_phi_x1_x1) * int_phi_y0_y0 * int_phi_y1_y1 * int_phi_z0_z0 * int_phi_z1_z1
    raw_grad12_x_ex = (int_grad_phi_x0_x1 * int_phi_x1_x0 + int_phi_x0_x1 * int_grad_phi_x1_x0) * int_phi_y0_y1 * int_phi_y1_y0 * int_phi_z0_z1 * int_phi_z1_z0
    raw_grad12_y = (int_grad_phi_y0_y0 * int_phi_y1_y1 + int_phi_y0_y0 * int_grad_phi_y1_y1) * int_phi_x0_x0 * int_phi_x1_x1 * int_phi_z0_z0 * int_phi_z1_z1
    raw_grad12_y_ex = (int_grad_phi_y0_y1 * int_phi_y1_y0 + int_phi_y0_y1 * int_grad_phi_y1_y0) * int_phi_x0_x1 * int_phi_x1_x0 * int_phi_z0_z1 * int_phi_z1_z0
    raw_grad12_z = (int_grad_phi_z0_z0 * int_phi_z1_z1 + int_phi_z0_z0 * int_grad_phi_z1_z1) * int_phi_x0_x0 * int_phi_x1_x1 * int_phi_y0_y0 * int_phi_y1_y1
    raw_grad12_z_ex = (int_grad_phi_z0_z1 * int_phi_z1_z0 + int_phi_z0_z1 * int_grad_phi_z1_z0) * int_phi_x0_x1 * int_phi_x1_x0 * int_phi_y0_y1 * int_phi_y1_y0
    raw_grad12 = (raw_grad12_x + raw_grad12_y + raw_grad12_z) - (raw_grad12_x_ex + raw_grad12_y_ex + raw_grad12_z_ex)
    grad3 = int_grad_phi_x2_x2 * int_phi_y2_y2 * int_phi_z2_z2 + int_phi_x2_x2 * int_grad_phi_y2_y2 * int_phi_z2_z2 + int_phi_x2_x2 * int_phi_y2_y2 * int_grad_phi_z2_z2
    partgrad = (0.25 / (scale ** 2)) * (raw_grad12 * S22 + raw_norm12 * grad3)
    def calc_V_element(phi_xL, phi_xR, phi_yL, phi_yR, phi_zL, phi_zR, phi_sxL, phi_sxR, phi_syL, phi_syR, phi_szL, phi_szR):
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
        term_long = torch.sum(op_long(w_x, ewald, phi_xL, phi_xR) * op_long(w_y, ewald1, phi_yL, phi_yR) * op_long(w_z, ewald1, phi_zL, phi_zR), dim=0)
        term_short = torch.sum(op_short(w_0, point_sx, phi_sxL, phi_sxR) * op_short(w_0, point_sy, phi_syL, phi_syR) * op_short(w_0, point_sy, phi_szL, phi_szR), dim=0)
        return term_long + term_short
    partr_r0_r0 = calc_V_element(phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0, phi_s_x0, phi_s_x0, phi_s_y0, phi_s_y0, phi_s_z0, phi_s_z0)
    partr_r1_r1 = calc_V_element(phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1, phi_s_x1, phi_s_x1, phi_s_y1, phi_s_y1, phi_s_z1, phi_s_z1)
    partr_r2_r2 = calc_V_element(phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2, phi_s_x2, phi_s_x2, phi_s_y2, phi_s_y2, phi_s_z2, phi_s_z2)
    partr_r0_r1 = calc_V_element(phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1, phi_s_x0, phi_s_x1, phi_s_y0, phi_s_y1, phi_s_z0, phi_s_z1)
    partr_r1_r0 = partr_r0_r1.t()
    partr12 = 0.5 * (-(Z / scale) * ((partr_r0_r0 * S11 + partr_r1_r1 * S00) - (partr_r0_r1 * S10 + partr_r1_r0 * S01))) * S22
    partr3 = 0.5 * raw_norm12 * (-(Z / scale) * partr_r2_r2)
    partr = partr12 + partr3
    int_cheb_phi_x0_x0 = overlap_1d_cheb(w_x, cheb_poly_cache, phi_x0, phi_x0)
    int_cheb_phi_x1_x1 = overlap_1d_cheb(w_x, cheb_poly_cache, phi_x1, phi_x1)
    int_cheb_phi_x2_x2 = overlap_1d_cheb(w_x, cheb_poly_cache, phi_x2, phi_x2)
    int_cheb_phi_x0_x1 = overlap_1d_cheb(w_x, cheb_poly_cache, phi_x0, phi_x1)
    int_cheb_phi_x1_x0 = int_cheb_phi_x0_x1.transpose(1, 2)
    int_cheb_phi_y0_y0 = overlap_1d_cheb(w_y, cheb_poly_cache, phi_y0, phi_y0)
    int_cheb_phi_y1_y1 = overlap_1d_cheb(w_y, cheb_poly_cache, phi_y1, phi_y1)
    int_cheb_phi_y2_y2 = overlap_1d_cheb(w_y, cheb_poly_cache, phi_y2, phi_y2)
    int_cheb_phi_y0_y1 = overlap_1d_cheb(w_y, cheb_poly_cache, phi_y0, phi_y1)
    int_cheb_phi_y1_y0 = int_cheb_phi_y0_y1.transpose(1, 2)
    int_cheb_phi_z0_z0 = overlap_1d_cheb(w_z, cheb_poly_cache, phi_z0, phi_z0)
    int_cheb_phi_z1_z1 = overlap_1d_cheb(w_z, cheb_poly_cache, phi_z1, phi_z1)
    int_cheb_phi_z2_z2 = overlap_1d_cheb(w_z, cheb_poly_cache, phi_z2, phi_z2)
    int_cheb_phi_z0_z1 = overlap_1d_cheb(w_z, cheb_poly_cache, phi_z0, phi_z1)
    int_cheb_phi_z1_z0 = int_cheb_phi_z0_z1.transpose(1, 2)
    def calc_rr_long(ix_ab, ix_cd, iy_ab, iy_cd, iz_ab, iz_cd):
        def one_block(kl_index, alpha_x, alpha_y):
            ix_term = ix_ab[kl_index[:, 0]] * ix_cd[kl_index[:, 1]] 
            iy_term = iy_ab[kl_index[:, 0]] * iy_cd[kl_index[:, 1]] 
            iz_term = iz_ab[kl_index[:, 0]] * iz_cd[kl_index[:, 1]] 
            P, Q = ix_term.shape[1], ix_term.shape[2]
            term_x = (alpha_x @ ix_term.reshape(ix_term.shape[0], -1)).reshape(-1, P, Q)
            term_y = (alpha_y @ iy_term.reshape(iy_term.shape[0], -1)).reshape(-1, P, Q)
            term_z = (alpha_y @ iz_term.reshape(iz_term.shape[0], -1)).reshape(-1, P, Q)
            return torch.sum(term_x * term_y * term_z, dim=0)
        return one_block(kl_index_long_1, alpha_l1_chev, alpha_l1_chev1) + one_block(kl_index_long_2, alpha_l2_chev, alpha_l2_chev1) + one_block(kl_index_long_3, alpha_l3_chev, alpha_l3_chev1)
    rr12_long_dir = calc_rr_long(int_cheb_phi_x0_x0, int_cheb_phi_x1_x1, int_cheb_phi_y0_y0, int_cheb_phi_y1_y1, int_cheb_phi_z0_z0, int_cheb_phi_z1_z1)
    rr12_long_ex = calc_rr_long(int_cheb_phi_x0_x1, int_cheb_phi_x1_x0, int_cheb_phi_y0_y1, int_cheb_phi_y1_y0, int_cheb_phi_z0_z1, int_cheb_phi_z1_z0)
    partrr12_long = 0.5 * (rr12_long_dir - rr12_long_ex) * S22
    rr02_long = calc_rr_long(int_cheb_phi_x0_x0, int_cheb_phi_x2_x2, int_cheb_phi_y0_y0, int_cheb_phi_y2_y2, int_cheb_phi_z0_z0, int_cheb_phi_z2_z2)
    rr12_long = calc_rr_long(int_cheb_phi_x1_x1, int_cheb_phi_x2_x2, int_cheb_phi_y1_y1, int_cheb_phi_y2_y2, int_cheb_phi_z1_z1, int_cheb_phi_z2_z2)
    rr02_12_long = calc_rr_long(int_cheb_phi_x0_x1, int_cheb_phi_x2_x2, int_cheb_phi_y0_y1, int_cheb_phi_y2_y2, int_cheb_phi_z0_z1, int_cheb_phi_z2_z2)
    rr12_02_long = calc_rr_long(int_cheb_phi_x1_x0, int_cheb_phi_x2_x2, int_cheb_phi_y1_y0, int_cheb_phi_y2_y2, int_cheb_phi_z1_z0, int_cheb_phi_z2_z2)
    partrr13_23_long = 0.5 * (rr02_long * S11 + rr12_long * S00 - rr02_12_long * S10 - rr12_02_long * S01)
    def calc_middle_interaction(px_L1, px_R1, py_L1, py_R1, pz_L1, pz_R1, px_L2, px_R2, py_L2, py_R2, pz_L2, pz_R2):
        proj_alpha = lambda w, alpha, psi, pL, pR: torch.prod(torch.einsum('n,tp,tdpn,dqn,dln->tdpql', w, alpha, psi, pL, pR), dim=1)
        proj_pure = lambda w, psi, pL, pR: torch.prod(torch.einsum('n,tdpn,dqn,dln->tdpql', w, psi, pL, pR), dim=1)
        def calc_single_mode(alpha_x, alpha_y, psi_x0, psi_x1, psi_y0, psi_y1):
            term_x = torch.sum(proj_alpha(w_x, alpha_x, psi_x0, px_L1, px_R1) * proj_pure(w_x, psi_x1, px_L2, px_R2), dim=1)
            term_y = torch.sum(proj_alpha(w_y, alpha_y, psi_y0, py_L1, py_R1) * proj_pure(w_y, psi_y1, py_L2, py_R2), dim=1)
            term_z = torch.sum(proj_alpha(w_z, alpha_y, psi_y0, pz_L1, pz_R1) * proj_pure(w_z, psi_y1, pz_L2, pz_R2), dim=1)
            return torch.sum(term_x * term_y * term_z, dim=0)
        res_m1 = calc_single_mode(alpha_psi_m1_x, alpha_psi_m1_y, psi_m1_x0, psi_m1_x1, psi_m1_y0, psi_m1_y1)
        res_m2 = calc_single_mode(alpha_psi_m2_x, alpha_psi_m2_y, psi_m2_x0, psi_m2_x1, psi_m2_y0, psi_m2_y1)
        res_m3 = calc_single_mode(alpha_psi_m3_x, alpha_psi_m3_y, psi_m3_x0, psi_m3_x1, psi_m3_y0, psi_m3_y1)
        return res_m1 + res_m2 + res_m3
    mid_12_direct = calc_middle_interaction(phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0, phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1)
    mid_12_exchange = calc_middle_interaction(phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1, phi_x1, phi_x0, phi_y1, phi_y0, phi_z1, phi_z0)
    partrr12_middle = 0.5 * (mid_12_direct - mid_12_exchange) * S22
    mid_02 = calc_middle_interaction(phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0, phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2)
    mid_12 = calc_middle_interaction(phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1, phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2)
    mid_02_12 = calc_middle_interaction(phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1, phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2)
    mid_12_02 = calc_middle_interaction(phi_x1, phi_x0, phi_y1, phi_y0, phi_z1, phi_z0, phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2)
    partrr13_23_middle = 0.5 * (mid_02 * S11 + mid_12 * S00 - mid_02_12 * S10 - mid_12_02 * S01)
    def calc_short_interaction(px_i, px_j, py_i, py_j, pz_i, pz_j, px_k, px_l, py_k, py_l, pz_k, pz_l):
        def calc_term(w, pt, p_i, p_j, p_k, p_l):
            pA = p_i * p_k
            pB = p_j * p_l
            pA_w = pA * w.view(1, 1, -1)
            core = pA_w @ pB.transpose(-1, -2)
            res = pt.view(-1, 1, 1, 1) * core.unsqueeze(0)
            return torch.prod(res, dim=1)
        term_x = calc_term(w_x, point_ssx, px_i, px_j, px_k, px_l)
        term_y = calc_term(w_y, point_ssy, py_i, py_j, py_k, py_l)
        term_z = calc_term(w_z, point_ssy, pz_i, pz_j, pz_k, pz_l)
        return torch.sum(term_x * term_y * term_z, dim=0)
    short_12_direct = calc_short_interaction(phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0,phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1,)
    short_12_exchange = calc_short_interaction(phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1,phi_x1, phi_x0, phi_y1, phi_y0, phi_z1, phi_z0,)
    partrr12_short = 0.5 * (short_12_direct - short_12_exchange) * S22
    short_02 = calc_short_interaction(phi_x0, phi_x0, phi_y0, phi_y0, phi_z0, phi_z0,phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2,)
    short_12 = calc_short_interaction(phi_x1, phi_x1, phi_y1, phi_y1, phi_z1, phi_z1,phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2,)
    short_02_12 = calc_short_interaction(phi_x0, phi_x1, phi_y0, phi_y1, phi_z0, phi_z1,phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2,)
    short_12_02 = calc_short_interaction(phi_x1, phi_x0, phi_y1, phi_y0, phi_z1, phi_z0,phi_x2, phi_x2, phi_y2, phi_y2, phi_z2, phi_z2,)
    partrr13_23_short = 0.5 * (short_02 * S11 + short_12 * S00 - short_02_12 * S10 - short_12_02 * S01)
    partrr = (1 / scale) * (partrr12_long + partrr12_middle + partrr12_short+ partrr13_23_long + partrr13_23_middle + partrr13_23_short)
    del partrr12_long, partrr12_middle, partrr12_short
    del partrr13_23_long, partrr13_23_middle, partrr13_23_short
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
# build_matrices_logic = torch.compile(build_matrices_logic, backend="inductor",mode="max-autotune",dynamic=False,fullgraph=False)

def criterion(model_x0, model_x1, model_x2, model_y0, model_y1, model_y2, model_z0, model_z1, model_z2):
    def eval_axis(model, w, point):
        phi, grad_phi = model(w, point * scale, need_grad=1, normed=False)
        phi_norm = torch.sqrt(torch.sum(w * phi ** 2, dim=-1)).unsqueeze(dim=-1)
        return phi / phi_norm, grad_phi * scale / phi_norm, phi_norm
    phi_x0, grad_phi_x0, phi_x0_norm = eval_axis(model_x0, w_x, point_x)
    phi_x1, grad_phi_x1, phi_x1_norm = eval_axis(model_x1, w_x, point_x)
    phi_x2, grad_phi_x2, phi_x2_norm = eval_axis(model_x2, w_x, point_x)
    phi_y0, grad_phi_y0, phi_y0_norm = eval_axis(model_y0, w_y, point_y)
    phi_y1, grad_phi_y1, phi_y1_norm = eval_axis(model_y1, w_y, point_y)
    phi_y2, grad_phi_y2, phi_y2_norm = eval_axis(model_y2, w_y, point_y)
    phi_z0, grad_phi_z0, phi_z0_norm = eval_axis(model_z0, w_z, point_z)
    phi_z1, grad_phi_z1, phi_z1_norm = eval_axis(model_z1, w_z, point_z)
    phi_z2, grad_phi_z2, phi_z2_norm = eval_axis(model_z2, w_z, point_z)
    phi_s_x0 = model_x0(w_0, point0 * scale, need_grad=0, normed=False) / phi_x0_norm
    phi_s_x1 = model_x1(w_0, point0 * scale, need_grad=0, normed=False) / phi_x1_norm
    phi_s_x2 = model_x2(w_0, point0 * scale, need_grad=0, normed=False) / phi_x2_norm
    phi_s_y0 = model_y0(w_0, point0 * scale, need_grad=0, normed=False) / phi_y0_norm
    phi_s_y1 = model_y1(w_0, point0 * scale, need_grad=0, normed=False) / phi_y1_norm
    phi_s_y2 = model_y2(w_0, point0 * scale, need_grad=0, normed=False) / phi_y2_norm
    phi_s_z0 = model_z0(w_0, point0 * scale, need_grad=0, normed=False) / phi_z0_norm
    phi_s_z1 = model_z1(w_0, point0 * scale, need_grad=0, normed=False) / phi_z1_norm
    phi_s_z2 = model_z2(w_0, point0 * scale, need_grad=0, normed=False) / phi_z2_norm
    partdown, partgrad, partr, partrr = build_matrices_logic(
        phi_x0, phi_x1, phi_x2, phi_y0, phi_y1, phi_y2, phi_z0, phi_z1, phi_z2,
        grad_phi_x0, grad_phi_x1, grad_phi_x2, grad_phi_y0, grad_phi_y1, grad_phi_y2, grad_phi_z0, grad_phi_z1, grad_phi_z2,
        phi_s_x0, phi_s_x1, phi_s_x2, phi_s_y0, phi_s_y1, phi_s_y2, phi_s_z0, phi_s_z1, phi_s_z2,
        w_x, w_y, w_z, w_0,
        ewald, ewald1, point_sx, point_sy, point_ssx, point_ssy,
        cheb_poly_cache, kl_index_long_1, kl_index_long_2, kl_index_long_3,
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
        partgrad_val = torch.sum(alpha_outer * partgrad)
        partr_val = torch.sum(alpha_outer * partr)
        partrr_val = torch.sum(alpha_outer * partrr)
        partdown_val = torch.sum(alpha_outer * partdown)
        lossE_val = (partgrad_val + partr_val + partrr_val) / partdown_val
    return loss, lossE_val, partgrad_val, partr_val, partrr_val, partdown_val, alpha 

# ********** Multi-stage Training Adam **********
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
lr_list = [1e-2, 5e-6, 5e-7, 3e-8, 3e-9]
epoch_list = [24000, 18000, 11000, 14000, 14000]
lossbest, errorbest = [], []
current_min_error = float('inf')
print_every = 100
best_model_states = None
best_alpha = None
reference_energy = -7.47806032
print(f'reference_energy={reference_energy:.17g} hartree')
def get_optimizer(lr):
    return optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(model_x0.parameters(), model_y0.parameters(), model_z0.parameters(), model_x1.parameters(), model_y1.parameters(), model_z1.parameters(), model_x2.parameters(), model_y2.parameters(), model_z2.parameters())),lr=lr)
starttime = time.time()
trainable_params = sum(p.numel()for p in itertools.chain(model_x0.parameters(),model_y0.parameters(),model_z0.parameters(),model_x1.parameters(),model_y1.parameters(),model_z1.parameters(),model_x2.parameters(),model_y2.parameters(),model_z2.parameters())if p.requires_grad)
for stage, (learning_rate, epochs) in enumerate(zip(lr_list, epoch_list), 1):
    if best_model_states is not None:
        model_x0.load_state_dict(best_model_states['x0'])
        model_x1.load_state_dict(best_model_states['x1'])
        model_x2.load_state_dict(best_model_states['x2'])
        model_y0.load_state_dict(best_model_states['y0'])
        model_y1.load_state_dict(best_model_states['y1'])
        model_y2.load_state_dict(best_model_states['y2'])
        model_z0.load_state_dict(best_model_states['z0'])
        model_z1.load_state_dict(best_model_states['z1'])
        model_z2.load_state_dict(best_model_states['z2'])
        alpha.data.copy_(best_alpha)
    optimizer = get_optimizer(learning_rate)
    for e in range(epochs):
        loss, lossE, partgrad, part2, part3, partdown, alpha = criterion(model_x0, model_x1, model_x2, model_y0, model_y1, model_y2, model_z0, model_z1, model_z2)
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
                    'y0': copy.deepcopy(model_y0.state_dict()),
                    'y1': copy.deepcopy(model_y1.state_dict()),
                    'y2': copy.deepcopy(model_y2.state_dict()),
                    'z0': copy.deepcopy(model_z0.state_dict()),
                    'z1': copy.deepcopy(model_z1.state_dict()),
                    'z2': copy.deepcopy(model_z2.state_dict())
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