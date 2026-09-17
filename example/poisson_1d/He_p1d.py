import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.optim as optim
import itertools
import time
import copy
from scipy.io import loadmat
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from sogtnn.tnn_larger import *
from sogtnn.quad_larger import *
from sogtnn.integration import *
torch.set_printoptions(precision=16)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
dtype = torch.float64
device = 'cuda:0'
r_c = 10.
a = -1.
b = 1.
dim = 1
Z = 2.
def generate_shen_data(m, n, r, device='cpu', dtype=torch.float64):
    xi, w_std = np.polynomial.legendre.leggauss(n)
    total_points = m * n
    Nodes = np.zeros(total_points)
    Weights = np.zeros(total_points)
    m_half = m // 2
    edges_left = np.linspace(-1, 0, m_half + 1)
    for i in range(m_half):
        a = edges_left[i]
        b = edges_left[i+1]
        jac = (b - a) / 2.0
        center = (b + a) / 2.0
        idx_start = i * n
        idx_end = (i + 1) * n
        Nodes[idx_start:idx_end] = jac * xi + center
        Weights[idx_start:idx_end] = jac * w_std
    edges_right = np.linspace(0, 1, m_half + 1)
    offset = m_half * n
    for i in range(m_half):
        a = edges_right[i]
        b = edges_right[i+1]
        jac = (b - a) / 2.0
        center = (b + a) / 2.0
        idx_start = offset + i * n
        idx_end = offset + (i + 1) * n
        Nodes[idx_start:idx_end] = jac * xi + center
        Weights[idx_start:idx_end] = jac * w_std
    N = 2 * r + 1
    BasisMatrix = np.zeros((N, total_points))
    indices_map = np.arange(-r, r + 1) 
    s_vals = np.zeros_like(Nodes)
    # x > 0: s = 2x - 1
    mask_pos = Nodes > 0
    s_vals[mask_pos] = 2 * Nodes[mask_pos] - 1
    # x < 0: s = -1 - 2x
    mask_neg = Nodes < 0
    s_vals[mask_neg] = -1 - 2 * Nodes[mask_neg]
    max_deg = r + 1
    L_val_mat = np.zeros((max_deg + 1, total_points))
    # Degree 0
    L_val_mat[0, :] = 1.0
    # Degree 1
    if max_deg >= 1:
        L_val_mat[1, :] = s_vals
    for k in range(1, max_deg):
        L_val_mat[k+1, :] = ((2*k + 1) * s_vals * L_val_mat[k, :] - k * L_val_mat[k-1, :]) / (k + 1)
    for row in range(N):
        k = indices_map[row]
        if k == 0:
            BasisMatrix[row, :] = 1.0 - np.abs(Nodes)
        elif k > 0:
            # L_{k-1} - L_{k+1} (仅 x > 0)
            val = L_val_mat[k-1, :] - L_val_mat[k+1, :]
            BasisMatrix[row, mask_pos] = val[mask_pos]
        else: # k < 0
            # L_{|k|-1} - L_{|k|+1} (仅 x < 0)
            k_abs = abs(k)
            val = L_val_mat[k_abs-1, :] - L_val_mat[k_abs+1, :]
            BasisMatrix[row, mask_neg] = val[mask_neg]
    T_diag = np.zeros(N)
    for row in range(N):
        k = indices_map[row]
        if k == 0:
            T_diag[row] = 2.0
        else:
            T_diag[row] = 4.0 * (2.0 * abs(k) + 1.0)
    sigma_vec = - 1.0 / T_diag
    point_r = torch.tensor(Nodes, device=device, dtype=dtype)
    w_r = torch.tensor(Weights, device=device, dtype=dtype)
    Shen_poly = torch.tensor(BasisMatrix, device=device, dtype=dtype)
    sigma = torch.tensor(sigma_vec, device=device, dtype=dtype)
    return point_r, w_r, Shen_poly, sigma
max_mode = 256
# Quadrature setup
quad = 256
n = 10
point_r, w_r, Shen_poly, sigma = generate_shen_data(n, quad, max_mode, device=device, dtype=dtype)
K_r = len(point_r)
V_r = torch.abs(point_r)
V_r_left = 1.0 + point_r
V_r_right = 1.0 - point_r
max_rank = Shen_poly.shape[0]
V_Shen_w = Shen_poly * w_r 
p = 16
sizes = [1, 64, 128, 256, p]
def bd(x): return (x-a) * (b - x)
def grad_bd(x): return a+b-2*x
activation = TNN_Sin
model_r1 = TNN(1, sizes, activation, bd=bd, grad_bd=grad_bd, scaling=False).to(dtype).to(device)
model_r2 = TNN(1, sizes, activation, bd=bd, grad_bd=grad_bd, scaling=False).to(dtype).to(device)
print("Compiling models...")
# model_r1 = torch.compile(model_r1)
# model_r2 = torch.compile(model_r2)
def build_matrices_logic(phi_r1, phi_r2, grad_r1, grad_r2, w_r, V_r, V_Shen_w, sigma, point_r, V_r_left, V_r_right):
    p, n = phi_r1.shape
    sqrt_w = torch.sqrt(w_r)
    phi_r1_w = phi_r1 * sqrt_w
    phi_r2_w = phi_r2 * sqrt_w
    # Overlap Matrix (S)
    M1 = phi_r1_w @ phi_r1_w.t()  # [p, p]
    M2 = phi_r2_w @ phi_r2_w.t()  # [p, p]
    S_mat = M1 * M2
    # Kinetic Matrix (T)
    grad_r1_w = grad_r1 * sqrt_w
    grad_r2_w = grad_r2 * sqrt_w
    K1 = grad_r1_w @ grad_r1_w.t()
    K2 = grad_r2_w @ grad_r2_w.t()
    T_mat = (K1 * M2 + M1 * K2) * (0.5 / (r_c**2)) 
    w_V = w_r * V_r
    V1 = (phi_r1 * w_V) @ phi_r1.t()
    V2 = (phi_r2 * w_V) @ phi_r2.t()
    V_ie_mat = (V1 * M2 + M1 * V2) * (Z * r_c) 
    rho2 = (phi_r2.unsqueeze(1) * phi_r2.unsqueeze(0)).reshape(p*p, n)
    spectral_basis_weighted = V_Shen_w * sigma.unsqueeze(1)
    alpha_coeffs = (spectral_basis_weighted @ rho2.t()) * 2.0 
    rho1 = (phi_r1.unsqueeze(1) * phi_r1.unsqueeze(0)).reshape(p*p, n)
    V_recon = alpha_coeffs.t() @ V_Shen_w # [P^2, R] @ [R, n] -> [P^2, n]
    partV = torch.sum(V_recon * rho1, dim=1).reshape(p, p)
    w_V_left = w_r * V_r_left
    w_V_right = w_r * V_r_right
    W_left_mat = (phi_r2 * w_V_left) @ phi_r2.t()
    W_right_mat = (phi_r2 * w_V_right) @ phi_r2.t()
    diff_W = (W_right_mat - W_left_mat) * 0.5
    avg_W = (W_left_mat + W_right_mat) * 0.5
    w_point = w_r * point_r
    X_mat_1 = (phi_r1 * w_point) @ phi_r1.t()
    partW_x = diff_W * X_mat_1
    partW_b = avg_W * M1
    V_ee_mat = (partV + partW_x + partW_b) * (-r_c) 
    return S_mat, T_mat, V_ie_mat, V_ee_mat

def solve_eigenvalue_problem(M, T, Vs, Vd, tol=1e-15):
    H = Vd + Vs + T
    H = 0.5 * (H + H.T)
    M = 0.5 * (M + M.T)
    s, U = torch.linalg.eigh(M) 
    mask = s > tol
    s_eff = s[mask]      # [k]
    U_eff = U[:, mask]   # [p, k]
    inv_sqrt_s = torch.rsqrt(s_eff)
    X = U_eff * inv_sqrt_s.unsqueeze(0)
    H_prime = X.T @ H @ X
    E_eff, V_eff = torch.linalg.eigh(H_prime)
    ground_E = E_eff[0]        
    v_ground = V_eff[:, 0]     
    ground_alpha = X @ v_ground
    return ground_E, ground_alpha

# print("Compiling Matrix Construction (Mode: reduce-overhead)...")
# build_matrices_logic = torch.compile(build_matrices_logic, mode='max-autotune', fullgraph=True)

def criterion(model_r1, model_r2):
    # 1. Forward Pass
    phi_r1_raw, grad_phi_r1_raw = model_r1(w_r, point_r, need_grad=True, normed=True)
    phi_r2_raw, grad_phi_r2_raw = model_r2(w_r, point_r, need_grad=True, normed=True)
    phi_r1 = phi_r1_raw.squeeze(0)          # [p, n]
    grad_phi_r1 = grad_phi_r1_raw.squeeze(0)
    phi_r2 = phi_r2_raw.squeeze(0)
    grad_phi_r2 = grad_phi_r2_raw.squeeze(0)
    # 2. Matrix Build (Compiled)
    S_mat, T_mat, V_ie_mat, V_ee_mat = build_matrices_logic(phi_r1, phi_r2, grad_phi_r1, grad_phi_r2, w_r, V_r, V_Shen_w, sigma, point_r, V_r_left, V_r_right)
    # 3. Solver (Eager)
    loss, alpha = solve_eigenvalue_problem(S_mat, T_mat, V_ie_mat, V_ee_mat)
    # 4. Metrics
    with torch.no_grad():
        P = torch.outer(alpha, alpha)
        M_val = torch.sum(P * S_mat)
        T_val = torch.sum(P * T_mat) / M_val
        V_ie_val = torch.sum(P * V_ie_mat) / M_val
        V_ee_val = torch.sum(P * V_ee_mat) / M_val
    return loss, M_val, T_val, V_ie_val, V_ee_val, alpha

# ********** Training Process (RAdam + Cosine Scheduler with warmrestarts) **********
# --- Configuration ---
phase1_lr = 1e-3
phase1_epochs = 635000     
print_every = 100
optimizer_choice = 'RAdam' 
print(f"{'='*20} PHASE 1: Exploration (0 - {phase1_epochs}) {'='*20}")
# --- Optimizer Setup ---
params = filter(lambda p: p.requires_grad, itertools.chain(model_r1.parameters(), model_r2.parameters()))
if optimizer_choice == 'AdamW':
    optimizer = optim.AdamW(params, lr=phase1_lr)
elif optimizer_choice == 'RAdam':
    optimizer = optim.RAdam(params, lr=phase1_lr)
else:
    optimizer = optim.Adam(params, lr=phase1_lr)
# --- Scheduler Setup: Cosine Annealing with Warm Restarts ---
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2, eta_min=1e-5)
# --- Best Model Tracker ---
min_loss = float('inf')
best_model_r1_state = None
best_model_r2_state = None
# training
starttime = time.time()
for e in range(phase1_epochs):
    loss, M, T, V_ie, V_ee, alpha = criterion(model_r1, model_r2)
    # Check for new minimum loss
    current_loss = loss.item()
    if (current_loss < min_loss) and ((e + 1) % 5 == 0):
        min_loss = current_loss  # Update the minimum loss
        best_model_r1_state = copy.deepcopy(model_r1.state_dict())
        best_model_r2_state = copy.deepcopy(model_r2.state_dict())
        # Print detailed info when new minimum loss is achieved
        print('*' * 40)
        print(f"{'NEW MIN LOSS':^40}")
        print('*' * 40)
        print('{:<9}{:<25}'.format('epoch = ', e + 1))
        print('{:<9}{:<25}'.format('loss = ', current_loss))
        print('*' * 40)
    # optimization process
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    # Regular periodic printing
    if (e + 1) % print_every == 0:
        print('*' * 40)
        print('{:<9}{:<25}'.format('epoch = ', e + 1))
        print('{:<9}{:<25}'.format('loss = ', current_loss))
endtime = time.time()
print('*' * 40)
print('Done!')
print('Training took: {:.2f}s'.format(endtime - starttime))
print(f"Phase 1 Finished. Best Loss Found: {min_loss:.16f}")

# ********** Training Process (RAdam + Cosine Scheduler) **********
# --- Configuration ---
phase2_lr = 1e-4 
phase2_epochs = 160000     
print_every = 100
optimizer_choice = 'RAdam' 
print(f"\n{'='*20} PHASE 2: Convergence {phase2_epochs} {'='*20}")
# Back to best model from previous training
print("Loading BEST model from Phase 1...")
model_r1.load_state_dict(best_model_r1_state)
model_r2.load_state_dict(best_model_r2_state)
# --- Optimizer Setup ---
params = filter(lambda p: p.requires_grad, itertools.chain(model_r1.parameters(), model_r2.parameters()))
if optimizer_choice == 'AdamW':
    optimizer = optim.AdamW(params, lr=phase2_lr)
elif optimizer_choice == 'RAdam':
    optimizer = optim.RAdam(params, lr=phase2_lr)
else:
    optimizer = optim.Adam(params, lr=phase2_lr)
# --- Scheduler Setup: Cosine Annealing with Warm Restarts ---
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-7)
# --- Best Model Tracker ---
min_loss = float('inf')
best_model_r1_state = None
best_model_r2_state = None
# training
starttime = time.time()
for e in range(phase2_epochs):
    e = e + phase1_epochs
    loss, M, T, V_ie, V_ee, alpha = criterion(model_r1, model_r2)
    # Check for new minimum loss
    current_loss = loss.item()
    if (current_loss < min_loss) and ((e + 1) % 5 == 0):
        min_loss = current_loss  # Update the minimum loss
        best_model_r1_state = copy.deepcopy(model_r1.state_dict())
        best_model_r2_state = copy.deepcopy(model_r2.state_dict())
        # Print detailed info when new minimum loss is achieved
        print('*' * 40)
        print(f"{'NEW MIN LOSS':^40}")
        print('*' * 40)
        print('{:<9}{:<25}'.format('epoch = ', e + 1))
        print('{:<9}{:<25}'.format('loss = ', current_loss))
        print('*' * 40)
    # optimization process
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    # Regular periodic printing
    if (e + 1) % print_every == 0:
        print('*' * 40)
        print('{:<9}{:<25}'.format('epoch = ', e + 1))
        print('{:<9}{:<25}'.format('loss = ', current_loss))
endtime = time.time()
print('*' * 40)
print('Done!')
print('Training took: {:.2f}s'.format(endtime - starttime))
print(f"Phase 2 Finished. Global Best Loss: {min_loss:.16f}")

# ********** training process LBFGS **********
model_r1.load_state_dict(best_model_r1_state)
model_r2.load_state_dict(best_model_r2_state)
# parameters
lr = 0.01
epoch = 5000
print_every = 100
save = True
# optimizer used
optimizer = torch.optim.LBFGS(itertools.chain(model_r1.parameters(), model_r2.parameters()), lr=lr, max_iter=128, history_size=256, tolerance_grad=1e-13, tolerance_change=1e-15)
# training
for e in range(epoch):
    # e = e + epochs
    e = e + phase1_epochs + phase2_epochs 
    # initial info
    def closure():
        loss, M, T, V_ie, V_ee, alpha = criterion(model_r1, model_r2)
        optimizer.zero_grad()
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    # post process
    if (e+1) % print_every == 0:
        print('*' * 40)
        print('{:<9}{:<25}'.format('epoch = ', e + 1))
        print('{:<9}{:<25}'.format('loss = ', loss.item()))
print('*' * 40)
print('Done!')
print('Training took: {}s'.format(endtime - starttime))