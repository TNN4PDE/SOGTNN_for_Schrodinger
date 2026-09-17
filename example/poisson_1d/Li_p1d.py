import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
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
N = 3
N_up = 2
N_down = 1
Z = 3.
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
quad = 256
n = 10
max_mode = 256
point_r, w_r, Shen_poly, sigma = generate_shen_data(n, quad, max_mode, device=device, dtype=dtype)
K = len(point_r)
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
model_up = Multi_TNN(N_up, dim, sizes, activation, bd=bd, grad_bd=grad_bd, scaling=False).to(dtype).to(device)
model_down = Multi_TNN(N_down, dim, sizes, activation, bd=bd, grad_bd=grad_bd, scaling=False).to(dtype).to(device)
# model_up = torch.compile(model_up)
# model_down = torch.compile(model_down)
def build_matrices_logic(phi_up, grad_phi_up, phi_down, grad_phi_down, w_r, V_r, V_Shen_w, sigma, point_r, V_r_left, V_r_right):
    # --- 1. Mass Matrix ---
    # [p, N, K] -> [p, p, N, N]
    M_up = torch.einsum('k, pnk, qmk -> pqnm', w_r, phi_up, phi_up)
    M_down = torch.einsum('k, pnk, qmk -> pqnm', w_r, phi_down, phi_down)
    det_M_up = torch.linalg.det(M_up)      # [p, p]
    det_M_down = torch.linalg.det(M_down)  # [p, p]
    M = det_M_up * det_M_down
    # --- 2. Biorthogonalization ---
    phi_up_exp = phi_up.unsqueeze(1).expand(p, p, N_up, K)
    grad_phi_up_exp = grad_phi_up.unsqueeze(1).expand(p, p, N_up, K)
    phi_down_exp = phi_down.unsqueeze(1).expand(p, p, N_down, K)
    grad_phi_down_exp = grad_phi_down.unsqueeze(1).expand(p, p, N_down, K)
    theta_up = torch.linalg.solve(M_up, phi_up_exp)
    grad_theta_up = torch.linalg.solve(M_up, grad_phi_up_exp)
    theta_down = torch.linalg.solve(M_down, phi_down_exp)
    grad_theta_down = torch.linalg.solve(M_down, grad_phi_down_exp)
    # --- 3. Kinetic Energy ---
    T_up = torch.einsum('k, pqnk, qnk -> pq', w_r, grad_theta_up, grad_phi_up)
    T_down = torch.einsum('k, pqnk, qnk -> pq', w_r, grad_theta_down, grad_phi_down)
    T = (0.5/(r_c**2)) * M * (T_up + T_down)
    # --- 4. Single Electron Potential ---
    Vs_up = torch.einsum('k, pqnk, qnk -> pq', w_r * V_r, theta_up, phi_up)
    Vs_down = torch.einsum('k, pqnk, qnk -> pq', w_r * V_r, theta_down, phi_down)
    Vs = (r_c * Z) * M * (Vs_up + Vs_down)
    # --- 5. Double Electron Potential  ---
    alpha_up = torch.einsum('r, rk, pqmk, qnk -> pqmnr', sigma, V_Shen_w, 2*theta_up, phi_up)
    alpha_down = torch.einsum('r, rk, pqmk, qnk -> pqmnr', sigma, V_Shen_w, 2*theta_down, phi_down)
    U_up_left = torch.einsum('k, pqmk, qnk -> pqmn', w_r * V_r_left, theta_up, phi_up)
    U_up_right = torch.einsum('k, pqmk, qnk -> pqmn', w_r * V_r_right, theta_up, phi_up)
    U_down_left = torch.einsum('k, pqmk, qnk -> pqmn', w_r * V_r_left, theta_down, phi_down)
    U_down_right = torch.einsum('k, pqmk, qnk -> pqmn', w_r * V_r_right, theta_down, phi_down)
    # trace
    alpha_up_tr = alpha_up.diagonal(dim1=2, dim2=3).movedim(-1, 2) 
    alpha_down_tr = alpha_down.diagonal(dim1=2, dim2=3).movedim(-1, 2)
    U_up_left_tr = U_up_left.diagonal(dim1=2, dim2=3).movedim(-1, 2)
    U_up_right_tr = U_up_right.diagonal(dim1=2, dim2=3).movedim(-1, 2)
    U_down_left_tr = U_down_left.diagonal(dim1=2, dim2=3).movedim(-1, 2)
    U_down_right_tr = U_down_right.diagonal(dim1=2, dim2=3).movedim(-1, 2)
    # Vd
    Vd_up_up_partv = torch.einsum('pqmr, rk, pqnk, qnk -> pq', alpha_up_tr, V_Shen_w, theta_up, phi_up) - torch.einsum('pqmnr, rk, pqnk, qmk -> pq', alpha_up, V_Shen_w, theta_up, phi_up)
    Vd_up_up_partw1 = 0.5 * (torch.einsum('pqm, k, pqnk, qnk -> pq', U_up_right_tr - U_up_left_tr, w_r * point_r, theta_up, phi_up) - torch.einsum('pqmn, k, pqnk, qmk -> pq', U_up_right - U_up_left, w_r * point_r, theta_up, phi_up))
    Vd_up_up_partw2 = 0.5 * (torch.einsum('pqm, k, pqnk, qnk -> pq', U_up_right_tr + U_up_left_tr, w_r, theta_up, phi_up) - torch.einsum('pqmn, k, pqnk, qmk -> pq', U_up_right + U_up_left, w_r, theta_up, phi_up))
    Vd_up_up = (Vd_up_up_partv + Vd_up_up_partw1 + Vd_up_up_partw2)
    Vd_down_down_partv = torch.einsum('pqmr, rk, pqnk, qnk -> pq', alpha_down_tr, V_Shen_w, theta_down, phi_down) - torch.einsum('pqmnr, rk, pqnk, qmk -> pq', alpha_down, V_Shen_w, theta_down, phi_down)
    Vd_down_down_partw1 = 0.5 * (torch.einsum('pqm, k, pqnk, qnk -> pq', U_down_right_tr - U_down_left_tr, w_r * point_r, theta_down, phi_down) - torch.einsum('pqmn, k, pqnk, qmk -> pq', U_down_right - U_down_left, w_r * point_r, theta_down, phi_down))
    Vd_down_down_partw2 = 0.5 * (torch.einsum('pqm, k, pqnk, qnk -> pq', U_down_right_tr + U_down_left_tr, w_r, theta_down, phi_down) - torch.einsum('pqmn, k, pqnk, qmk -> pq', U_down_right + U_down_left, w_r, theta_down, phi_down))
    Vd_down_down = (Vd_down_down_partv + Vd_down_down_partw1 + Vd_down_down_partw2)
    Vd_up_down_partv = torch.einsum('pqmr, rk, pqnk, qnk -> pq', alpha_up_tr, V_Shen_w, theta_down, phi_down)
    Vd_up_down_partw1 = 0.5 * torch.einsum('pqm, k, pqnk, qnk -> pq', U_up_right_tr - U_up_left_tr, w_r * point_r, theta_down, phi_down)
    Vd_up_down_partw2 = 0.5 * torch.einsum('pqm, k, pqnk, qnk -> pq', U_up_right_tr + U_up_left_tr, w_r, theta_down, phi_down)
    Vd_up_down = (Vd_up_down_partv + Vd_up_down_partw1 + Vd_up_down_partw2)
    Vd = (-0.5 * r_c) * M * (Vd_up_up + 2 * Vd_up_down + Vd_down_down)
    return M, T, Vs, Vd

def solve_eigenvalue_problem(M, T, Vs, Vd, tol=1e-14):
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
# build_matrices_logic = torch.compile(build_matrices_logic, mode='max-autotune')

def criterion(model_up, model_down):
    phi_up, grad_phi_up = model_up(w_r, point_r, need_grad=1, normed=True) # [N_up, dim, p, K] (dim=1)
    phi_up = phi_up.squeeze(1).transpose_(dim0=0,dim1=1) # [p, N_up, K]
    grad_phi_up = grad_phi_up.squeeze(1).transpose_(dim0=0,dim1=1) # [p, N_up, K]
    phi_down, grad_phi_down = model_down(w_r, point_r, need_grad=1, normed=True) # [N_down, dim, p, K] (dim=1)
    phi_down = phi_down.squeeze(1).transpose_(dim0=0,dim1=1) # [p, N_down, K]
    grad_phi_down = grad_phi_down.squeeze(1).transpose_(dim0=0,dim1=1) # [p, N_down, K]
    M, T, Vs, Vd = build_matrices_logic(
        phi_up, grad_phi_up, phi_down, grad_phi_down, w_r, V_r, V_Shen_w, sigma, point_r, V_r_left, V_r_right
    )
    loss, alpha = solve_eigenvalue_problem(M, T, Vs, Vd)
    with torch.no_grad():
        P = torch.outer(alpha, alpha)
        M_val = torch.sum(P * M)
        T_val = torch.sum(P * T)
        Vs_val = torch.sum(P * Vs)
        Vd_val = torch.sum(P * Vd)
    return loss, M_val, T_val, Vs_val, Vd_val, alpha

# ********** Training Process (RAdam + Cosine Scheduler with warmrestarts) **********
# --- Configuration ---
phase1_lr = 1e-3 
phase1_epochs = 635000     
print_every = 100
optimizer_choice = 'RAdam' 
print(f"{'='*20} PHASE 1: Exploration (0 - {phase1_epochs}) {'='*20}")
# --- Optimizer Setup ---
params = filter(lambda p: p.requires_grad, itertools.chain(model_up.parameters(), model_down.parameters()))

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
    loss, M, T, V_ie, V_ee, alpha = criterion(model_up, model_down)
    # Check for new minimum loss
    current_loss = loss.item()
    if (current_loss < min_loss) and ((e + 1) % 5 == 0):
        min_loss = current_loss  # Update the minimum loss
        best_model_r1_state = copy.deepcopy(model_up.state_dict())
        best_model_r2_state = copy.deepcopy(model_down.state_dict())
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
model_up.load_state_dict(best_model_r1_state)
model_down.load_state_dict(best_model_r2_state)
# --- Optimizer Setup ---
params = filter(lambda p: p.requires_grad, itertools.chain(model_up.parameters(), model_down.parameters()))
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
    loss, M, T, V_ie, V_ee, alpha = criterion(model_up, model_down)
    # Check for new minimum loss
    current_loss = loss.item()
    if (current_loss < min_loss) and ((e + 1) % 5 == 0):
        min_loss = current_loss  # Update the minimum loss
        best_model_r1_state = copy.deepcopy(model_up.state_dict())
        best_model_r2_state = copy.deepcopy(model_down.state_dict())
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
model_up.load_state_dict(best_model_r1_state)
model_down.load_state_dict(best_model_r2_state)
# parameters
lr = 0.01
epochs = 5000
print_every = 100
save = True
# optimizer used
optimizer = torch.optim.LBFGS(itertools.chain(model_up.parameters(), model_down.parameters()), lr=lr, max_iter=128, history_size=256, tolerance_grad=1e-13, tolerance_change=1e-15)
# training
for e in range(epochs):
    e = e + phase1_epochs + phase2_epochs 
    # initial info
    def closure():
        loss, M, T, V_ie, V_ee, alpha = criterion(model_up, model_down)
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