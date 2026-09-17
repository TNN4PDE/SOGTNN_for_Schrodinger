import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
r_c = 20.
a = -r_c
b = r_c
dim = 1
N = 8
N_up = 4
N_down = 4
Z = 8.
# Quadrature setup
quad = 512
n = 10
point_r, w_r = composite_quadrature_1d(quad, a, b, n, device=device, dtype=dtype)
K = len(point_r)
V_r = 1. / torch.sqrt(point_r**2 + 1)
# EE Interaction data
sigma_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'Sigma_rc20_1e-12.mat'))
sigma = torch.tensor(sigma_data['lambda_r'], device=device, dtype=dtype).squeeze(1)
V_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'V_rc20_1e-12.mat'))
V = torch.tensor(V_data['V_r'], device=device, dtype=dtype).t()
max_mode = V.shape[1]-1
print(f"Max Mode: {max_mode}")
# Chebyshev initialization
def compute_chebyshev_ploy(x, N):
    x_flat = x.reshape(-1)
    m = x_flat.shape[0]
    T_mat = torch.zeros(m, N + 1, device=device, dtype=dtype)
    T_mat[:, 0] = 1.0
    if N >= 1: T_mat[:, 1] = x_flat
    for k in range(2, N + 1):
        T_mat[:, k] = 2 * x_flat * T_mat[:, k-1] - T_mat[:, k-2]
    return T_mat
chebyshev_poly = compute_chebyshev_ploy((point_r / r_c), max_mode).t()
V_cheb_w = torch.mm(V, chebyshev_poly * w_r) # [r, K]
p = 84
sizes = [1, 64, 128, 256, p]
def bd(x): return (x-a) * (b - x)
def grad_bd(x): return a+b-2*x
activation = TNN_Sin
model_up = Multi_TNN(N_up, dim, sizes, activation, bd=bd, grad_bd=grad_bd, scaling=False).to(dtype).to(device)
model_down = Multi_TNN(N_down, dim, sizes, activation, bd=bd, grad_bd=grad_bd, scaling=False).to(dtype).to(device)
# model_up = torch.compile(model_up)
# model_down = torch.compile(model_down)
def build_matrices_logic(phi_up, grad_phi_up, phi_down, grad_phi_down, w_r, V_r, V_cheb_w, sigma):
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
    T = 0.5 * M * (T_up + T_down)
    # --- 4. Single Electron Potential ---
    Vs_up = torch.einsum('k, pqnk, qnk -> pq', w_r * V_r, theta_up, phi_up)
    Vs_down = torch.einsum('k, pqnk, qnk -> pq', w_r * V_r, theta_down, phi_down)
    Vs = -Z * M * (Vs_up + Vs_down)
    # --- 5. Double Electron Potential---
    cheb_up = torch.einsum('rk, pqnk, qmk -> pqmnr', V_cheb_w, theta_up, phi_up)
    cheb_down = torch.einsum('rk, pqnk, qmk -> pqmnr', V_cheb_w, theta_down, phi_down)
    cheb_up_tr = cheb_up.diagonal(dim1=2, dim2=3).movedim(-1, 2) 
    cheb_down_tr = cheb_down.diagonal(dim1=2, dim2=3).movedim(-1, 2)
    Vd_up_up = torch.einsum('r, pqnr, pqmr -> pq', sigma, cheb_up_tr, cheb_up_tr) - \
               torch.einsum('r, pqmnr, pqnmr -> pq', sigma, cheb_up, cheb_up)
    Vd_down_down = torch.einsum('r, pqnr, pqmr -> pq', sigma, cheb_down_tr, cheb_down_tr) - \
                   torch.einsum('r, pqmnr, pqnmr -> pq', sigma, cheb_down, cheb_down)
    Vd_up_down = torch.einsum('r, pqnr, pqmr -> pq', sigma, cheb_up_tr, cheb_down_tr)
    Vd = 0.5 * M * (Vd_up_up + 2 * Vd_up_down + Vd_down_down)
    return M, T, Vs, Vd

def solve_eigenvalue_problem(M, T, Vs, Vd):
    H = Vd + Vs + T
    L = torch.linalg.cholesky(M)
    C = torch.linalg.solve(L, H.t()).t()
    D = torch.linalg.solve(L, C)
    E_vals, U = torch.linalg.eigh(D)
    ind = torch.argmin(E_vals)
    lam = E_vals[ind]
    alpha = torch.linalg.solve(L.t(), U[:, ind])
    return lam, alpha

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
        phi_up, grad_phi_up, phi_down, grad_phi_down, w_r, V_r, V_cheb_w, sigma
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
optimizer = torch.optim.LBFGS(itertools.chain(model_up.parameters(), model_down.parameters()), lr=lr, max_iter=32, history_size=64, tolerance_grad=1e-13, tolerance_change=1e-15)
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