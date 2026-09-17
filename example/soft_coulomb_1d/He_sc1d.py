import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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
r_c = 13.
a = -r_c
b = r_c
dim = 1
Z = 2.
# Quadrature setup
quad = 512
n = 10
point_r, w_r = composite_quadrature_1d(quad, a, b, n, device=device, dtype=dtype)
print(f"Integration points: {len(point_r)}")
V_r = 1. / torch.sqrt(point_r**2 + 1)
sigma_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'Sigma_rc13_1e-12.mat'))
sigma = torch.tensor(sigma_data['lambda_r'], device=device, dtype=dtype)
V_data = loadmat(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'V_rc13_1e-12.mat'))
V = torch.tensor(V_data['V_r'], device=device, dtype=dtype).t()
max_mode = V.shape[1]-1
print(f"Max Mode: {max_mode}")
# Chebyshev initialization
def compute_chebyshev_ploy(x, N):
    x_flat = x.reshape(-1)  
    m = x_flat.shape[0] 
    T_mat = torch.zeros(m, N + 1, device=device, dtype=dtype)
    T_mat[:, 0] = 1.0
    if N >= 1:
        T_mat[:, 1] = x_flat     
    for k in range(2, N + 1):
        T_mat[:, k] = 2 * x_flat * T_mat[:, k-1] - T_mat[:, k-2]
    return T_mat
chebyshev_poly = compute_chebyshev_ploy((point_r / r_c), max_mode).t() # shape [max_mode+1, K]
V_cheb_w = torch.mm(V, chebyshev_poly * w_r) # [r, n]
p = 24
sizes = [1, 64, 128, p]
def bd(x): return (x-a) * (b - x)
def grad_bd(x): return a+b-2*x
activation = TNN_Sin
model_r1 = TNN(1, sizes, activation, bd=bd, grad_bd=grad_bd, scaling=False).to(dtype).to(device)
model_r2 = TNN(1, sizes, activation, bd=bd, grad_bd=grad_bd, scaling=False).to(dtype).to(device)
# print("Compiling models...")
# model_r1 = torch.compile(model_r1)
# model_r2 = torch.compile(model_r2)
def build_matrices_logic(phi_r1, phi_r2, grad_r1, grad_r2, w_r, V_r, V_cheb_w, sigma):
    sqrt_w = torch.sqrt(w_r)
    phi_r1_w = phi_r1 * sqrt_w
    phi_r2_w = phi_r2 * sqrt_w
    grad_r1_w = grad_r1 * sqrt_w
    grad_r2_w = grad_r2 * sqrt_w
    # Overlap Matrix (S)
    M1 = phi_r1_w @ phi_r1_w.t()
    M2 = phi_r2_w @ phi_r2_w.t()
    S_mat = M1 * M2
    # Kinetic Matrix (T)
    K1 = grad_r1_w @ grad_r1_w.t()
    K2 = grad_r2_w @ grad_r2_w.t()
    T_mat = 0.5 * (K1 * M2 + M1 * K2)
    # Potential Matrix (V_ie)
    w_V = w_r * V_r
    V1 = (phi_r1 * w_V) @ phi_r1.t() 
    V2 = (phi_r2 * w_V) @ phi_r2.t()
    V_ie_mat = -Z * (V1 * M2 + M1 * V2)
    # Electron-Electron Interaction (V_ee)
    # cheb_int_r1: [r, p, p]
    cheb_int_r1 = torch.einsum('rn, pn, qn -> rpq', V_cheb_w, phi_r1, phi_r1)
    cheb_int_r2 = torch.einsum('rn, pn, qn -> rpq', V_cheb_w, phi_r2, phi_r2)
    # sigma: [r, 1], cheb_int: [r, p, p]
    # V_ee_mat = Sum_r (sigma * cheb_int_r1 * cheb_int_r2)
    V_ee_mat = torch.sum(sigma.view(-1, 1, 1) * cheb_int_r1 * cheb_int_r2, dim=0)
    return S_mat, T_mat, V_ie_mat, V_ee_mat

def solve_eigenvalue_problem(S_mat, T_mat, V_ie_mat, V_ee_mat):
    H_mat = T_mat + V_ie_mat + V_ee_mat
    L = torch.linalg.cholesky(S_mat)
    C = torch.linalg.solve(L, H_mat.t()).t()
    D = torch.linalg.solve(L, C)
    E, U = torch.linalg.eigh(D)
    ind = torch.argmin(E)
    lam = E[ind]
    alpha = torch.linalg.solve(L.t(), U[:, ind])
    return lam, alpha

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
    S_mat, T_mat, V_ie_mat, V_ee_mat = build_matrices_logic(
        phi_r1, phi_r2, grad_phi_r1, grad_phi_r2, w_r, V_r, V_cheb_w, sigma
    )
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