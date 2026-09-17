import os
import torch
import torch.optim as optim
torch.set_printoptions(precision=16)
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from sogtnn.quad_larger import *
from sogtnn.integration import *
from sogtnn.tnn_larger import *
import itertools
import time
import copy
# ********** choose data type and device **********
dtype = torch.float64
device = 'cuda:0' 
# ********** generate data points **********
a = -15.
b = 15.
dim = 1
# number of quad points
quad = 512
# number of partitions for [a,b]
n = 10
# quad points and quad weights.
point_r, w_r = composite_quadrature_1d(quad, a, b, n, device=device, dtype=dtype)
K_r = len(point_r)
print(K_r)
V_r = 1. / torch.sqrt(point_r ** 2 + 1)
# TNN construction
p = 1
sizes = [1, 64, 128, p]
def bd(x):
    return (x-a) * (b - x)

def grad_bd(x):
    return a+b-2*x
activation = TNN_Sin
model_r = TNN(1, sizes, activation, bd=bd, grad_bd=grad_bd, scaling=True).to(dtype).to(device)
# ********** Loss Function **********
def criterion(model_r):
    phi_r, grad_phi_r = model_r(w_r, point_r, need_grad=1, normed=True)
    alpha = model_r.scaling_par()
    # WF inner product
    int_phi = Int2TNN(w_r, alpha, phi_r, alpha, phi_r, if_sum=True)
    # Kinetic energy part
    K = Int2TNN(w_r, alpha, grad_phi_r, alpha, grad_phi_r, if_sum=True)
    # Potential energy part
    V = Int2TNN(w_r * V_r, alpha, phi_r, alpha, phi_r, if_sum=True)
    # Rayleigh quotient
    E = (K / 2 - V) / int_phi
    loss = E
    return loss, int_phi, K, V, alpha

# ********** Training Process (RAdam + Cosine Scheduler with warmrestarts) **********
# --- Configuration ---
phase1_lr = 1e-3
phase1_epochs = 635000    
print_every = 100
optimizer_choice = 'RAdam' 
print(f"{'='*20} PHASE 1: Exploration (0 - {phase1_epochs}) {'='*20}")
# --- Optimizer Setup ---
params = filter(lambda p: p.requires_grad, itertools.chain(model_r.parameters()))
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
best_model_r_state = None
# training
starttime = time.time()
for e in range(phase1_epochs):
    loss, M, T, V_ie, alpha = criterion(model_r)
    # Check for new minimum loss
    current_loss = loss.item()
    if (current_loss < min_loss) and ((e + 1) % 5 == 0):
        min_loss = current_loss  # Update the minimum loss
        best_model_r_state = copy.deepcopy(model_r.state_dict())
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
model_r.load_state_dict(best_model_r_state)
# --- Optimizer Setup ---
params = filter(lambda p: p.requires_grad, itertools.chain(model_r.parameters()))
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
    loss, M, T, V_ie, alpha = criterion(model_r)
    # Check for new minimum loss
    current_loss = loss.item()
    if (current_loss < min_loss) and ((e + 1) % 5 == 0):
        min_loss = current_loss  # Update the minimum loss
        best_model_r_state = copy.deepcopy(model_r.state_dict())
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
model_r.load_state_dict(best_model_r_state)
# parameters
lr = 0.01
epochs = 5000
print_every = 100
save = True
# optimizer used
optimizer = torch.optim.LBFGS(itertools.chain(model_r.parameters()), lr=lr, max_iter=32, history_size=64, tolerance_grad=1e-13, tolerance_change=1e-15)
# training
for e in range(epochs):
    e = e + phase1_epochs + phase2_epochs 
    # initial info
    def closure():
        loss, M, T, V_ie, alpha = criterion(model_r)
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