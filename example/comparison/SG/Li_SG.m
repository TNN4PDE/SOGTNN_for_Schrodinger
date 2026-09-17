% =========================================================================
% Full Grid Spectral Element Method for 1D Li Atom (Soft-Coulomb)
% System: 3 Electrons (2 Spin-Up, 1 Spin-Down)
% Ansatz: Biorthogonalized Slater Determinants (SG-CI)
% Basis: Piecewise Legendre-Shen Polynomials on [-1,0] and [0,1]
% Truncation: Hyperbolic Cross Sparse Grid (Extreme Vectorized Assembly)
% Memory: Dynamic VRAM Autopilot + Aggressive Clearance for 40k+ Basis
% =========================================================================
clear;
%% 1. Parameters
Z = 3;              % Nuclear charge for Lithium
rc = 15;            % Cutoff radius
max_mode = 24;      % 1D Basis maximum order
N = 2*max_mode + 1; % Size of 1D basis
nq_sub = 256;       % Quadrature points

% --- Hyperbolic Cross Truncation Parameters ---
use_truncation = true; 
hc_multiplier = 48;
hc_threshold = max(1, max_mode) * hc_multiplier; 

fprintf('Initializing Li Atom SG-CI Solver (Extreme Scale & Memory Safe)...\n');
fprintf('rc = %.1f, Basis Order = %d (Size: %d)\n', rc, max_mode, N);
fprintf('Hyperbolic Cross Threshold: %d\n', hc_threshold);

%% 2. Load Coefficients
data_dir = fullfile(fileparts(mfilename('fullpath')), '..','..','..', 'data');
S1 = load(fullfile(data_dir, 'E_k_rc15.mat'),'coeffs');
alpha = S1.coeffs;
S2 = load(fullfile(data_dir, 'E_kl_rc15.mat'),'E_kl_total'); E_kl = S2.E_kl_total;
p = size(E_kl,1);

%% 3. Basis Construction (Fast Vectorized SG-CI for 3 Electrons)
fprintf('Constructing Basis Space (Vectorized Divide & Conquer)...\n');
tic;
range_modes = -max_mode:max_mode;

% --- Step 1: Generate Spin-Up combinations (m < n) ---
up_combs = nchoosek(range_modes, 2);
val_up = max(1, abs(up_combs));
prod_up = prod(val_up, 2);
valid_up_mask = (prod_up <= hc_threshold);
valid_up = up_combs(valid_up_mask, :);
prod_up_valid = prod_up(valid_up_mask);

% --- Step 2: Generate Spin-Down combinations (t) ---
dn_combs = range_modes(:);
val_dn = max(1, abs(dn_combs));
prod_dn = val_dn;
valid_dn_mask = (prod_dn <= hc_threshold);
valid_dn = dn_combs(valid_dn_mask, :);
prod_dn_valid = prod_dn(valid_dn_mask);

% --- Step 3: Combine and apply global SG Truncation ---
N_up = size(valid_up, 1);
N_dn = size(valid_dn, 1);
est_states = min(N_up * N_dn, 1000000); 
indices_buffer = zeros(est_states, 3);
count = 0;

for i = 1:N_up
    current_prod_up = prod_up_valid(i);
    match_mask = (current_prod_up .* prod_dn_valid) <= hc_threshold;
    num_matches = sum(match_mask);
    
    if num_matches > 0
        if count + num_matches > size(indices_buffer, 1)
            indices_buffer = [indices_buffer; zeros(est_states, 3)];
        end
        matched_dn = valid_dn(match_mask, :);
        rep_up = repmat(valid_up(i, :), num_matches, 1);
        indices_buffer(count+1 : count+num_matches, :) = [rep_up, matched_dn];
        count = count + num_matches;
    end
end
indices = indices_buffer(1:count, :);
N_states = size(indices, 1);
fprintf('Total SG DoFs (Li, 3-Electron): %d (Generated in %.3f sec)\n', N_states, toc);

%% 4. Precompute 1D Analytic Matrices (Piecewise L-S)
fprintf('Constructing Analytic 1D Matrices...\n');
M1 = zeros(N); T1_raw = zeros(N);
absidx = @(k) abs(k);
idx_map = @(k) k + max_mode + 1;

for ii = 1:N
    k1 = range_modes(ii); i = idx_map(k1);
    for jj = 1:N
        k2 = range_modes(jj); j = idx_map(k2);
        
        if k1 == 0 && k2 == 0, M1(i,j) = 2/3;
        elseif k1 == 0 && k2 ~= 0, q = absidx(k2); M1(i,j) = 0.5*(q==1) - (1/6)*(q==2);
        elseif k1 ~= 0 && k2 == 0, p_abs = absidx(k1); M1(i,j) = 0.5*(p_abs==1) - (1/6)*(p_abs==2);
        elseif k1 * k2 < 0, M1(i,j) = 0;
        else
            p_abs = absidx(k1); q_abs = absidx(k2); term = 0;
            if p_abs == q_abs, term = term + ( 1/(2*p_abs - 1) + 1/(2*p_abs + 3) ); end
            if (p_abs-1) == (q_abs+1), term = term - 1/(2*p_abs - 1); end
            if (p_abs+1) == (q_abs-1), term = term - 1/(2*p_abs + 3); end
            M1(i,j) = term;
        end
        if k1 == k2
            if k1 == 0, T1_raw(i,j) = 2; else, T1_raw(i,j) = 4*(2*absidx(k1) + 1); end
        else, T1_raw(i,j) = 0;
        end
    end
end

%% 5. Quadrature & Bstack Setup
fprintf('Preparing Quadrature and Bstack...\n');
currentPool = gcp('nocreate'); if isempty(currentPool), parpool; end
[x_ref, w_ref] = legendre_gauss(nq_sub);
s_pos = (x_ref + 1) / 2; w_pos = w_ref / 2;
s_neg = (x_ref - 1) / 2; w_neg = w_ref / 2;
Jneed = max(size(alpha, 1), size(E_kl, 1));

pos_modes = range_modes(range_modes > 0); pos_idx_local = arrayfun(idx_map, pos_modes);
neg_modes = range_modes(range_modes < 0); neg_idx_local = arrayfun(idx_map, neg_modes);
zero_idx_local = idx_map(0);
np = numel(pos_modes); nn = numel(neg_modes);

Phi_pos_sub = zeros(np, nq_sub); xarg_pos = 2*s_pos - 1;
for ii=1:np, Phi_pos_sub(ii,:) = legendreP_at_nodes(pos_modes(ii)-1, xarg_pos) - legendreP_at_nodes(pos_modes(ii)+1, xarg_pos); end
Phi_neg_sub = zeros(nn, nq_sub); xarg_neg = -1 - 2*s_neg;
for ii=1:nn, Phi_neg_sub(ii,:) = legendreP_at_nodes(abs(neg_modes(ii))-1, xarg_neg) - legendreP_at_nodes(abs(neg_modes(ii))+1, xarg_neg); end
Phi0_pos = 1 - s_pos; Phi0_neg = 1 - abs(s_neg);

Bstack = zeros(N, N, Jneed);
Wpos = w_pos(:)'; Wneg = w_neg(:)';

parfor j_idx = 1:Jneed
    j_poly = j_idx - 1; Pj_pos = legendreP_at_nodes(j_poly, s_pos); Pj_neg = legendreP_at_nodes(j_poly, s_neg);
    B_local = zeros(N, N);
    if np > 0, coeff = Pj_pos .* Wpos; B_local(pos_idx_local, pos_idx_local) = (Phi_pos_sub .* coeff) * Phi_pos_sub'; end
    if nn > 0, coeff = Pj_neg .* Wneg; B_local(neg_idx_local, neg_idx_local) = (Phi_neg_sub .* coeff) * Phi_neg_sub'; end
    if np > 0 && ~isempty(zero_idx_local)
        B0p = (Phi0_pos .* (Pj_pos .* Wpos)) * Phi_pos_sub';
        B_local(zero_idx_local, pos_idx_local) = B0p; B_local(pos_idx_local, zero_idx_local) = B0p';
    end
    if nn > 0 && ~isempty(zero_idx_local)
        B0n = (Phi0_neg .* (Pj_neg .* Wneg)) * Phi_neg_sub';
        B_local(zero_idx_local, neg_idx_local) = B0n; B_local(neg_idx_local, zero_idx_local) = B0n';
    end
    if ~isempty(zero_idx_local)
        B_local(zero_idx_local, zero_idx_local) = sum(Phi0_pos.^2 .* Pj_pos .* Wpos) + sum(Phi0_neg.^2 .* Pj_neg .* Wneg);
    end
    Bstack(:, :, j_idx) = B_local;
end

%% 6. Precompute Potential Tensors
fprintf('Precomputing Potential Tensors...\n');
V_s = reshape(reshape(Bstack, [], size(alpha,1)) * alpha(:), N, N);
V_vec = reshape(Bstack, N*N, p);
V_D_raw = V_vec * E_kl * V_vec.'; 

%% 7. Global Matrix Assembly (GPU + Smart VRAM Autopilot)
fprintf('Assembling Global Matrices...\n');
if parallel.gpu.GPUDevice.isAvailable
    g = gpuDevice; reset(g); 
    fprintf('Detected GPU: %s (VRAM Available: %.1f GB)\n', g.Name, g.AvailableMemory/1e9);
else
    error('GPU not found.');
end

G_M1 = gpuArray(M1); G_T1 = gpuArray(T1_raw); G_Vs = gpuArray(V_s);
G_VD_raw = gpuArray(V_D_raw); G_indices = gpuArray(indices); 
N2 = N^2;



mem_per_row = N_states * 8 * 25; 
safe_vram = g.AvailableMemory * 0.7;
gpu_chunk_size = floor(safe_vram / mem_per_row);
gpu_chunk_size = max(100, min(gpu_chunk_size, 2500));
num_chunks = ceil(N_states / gpu_chunk_size);
fprintf('  -> Autopilot selected Chunk Size: %d (~%d Chunks total)\n', gpu_chunk_size, num_chunks);

offset = max_mode + 1;
sparsity_threshold = 1e-8; 

est_nnz = min(N_states * 600, N_states^2);
I_M = zeros(est_nnz, 1); J_M = zeros(est_nnz, 1); V_M = zeros(est_nnz, 1);
I_H = zeros(est_nnz, 1); J_H = zeros(est_nnz, 1); V_H = zeros(est_nnz, 1);
idx_M = 1; idx_H = 1;

t_start = tic;
for k = 1:num_chunks
    s_idx = (k-1)*gpu_chunk_size + 1; e_idx = min(k*gpu_chunk_size, N_states);
    current_rows = s_idx:e_idx;
    
    % Up-Spin indices
    g_m_bra = G_indices(current_rows, 1) + offset; g_n_bra = G_indices(current_rows, 2) + offset;
    g_m_ket = G_indices(:, 1) + offset; g_n_ket = G_indices(:, 2) + offset;
    % Down-Spin indices
    g_t_bra = G_indices(current_rows, 3) + offset;
    g_t_ket = G_indices(:, 3) + offset;
    
    get_1D_gpu = @(Mat, br, kt) Mat(br + (kt.' - 1) * N);
    
    % --- UP Spin Determinant ---
    M_m1m2 = get_1D_gpu(G_M1, g_m_bra, g_m_ket); M_m1n2 = get_1D_gpu(G_M1, g_m_bra, g_n_ket);
    M_n1m2 = get_1D_gpu(G_M1, g_n_bra, g_m_ket); M_n1n2 = get_1D_gpu(G_M1, g_n_bra, g_n_ket);
    Dup_11 = M_n1n2; Dup_12 = -M_n1m2; Dup_21 = -M_m1n2; Dup_22 = M_m1m2;
    M_up = M_m1m2 .* M_n1n2 - M_m1n2 .* M_n1m2;
    clear M_m1m2 M_m1n2 M_n1m2 M_n1n2;
    
    % --- DOWN Spin Determinant ---
    M_dn = get_1D_gpu(G_M1, g_t_bra, g_t_ket);
    
    % 1. Global Mass Element
    Mg_local = M_up .* M_dn;
    
    % 2. Kinetic Elements
    T_m1m2 = get_1D_gpu(G_T1, g_m_bra, g_m_ket); T_m1n2 = get_1D_gpu(G_T1, g_m_bra, g_n_ket);
    T_n1m2 = get_1D_gpu(G_T1, g_n_bra, g_m_ket); T_n1n2 = get_1D_gpu(G_T1, g_n_bra, g_n_ket);
    T_up = T_m1m2.*Dup_11 + T_m1n2.*Dup_12 + T_n1m2.*Dup_21 + T_n1n2.*Dup_22;
    clear T_m1m2 T_m1n2 T_n1m2 T_n1n2;
    
    T_dn = get_1D_gpu(G_T1, g_t_bra, g_t_ket);
    T_total = (T_up .* M_dn + T_dn .* M_up) * (1 / (2 * rc^2));
    clear T_up T_dn;
    
    % 3. Nuclear Potential Elements (V_S)
    Vs_m1m2 = get_1D_gpu(G_Vs, g_m_bra, g_m_ket); Vs_m1n2 = get_1D_gpu(G_Vs, g_m_bra, g_n_ket);
    Vs_n1m2 = get_1D_gpu(G_Vs, g_n_bra, g_m_ket); Vs_n1n2 = get_1D_gpu(G_Vs, g_n_bra, g_n_ket);
    Vs_up = Vs_m1m2.*Dup_11 + Vs_m1n2.*Dup_12 + Vs_n1m2.*Dup_21 + Vs_n1n2.*Dup_22;
    clear Vs_m1m2 Vs_m1n2 Vs_n1m2 Vs_n1n2;
    
    Vs_dn = get_1D_gpu(G_Vs, g_t_bra, g_t_ket);
    Vs_total = (Vs_up .* M_dn + Vs_dn .* M_up) * (-Z / rc);
    clear Vs_up Vs_dn;
    
    % 4. Electron-Electron Repulsion (V_D) 
    % 4a. Same-Spin (UP-UP) -> 1 pair -> Inline calculation avoids memory spike
    v_up_up = G_VD_raw((g_m_bra + (g_m_ket.' - 1)*N) + (g_n_bra + (g_n_ket.' - 1)*N - 1)*N2) ...
            - G_VD_raw((g_m_bra + (g_n_ket.' - 1)*N) + (g_n_bra + (g_m_ket.' - 1)*N - 1)*N2) ...
            - G_VD_raw((g_n_bra + (g_m_ket.' - 1)*N) + (g_m_bra + (g_n_ket.' - 1)*N - 1)*N2) ...
            + G_VD_raw((g_n_bra + (g_n_ket.' - 1)*N) + (g_m_bra + (g_m_ket.' - 1)*N - 1)*N2);
    Vee_up_up = v_up_up .* M_dn;
    clear v_up_up;
    
    % 4c. Cross-Spin (UP-DOWN) -> Pre-calculate DOWN shift to save operations
    p_up = {g_m_bra, g_n_bra}; q_up = {g_m_ket, g_n_ket};
    D_up_cell = {Dup_11, Dup_12; Dup_21, Dup_22};
    idx_dn_shift = (g_t_bra + (g_t_ket.' - 1)*N - 1) * N2; 
    
    Vee_cross = zeros(size(Mg_local), 'gpuArray');
    for i_up = 1:2
        for j_up = 1:2
            vd = G_VD_raw(p_up{i_up} + (q_up{j_up}.' - 1)*N + idx_dn_shift);
            Vee_cross = Vee_cross + vd .* D_up_cell{i_up, j_up};
        end
    end
    clear idx_dn_shift vd p_up q_up D_up_cell Dup_11 Dup_12 Dup_21 Dup_22 M_up M_dn;
    
    % Final Hamiltonian addition
    Vee_total = (0.5*Vee_up_up + Vee_cross) * (1 / rc);
    clear Vee_up_up Vee_cross;
    
    Hg_local = T_total + Vs_total + Vee_total;
    clear T_total Vs_total Vee_total;
    
    % --- Gather and format to sparse ---
    M_cpu = gather(Mg_local); H_cpu = gather(Hg_local);
    clear Mg_local Hg_local;
    
    [r_M, c_M] = find(abs(M_cpu) > sparsity_threshold);
    [r_H, c_H] = find(abs(H_cpu) > sparsity_threshold);
    
    if ~isempty(r_M)
        len = length(r_M);
        if idx_M+len-1 > length(I_M), I_M=[I_M; zeros(est_nnz,1)]; J_M=[J_M; zeros(est_nnz,1)]; V_M=[V_M; zeros(est_nnz,1)]; end
        I_M(idx_M:idx_M+len-1) = reshape(current_rows(r_M),[],1); J_M(idx_M:idx_M+len-1) = reshape(c_M,[],1);
        V_M(idx_M:idx_M+len-1) = reshape(M_cpu(sub2ind(size(M_cpu), r_M, c_M)),[],1); idx_M = idx_M + len;
    end
    if ~isempty(r_H)
        len = length(r_H);
        if idx_H+len-1 > length(I_H), I_H=[I_H; zeros(est_nnz,1)]; J_H=[J_H; zeros(est_nnz,1)]; V_H=[V_H; zeros(est_nnz,1)]; end
        I_H(idx_H:idx_H+len-1) = reshape(current_rows(r_H),[],1); J_H(idx_H:idx_H+len-1) = reshape(c_H,[],1);
        V_H(idx_H:idx_H+len-1) = reshape(H_cpu(sub2ind(size(H_cpu), r_H, c_H)),[],1); idx_H = idx_H + len;
    end
    
    if mod(k, 10) == 0, fprintf('    Chunk %d/%d done. (%.1f sec)\n', k, num_chunks, toc(t_start)); end
end

%% 8. Solve
fprintf('  -> Constructing final sparse matrices...\n');
Mg = sparse(I_M(1:idx_M-1), J_M(1:idx_M-1), V_M(1:idx_M-1), N_states, N_states);
Hg = sparse(I_H(1:idx_H-1), J_H(1:idx_H-1), V_H(1:idx_H-1), N_states, N_states);
clear I_M J_M V_M I_H J_H V_H


Hg = (Hg + Hg') / 2; 
Mg = (Mg + Mg') / 2;

fprintf('Solving Generalized Eigenvalue Problem (Shift-and-Invert)...\n');

opts.p = 64; 
opts.maxit = 8192; 
opts.tol = 1e-8;

try
    [V, D] = eigs(Hg, Mg, 16, 'sa', opts);
    fprintf('\nResults (Ground State):\nE_0 = %.16f a.u.\n', diag(D));
catch ME
    fprintf('Solver failed: %s\n', ME.message);
end

%% --- Helpers ---
function Pj = legendreP_at_nodes(j, x)
    x = x(:)';
    if j == 0, Pj = ones(size(x)); elseif j == 1, Pj = x;
    else
        Pm2 = ones(size(x)); Pm1 = x;
        for n = 2:j, Pn = ((2*n-1) .* x .* Pm1 - (n-1) * Pm2)/n; Pm2 = Pm1; Pm1 = Pn; end
        Pj = Pn;
    end
end
function [x,w] = legendre_gauss(n)
    beta = (1:n-1) ./ sqrt((2*(1:n-1)).^2 - 1);
    [V,D] = eig(diag(beta,1) + diag(beta,-1)); x = diag(D)'; [x,perm] = sort(x);
    V = V(:,perm); w = 2 * (V(1,:).^2); x = x(:)'; w = w(:)';
end