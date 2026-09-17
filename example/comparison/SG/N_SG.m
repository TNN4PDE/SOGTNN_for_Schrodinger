% =========================================================================
% Sparse Grid Spectral Element Method for 1D Nitrogen Atom (Soft-Coulomb)
% System: 7 Electrons (4 Spin-Up, 3 Spin-Down, Quartet State)
% Ansatz: Biorthogonalized Slater Determinants (Löwdin Rules)
% Basis: Piecewise Legendre-Shen Polynomials on [-1,0] and [0,1]
% Truncation: Hyperbolic Cross Sparse Grid (Extreme Vectorized Assembly)
% =========================================================================
clear;
%% 1. Parameters
Z = 7;              % Nuclear charge for Nitrogen
rc = 20;            % Cutoff radius
max_mode = 7;       % 1D Basis maximum order 
N = 2*max_mode + 1; % Size of 1D basis
nq_sub = 128;       % Quadrature points
% --- Hyperbolic Cross Truncation Parameters ---
use_truncation = true; 
hc_multiplier = 1;
hc_threshold = max(1, max_mode) * hc_multiplier; 
fprintf('Initializing Nitrogen Atom SG-CI Solver (7-Electron Extreme Speed)...\n');
fprintf('rc = %.1f, Basis Order = %d (1D Size: %d)\n', rc, max_mode, N);
fprintf('Hyperbolic Cross Threshold: %d\n', hc_threshold);
%% 2. Load Coefficients
data_dir = fullfile(fileparts(mfilename('fullpath')), '..','..','..', 'data');
S1 = load(fullfile(data_dir, 'E_k_rc20.mat'),'coeffs');
alpha = S1.coeffs;
S2 = load(fullfile(data_dir, 'E_kl_rc20.mat'),'E_kl_total'); E_kl = S2.E_kl_total;
p = size(E_kl,1);
%% 3. Basis Construction (Fast Vectorized SG-CI for 7 Electrons)
fprintf('Constructing Basis Space (Vectorized Divide & Conquer)...\n');
tic;
range_modes = -max_mode:max_mode;
% --- Step 1: Generate Spin-Up combinations (m < n < l < v) ---
up_combs = nchoosek(range_modes, 4);
val_up = max(1, abs(up_combs));
prod_up = prod(val_up, 2);
valid_up_mask = (prod_up <= hc_threshold);
valid_up = up_combs(valid_up_mask, :);
prod_up_valid = prod_up(valid_up_mask);
% --- Step 2: Generate Spin-Down combinations (s < t < u) ---
dn_combs = nchoosek(range_modes, 3);
val_dn = max(1, abs(dn_combs));
prod_dn = prod(val_dn, 2);
valid_dn_mask = (prod_dn <= hc_threshold);
valid_dn = dn_combs(valid_dn_mask, :);
prod_dn_valid = prod_dn(valid_dn_mask);
% --- Step 3: Combine and apply global SG Truncation ---
N_up = size(valid_up, 1);
N_dn = size(valid_dn, 1);
est_states = min(N_up * N_dn, 1000000); 
indices_buffer = zeros(est_states, 7);
count = 0;
for i = 1:N_up
    current_prod_up = prod_up_valid(i);
    match_mask = (current_prod_up .* prod_dn_valid) <= hc_threshold;
    num_matches = sum(match_mask);
    
    if num_matches > 0
        if count + num_matches > size(indices_buffer, 1)
            indices_buffer = [indices_buffer; zeros(est_states, 7)];
        end
        matched_dn = valid_dn(match_mask, :);
        rep_up = repmat(valid_up(i, :), num_matches, 1);
        indices_buffer(count+1 : count+num_matches, :) = [rep_up, matched_dn];
        count = count + num_matches;
    end
end
indices = indices_buffer(1:count, :);
N_states = size(indices, 1);
fprintf('Total SG DoFs (Nitrogen, 7-Electron): %d (Generated in %.3f sec)\n', N_states, toc);
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
%% 7. Global Matrix Assembly (GPU + VRAM Autopilot + Aggressive Clear)
fprintf('Assembling Global Matrices...\n');
if parallel.gpu.GPUDevice.isAvailable
    g = gpuDevice; reset(g); 
    fprintf('Detected GPU: %s (VRAM Available: %.1f GB)\n', g.Name, g.AvailableMemory/1e9);
else
    error('GPU not found. GPU is strictly required for 7e assembly.');
end
G_M1 = gpuArray(M1); G_T1 = gpuArray(T1_raw); G_Vs = gpuArray(V_s);
G_VD_raw = gpuArray(V_D_raw); G_indices = gpuArray(indices); 
N2 = N^2;

% Tuned chunk size for optimal VRAM usage (7e has more cross terms)
mem_per_row = N_states * 8 * 200; 
safe_vram = g.AvailableMemory * 0.6; 
gpu_chunk_size = floor(safe_vram / mem_per_row);
gpu_chunk_size = max(20, min(gpu_chunk_size, 300));

num_chunks = ceil(N_states / gpu_chunk_size);
fprintf('  -> Autopilot selected Chunk Size: %d (~%d Chunks total)\n', gpu_chunk_size, num_chunks);
offset = max_mode + 1;
sparsity_threshold = 1e-7; 
est_nnz = min(N_states * 2000, N_states^2);
I_M = zeros(est_nnz, 1); J_M = zeros(est_nnz, 1); V_M = zeros(est_nnz, 1);
I_H = zeros(est_nnz, 1); J_H = zeros(est_nnz, 1); V_H = zeros(est_nnz, 1);
idx_M = 1; idx_H = 1;
t_start = tic;


cof_3x3 = @(a11,a12,a13, a21,a22,a23, a31,a32,a33) ...
    a11.*(a22.*a33 - a23.*a32) - a12.*(a21.*a33 - a23.*a31) + a13.*(a21.*a32 - a22.*a31);

for k = 1:num_chunks
    s_idx = (k-1)*gpu_chunk_size + 1; e_idx = min(k*gpu_chunk_size, N_states);
    current_rows = s_idx:e_idx;
    
    % Up-Spin indices (m, n, l, v)
    g_m_bra = G_indices(current_rows, 1) + offset; g_n_bra = G_indices(current_rows, 2) + offset; 
    g_l_bra = G_indices(current_rows, 3) + offset; g_v_bra = G_indices(current_rows, 4) + offset;
    g_m_ket = G_indices(:, 1) + offset; g_n_ket = G_indices(:, 2) + offset; 
    g_l_ket = G_indices(:, 3) + offset; g_v_ket = G_indices(:, 4) + offset;
    
    % Down-Spin indices (s, t, u)
    g_s_bra = G_indices(current_rows, 5) + offset; g_t_bra = G_indices(current_rows, 6) + offset; g_u_bra = G_indices(current_rows, 7) + offset;
    g_s_ket = G_indices(:, 5) + offset; g_t_ket = G_indices(:, 6) + offset; g_u_ket = G_indices(:, 7) + offset;
    
    get_1D = @(Mat, br, kt) Mat(br + (kt.' - 1) * N);
    
    % --- UP Spin: 4x4 Determinant & Cofactors ---
    Mup_11 = get_1D(G_M1, g_m_bra, g_m_ket); Mup_12 = get_1D(G_M1, g_m_bra, g_n_ket); Mup_13 = get_1D(G_M1, g_m_bra, g_l_ket); Mup_14 = get_1D(G_M1, g_m_bra, g_v_ket);
    Mup_21 = get_1D(G_M1, g_n_bra, g_m_ket); Mup_22 = get_1D(G_M1, g_n_bra, g_n_ket); Mup_23 = get_1D(G_M1, g_n_bra, g_l_ket); Mup_24 = get_1D(G_M1, g_n_bra, g_v_ket);
    Mup_31 = get_1D(G_M1, g_l_bra, g_m_ket); Mup_32 = get_1D(G_M1, g_l_bra, g_n_ket); Mup_33 = get_1D(G_M1, g_l_bra, g_l_ket); Mup_34 = get_1D(G_M1, g_l_bra, g_v_ket);
    Mup_41 = get_1D(G_M1, g_v_bra, g_m_ket); Mup_42 = get_1D(G_M1, g_v_bra, g_n_ket); Mup_43 = get_1D(G_M1, g_v_bra, g_l_ket); Mup_44 = get_1D(G_M1, g_v_bra, g_v_ket);
    
    Dup_11 =  cof_3x3(Mup_22, Mup_23, Mup_24, Mup_32, Mup_33, Mup_34, Mup_42, Mup_43, Mup_44);
    Dup_12 = -cof_3x3(Mup_21, Mup_23, Mup_24, Mup_31, Mup_33, Mup_34, Mup_41, Mup_43, Mup_44);
    Dup_13 =  cof_3x3(Mup_21, Mup_22, Mup_24, Mup_31, Mup_32, Mup_34, Mup_41, Mup_42, Mup_44);
    Dup_14 = -cof_3x3(Mup_21, Mup_22, Mup_23, Mup_31, Mup_32, Mup_33, Mup_41, Mup_42, Mup_43);
    
    Dup_21 = -cof_3x3(Mup_12, Mup_13, Mup_14, Mup_32, Mup_33, Mup_34, Mup_42, Mup_43, Mup_44);
    Dup_22 =  cof_3x3(Mup_11, Mup_13, Mup_14, Mup_31, Mup_33, Mup_34, Mup_41, Mup_43, Mup_44);
    Dup_23 = -cof_3x3(Mup_11, Mup_12, Mup_14, Mup_31, Mup_32, Mup_34, Mup_41, Mup_42, Mup_44);
    Dup_24 =  cof_3x3(Mup_11, Mup_12, Mup_13, Mup_31, Mup_32, Mup_33, Mup_41, Mup_42, Mup_43);
    
    Dup_31 =  cof_3x3(Mup_12, Mup_13, Mup_14, Mup_22, Mup_23, Mup_24, Mup_42, Mup_43, Mup_44);
    Dup_32 = -cof_3x3(Mup_11, Mup_13, Mup_14, Mup_21, Mup_23, Mup_24, Mup_41, Mup_43, Mup_44);
    Dup_33 =  cof_3x3(Mup_11, Mup_12, Mup_14, Mup_21, Mup_22, Mup_24, Mup_41, Mup_42, Mup_44);
    Dup_34 = -cof_3x3(Mup_11, Mup_12, Mup_13, Mup_21, Mup_22, Mup_23, Mup_41, Mup_42, Mup_43);
    
    Dup_41 = -cof_3x3(Mup_12, Mup_13, Mup_14, Mup_22, Mup_23, Mup_24, Mup_32, Mup_33, Mup_34);
    Dup_42 =  cof_3x3(Mup_11, Mup_13, Mup_14, Mup_21, Mup_23, Mup_24, Mup_31, Mup_33, Mup_34);
    Dup_43 = -cof_3x3(Mup_11, Mup_12, Mup_14, Mup_21, Mup_22, Mup_24, Mup_31, Mup_32, Mup_34);
    Dup_44 =  cof_3x3(Mup_11, Mup_12, Mup_13, Mup_21, Mup_22, Mup_23, Mup_31, Mup_32, Mup_33);
    
    M_up = Mup_11.*Dup_11 + Mup_12.*Dup_12 + Mup_13.*Dup_13 + Mup_14.*Dup_14;
    
    % --- DOWN Spin: 3x3 Determinant & Cofactors ---
    Mdn_11 = get_1D(G_M1, g_s_bra, g_s_ket); Mdn_12 = get_1D(G_M1, g_s_bra, g_t_ket); Mdn_13 = get_1D(G_M1, g_s_bra, g_u_ket);
    Mdn_21 = get_1D(G_M1, g_t_bra, g_s_ket); Mdn_22 = get_1D(G_M1, g_t_bra, g_t_ket); Mdn_23 = get_1D(G_M1, g_t_bra, g_u_ket);
    Mdn_31 = get_1D(G_M1, g_u_bra, g_s_ket); Mdn_32 = get_1D(G_M1, g_u_bra, g_t_ket); Mdn_33 = get_1D(G_M1, g_u_bra, g_u_ket);
    
    Ddn11 = Mdn_22.*Mdn_33 - Mdn_23.*Mdn_32; Ddn12 = -(Mdn_21.*Mdn_33 - Mdn_23.*Mdn_31); Ddn13 = Mdn_21.*Mdn_32 - Mdn_22.*Mdn_31;
    Ddn21 = -(Mdn_12.*Mdn_33 - Mdn_13.*Mdn_32); Ddn22 = Mdn_11.*Mdn_33 - Mdn_13.*Mdn_31; Ddn23 = -(Mdn_11.*Mdn_32 - Mdn_12.*Mdn_31);
    Ddn31 = Mdn_12.*Mdn_23 - Mdn_13.*Mdn_22; Ddn32 = -(Mdn_11.*Mdn_23 - Mdn_13.*Mdn_21); Ddn33 = Mdn_11.*Mdn_22 - Mdn_12.*Mdn_21;
    M_dn = Mdn_11.*Ddn11 + Mdn_12.*Ddn12 + Mdn_13.*Ddn13;
    
    % 1. Global Mass Element
    Mg_local = M_up .* M_dn;
    
    % 2. Kinetic Elements
    Tup_11 = get_1D(G_T1, g_m_bra, g_m_ket); Tup_12 = get_1D(G_T1, g_m_bra, g_n_ket); Tup_13 = get_1D(G_T1, g_m_bra, g_l_ket); Tup_14 = get_1D(G_T1, g_m_bra, g_v_ket);
    Tup_21 = get_1D(G_T1, g_n_bra, g_m_ket); Tup_22 = get_1D(G_T1, g_n_bra, g_n_ket); Tup_23 = get_1D(G_T1, g_n_bra, g_l_ket); Tup_24 = get_1D(G_T1, g_n_bra, g_v_ket);
    Tup_31 = get_1D(G_T1, g_l_bra, g_m_ket); Tup_32 = get_1D(G_T1, g_l_bra, g_n_ket); Tup_33 = get_1D(G_T1, g_l_bra, g_l_ket); Tup_34 = get_1D(G_T1, g_l_bra, g_v_ket);
    Tup_41 = get_1D(G_T1, g_v_bra, g_m_ket); Tup_42 = get_1D(G_T1, g_v_bra, g_n_ket); Tup_43 = get_1D(G_T1, g_v_bra, g_l_ket); Tup_44 = get_1D(G_T1, g_v_bra, g_v_ket);
    T_up = Tup_11.*Dup_11 + Tup_12.*Dup_12 + Tup_13.*Dup_13 + Tup_14.*Dup_14 ...
         + Tup_21.*Dup_21 + Tup_22.*Dup_22 + Tup_23.*Dup_23 + Tup_24.*Dup_24 ...
         + Tup_31.*Dup_31 + Tup_32.*Dup_32 + Tup_33.*Dup_33 + Tup_34.*Dup_34 ...
         + Tup_41.*Dup_41 + Tup_42.*Dup_42 + Tup_43.*Dup_43 + Tup_44.*Dup_44;
    clear Tup_11 Tup_12 Tup_13 Tup_14 Tup_21 Tup_22 Tup_23 Tup_24 Tup_31 Tup_32 Tup_33 Tup_34 Tup_41 Tup_42 Tup_43 Tup_44;
    
    Tdn_11 = get_1D(G_T1, g_s_bra, g_s_ket); Tdn_12 = get_1D(G_T1, g_s_bra, g_t_ket); Tdn_13 = get_1D(G_T1, g_s_bra, g_u_ket);
    Tdn_21 = get_1D(G_T1, g_t_bra, g_s_ket); Tdn_22 = get_1D(G_T1, g_t_bra, g_t_ket); Tdn_23 = get_1D(G_T1, g_t_bra, g_u_ket);
    Tdn_31 = get_1D(G_T1, g_u_bra, g_s_ket); Tdn_32 = get_1D(G_T1, g_u_bra, g_t_ket); Tdn_33 = get_1D(G_T1, g_u_bra, g_u_ket);
    T_dn = Tdn_11.*Ddn11 + Tdn_12.*Ddn12 + Tdn_13.*Ddn13 + Tdn_21.*Ddn21 + Tdn_22.*Ddn22 + Tdn_23.*Ddn23 + Tdn_31.*Ddn31 + Tdn_32.*Ddn32 + Tdn_33.*Ddn33;
    clear Tdn_11 Tdn_12 Tdn_13 Tdn_21 Tdn_22 Tdn_23 Tdn_31 Tdn_32 Tdn_33;
    
    T_total = (T_up .* M_dn + T_dn .* M_up) * (1 / (2 * rc^2));
    clear T_up T_dn;
    
    % 3. Nuclear Potential Elements (V_S)
    Vsup_11 = get_1D(G_Vs, g_m_bra, g_m_ket); Vsup_12 = get_1D(G_Vs, g_m_bra, g_n_ket); Vsup_13 = get_1D(G_Vs, g_m_bra, g_l_ket); Vsup_14 = get_1D(G_Vs, g_m_bra, g_v_ket);
    Vsup_21 = get_1D(G_Vs, g_n_bra, g_m_ket); Vsup_22 = get_1D(G_Vs, g_n_bra, g_n_ket); Vsup_23 = get_1D(G_Vs, g_n_bra, g_l_ket); Vsup_24 = get_1D(G_Vs, g_n_bra, g_v_ket);
    Vsup_31 = get_1D(G_Vs, g_l_bra, g_m_ket); Vsup_32 = get_1D(G_Vs, g_l_bra, g_n_ket); Vsup_33 = get_1D(G_Vs, g_l_bra, g_l_ket); Vsup_34 = get_1D(G_Vs, g_l_bra, g_v_ket);
    Vsup_41 = get_1D(G_Vs, g_v_bra, g_m_ket); Vsup_42 = get_1D(G_Vs, g_v_bra, g_n_ket); Vsup_43 = get_1D(G_Vs, g_v_bra, g_l_ket); Vsup_44 = get_1D(G_Vs, g_v_bra, g_v_ket);
    Vs_up = Vsup_11.*Dup_11 + Vsup_12.*Dup_12 + Vsup_13.*Dup_13 + Vsup_14.*Dup_14 ...
          + Vsup_21.*Dup_21 + Vsup_22.*Dup_22 + Vsup_23.*Dup_23 + Vsup_24.*Dup_24 ...
          + Vsup_31.*Dup_31 + Vsup_32.*Dup_32 + Vsup_33.*Dup_33 + Vsup_34.*Dup_34 ...
          + Vsup_41.*Dup_41 + Vsup_42.*Dup_42 + Vsup_43.*Dup_43 + Vsup_44.*Dup_44;
    clear Vsup_11 Vsup_12 Vsup_13 Vsup_14 Vsup_21 Vsup_22 Vsup_23 Vsup_24 Vsup_31 Vsup_32 Vsup_33 Vsup_34 Vsup_41 Vsup_42 Vsup_43 Vsup_44;
    
    Vsdn_11 = get_1D(G_Vs, g_s_bra, g_s_ket); Vsdn_12 = get_1D(G_Vs, g_s_bra, g_t_ket); Vsdn_13 = get_1D(G_Vs, g_s_bra, g_u_ket);
    Vsdn_21 = get_1D(G_Vs, g_t_bra, g_s_ket); Vsdn_22 = get_1D(G_Vs, g_t_bra, g_t_ket); Vsdn_23 = get_1D(G_Vs, g_t_bra, g_u_ket);
    Vsdn_31 = get_1D(G_Vs, g_u_bra, g_s_ket); Vsdn_32 = get_1D(G_Vs, g_u_bra, g_t_ket); Vsdn_33 = get_1D(G_Vs, g_u_bra, g_u_ket);
    Vs_dn = Vsdn_11.*Ddn11 + Vsdn_12.*Ddn12 + Vsdn_13.*Ddn13 + Vsdn_21.*Ddn21 + Vsdn_22.*Ddn22 + Vsdn_23.*Ddn23 + Vsdn_31.*Ddn31 + Vsdn_32.*Ddn32 + Vsdn_33.*Ddn33;
    clear Vsdn_11 Vsdn_12 Vsdn_13 Vsdn_21 Vsdn_22 Vsdn_23 Vsdn_31 Vsdn_32 Vsdn_33;
    
    Vs_total = (Vs_up .* M_dn + Vs_dn .* M_up) * (-Z / rc);
    clear Vs_up Vs_dn;
    
    % 4. Electron-Electron Repulsion (V_D) - EXTREME UNROLLING
    % 4a. Same-Spin (UP-UP) - 4 particles -> 6 pairs -> 36 terms 
    P1_up = {g_m_bra, g_m_bra, g_m_bra, g_n_bra, g_n_bra, g_l_bra}; 
    P2_up = {g_n_bra, g_l_bra, g_v_bra, g_l_bra, g_v_bra, g_v_bra};
    Q1_up = {g_m_ket, g_m_ket, g_m_ket, g_n_ket, g_n_ket, g_l_ket}; 
    Q2_up = {g_n_ket, g_l_ket, g_v_ket, g_l_ket, g_v_ket, g_v_ket};
    

    rem_r = {[3,4], [2,4], [2,3], [1,4], [1,3], [1,2]};
    sgn_r = [-1, 1, -1, -1, 1, -1];
    Mup_cell = {Mup_11, Mup_12, Mup_13, Mup_14; Mup_21, Mup_22, Mup_23, Mup_24; Mup_31, Mup_32, Mup_33, Mup_34; Mup_41, Mup_42, Mup_43, Mup_44};
    Gamma_up = cell(6,6);
    for I=1:6
        for J=1:6
            r1 = rem_r{I}(1); r2 = rem_r{I}(2);
            c1 = rem_r{J}(1); c2 = rem_r{J}(2);
            Gamma_up{I,J} = (sgn_r(I)*sgn_r(J)) .* (Mup_cell{r1,c1}.*Mup_cell{r2,c2} - Mup_cell{r1,c2}.*Mup_cell{r2,c1});
        end
    end
    
    Vee_up_up = zeros(size(Mg_local), 'gpuArray'); 
    for I = 1:6
        for J = 1:6
            i1 = P1_up{I} + (Q1_up{J}.' - 1)*N;
            i2 = P2_up{I} + (Q2_up{J}.' - 1)*N;
            i3 = P1_up{I} + (Q2_up{J}.' - 1)*N;
            i4 = P2_up{I} + (Q1_up{J}.' - 1)*N;
            vd = G_VD_raw(i1 + (i2 - 1)*N2) - G_VD_raw(i3 + (i4 - 1)*N2) - G_VD_raw(i4 + (i3 - 1)*N2) + G_VD_raw(i2 + (i1 - 1)*N2);
            Vee_up_up = Vee_up_up + vd .* Gamma_up{I,J};
        end
    end
    Vee_up_up = Vee_up_up .* M_dn;
    clear Mup_cell Mup_11 Mup_12 Mup_13 Mup_14 Mup_21 Mup_22 Mup_23 Mup_24 Mup_31 Mup_32 Mup_33 Mup_34 Mup_41 Mup_42 Mup_43 Mup_44 Gamma_up P1_up P2_up Q1_up Q2_up;
    
    % 4b. Same-Spin (DOWN-DOWN) - 3 particles -> 3 pairs -> 9 terms
    P1_dn = {g_s_bra, g_s_bra, g_t_bra}; P2_dn = {g_t_bra, g_u_bra, g_u_bra};
    Q1_dn = {g_s_ket, g_s_ket, g_t_ket}; Q2_dn = {g_t_ket, g_u_ket, g_u_ket};
    Gamma_dn = {Mdn_33, -Mdn_32, Mdn_31; -Mdn_23, Mdn_22, -Mdn_21; Mdn_13, -Mdn_12, Mdn_11};
    
    Vee_dn_dn = zeros(size(Mg_local), 'gpuArray');
    for I = 1:3
        for J = 1:3
            i1 = P1_dn{I} + (Q1_dn{J}.' - 1)*N;
            i2 = P2_dn{I} + (Q2_dn{J}.' - 1)*N;
            i3 = P1_dn{I} + (Q2_dn{J}.' - 1)*N;
            i4 = P2_dn{I} + (Q1_dn{J}.' - 1)*N;
            vd = G_VD_raw(i1 + (i2 - 1)*N2) - G_VD_raw(i3 + (i4 - 1)*N2) - G_VD_raw(i4 + (i3 - 1)*N2) + G_VD_raw(i2 + (i1 - 1)*N2);
            Vee_dn_dn = Vee_dn_dn + vd .* Gamma_dn{I, J};
        end
    end
    Vee_dn_dn = Vee_dn_dn .* M_up;
    clear Mdn_11 Mdn_12 Mdn_13 Mdn_21 Mdn_22 Mdn_23 Mdn_31 Mdn_32 Mdn_33 Gamma_dn P1_dn P2_dn Q1_dn Q2_dn M_up M_dn;
    
    % 4c. Cross-Spin (UP-DOWN) - 4x3 particles -> 4x4 x 3x3 = 144 terms
    p_up = {g_m_bra, g_n_bra, g_l_bra, g_v_bra}; q_up = {g_m_ket, g_n_ket, g_l_ket, g_v_ket};
    p_dn = {g_s_bra, g_t_bra, g_u_bra}; q_dn = {g_s_ket, g_t_ket, g_u_ket};
    D_up_cell = {Dup_11, Dup_12, Dup_13, Dup_14; Dup_21, Dup_22, Dup_23, Dup_24; Dup_31, Dup_32, Dup_33, Dup_34; Dup_41, Dup_42, Dup_43, Dup_44};
    D_dn_cell = {Ddn11, Ddn12, Ddn13; Ddn21, Ddn22, Ddn23; Ddn31, Ddn32, Ddn33};
    
    Vee_cross = zeros(size(Mg_local), 'gpuArray');
    for i_up = 1:4
        for j_up = 1:4
            idx_up = p_up{i_up} + (q_up{j_up}.' - 1)*N;
            D_u = D_up_cell{i_up, j_up};
            for i_dn = 1:3
                for j_dn = 1:3
                    idx_dn = p_dn{i_dn} + (q_dn{j_dn}.' - 1)*N;
                    vd = G_VD_raw(idx_up + (idx_dn - 1)*N2);
                    Vee_cross = Vee_cross + vd .* D_u .* D_dn_cell{i_dn, j_dn};
                end
            end
        end
    end
    clear idx_up idx_dn vd D_u D_up_cell D_dn_cell p_up q_up p_dn q_dn Dup_11 Dup_12 Dup_13 Dup_14 Dup_21 Dup_22 Dup_23 Dup_24 Dup_31 Dup_32 Dup_33 Dup_34 Dup_41 Dup_42 Dup_43 Dup_44 Ddn11 Ddn12 Ddn13 Ddn21 Ddn22 Ddn23 Ddn31 Ddn32 Ddn33;
    
    Vee_total = (0.5*(Vee_up_up + Vee_dn_dn) + Vee_cross) * (1 / rc);
    clear Vee_up_up Vee_dn_dn Vee_cross;
    
    Hg_local = T_total + Vs_total + Vee_total;
    clear T_total Vs_total Vee_total;
    
    % --- Gather and sparse extraction ---
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
Hg = (Hg + Hg') / 2; Mg = (Mg + Mg') / 2;
fprintf('Solving Generalized Eigenvalue Problem...\n');
opts.p = 16; 
opts.maxit = 1e8; 
opts.tol = 1e-7; 
opts.isreal = true; 
try
    [V, D] = eigs(Hg, Mg, 1, 'sa', opts);
    fprintf('\nResults (Ground State):\nE_0 = %.8f a.u.\n', D(1,1));
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