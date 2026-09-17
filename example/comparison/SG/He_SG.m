% =========================================================================
% Full Grid Spectral Element Method for 1D He Atom (Soft-Coulomb)
% System: 2 Electrons (Singlet State, 1 Spin-Up, 1 Spin-Down)
% Ansatz: Sparse Grid Configuration Interaction (SG-CI)
% Formula: |k|_mix = max(1,|k1|) * max(1,|k2|) <= n_SG
% =========================================================================

%% 1. Parameters
Z = 2;              % Nuclear charge for Helium
rc = 15;            % Cutoff radius
max_mode = 48;      % 1D Basis maximum order
N = 2*max_mode + 1; % Size of 1D basis
nq_sub = 512;       % Quadrature points

% --- SG Truncation Parameter (Based on the formula in your image) ---
% n_SG corresponds to the 'n' in your formula: |\mathbf{k}|_{mix} <= n
% For He, setting it to 2~4 times max_mode captures excellent correlation energy
n_SG = 2048; 

fprintf('Initializing He Atom SG-CI Solver...\n');
fprintf('rc = %.1f, 1D max_mode = %d (1D Size: %d)\n', rc, max_mode, N);
fprintf('Sparse Grid Threshold (n_SG) = %d\n', n_SG);

%% 2. Load Coefficients
% Ensure 'E_k_rc20.mat' and 'E_kl_rc20.mat' are in your current directory
data_dir = fullfile(fileparts(mfilename('fullpath')), '..','..','..', 'data');
S1 = load(fullfile(data_dir, 'E_k_rc15.mat'),'coeffs');
alpha = S1.coeffs;
S2 = load(fullfile(data_dir, 'E_kl_rc15.mat'),'E_kl_total'); E_kl = S2.E_kl_total;
p = size(E_kl,1);

%% 3. Basis Construction & DoFs Calculation (Using your SG Formula)
indices = [];
range_modes = -max_mode:max_mode;
idx_map = @(k) k + max_mode + 1;

for k1 = range_modes
    for k2 = range_modes
        % Calculate |k|_mix = max(1, |k1|) * max(1, |k2|)
        k_mix = max(1, abs(k1)) * max(1, abs(k2));
        
        % SG Truncation Condition: |k|_mix <= n_SG
        if k_mix <= n_SG
            indices = [indices; k1, k2];
        end
    end
end
N_states = size(indices, 1);
fprintf('Total SG DoFs (He, 2-Electron): %d\n', N_states);

%% 4. Precompute 1D Analytic Matrices
fprintf('Constructing Analytic 1D Matrices...\n');
M1 = zeros(N); T1_raw = zeros(N);
absidx = @(k) abs(k);

for ii = 1:N
    k1 = range_modes(ii); i = idx_map(k1);
    for jj = 1:N
        k2 = range_modes(jj); j = idx_map(k2);
        
        % Mass Matrix
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
        
        % Stiffness Matrix
        if k1 == k2
            if k1 == 0, T1_raw(i,j) = 2; else, T1_raw(i,j) = 4*(2*absidx(k1) + 1); end
        else
            T1_raw(i,j) = 0;
        end
    end
end

%% 5. Quadrature & Bstack 
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

Wpos = w_pos(:)'; 
Wneg = w_neg(:)';

parfor j_idx = 1:Jneed
    j_poly = j_idx - 1; 
    Pj_pos = legendreP_at_nodes(j_poly, s_pos); 
    Pj_neg = legendreP_at_nodes(j_poly, s_neg);
    B_local = zeros(N, N);
    
    if np > 0
        coeff = Pj_pos .* Wpos;
        B_local(pos_idx_local, pos_idx_local) = (Phi_pos_sub .* coeff) * Phi_pos_sub'; 
    end
    if nn > 0
        coeff = Pj_neg .* Wneg;
        B_local(neg_idx_local, neg_idx_local) = (Phi_neg_sub .* coeff) * Phi_neg_sub'; 
    end
    if np > 0 && ~isempty(zero_idx_local)
        coeff = Pj_pos .* Wpos;
        B0p = (Phi0_pos .* coeff) * Phi_pos_sub';
        B_local(zero_idx_local, pos_idx_local) = B0p; 
        B_local(pos_idx_local, zero_idx_local) = B0p';
    end
    if nn > 0 && ~isempty(zero_idx_local)
        coeff = Pj_neg .* Wneg;
        B0n = (Phi0_neg .* coeff) * Phi_neg_sub';
        B_local(zero_idx_local, neg_idx_local) = B0n; 
        B_local(neg_idx_local, zero_idx_local) = B0n';
    end
    if ~isempty(zero_idx_local)
        B_local(zero_idx_local, zero_idx_local) = sum(Phi0_pos.^2 .* Pj_pos .* Wpos) + sum(Phi0_neg.^2 .* Pj_neg .* Wneg);
    end
    Bstack(:, :, j_idx) = B_local;
end

V_s = reshape(reshape(Bstack, [], size(alpha,1)) * alpha(:), N, N);
V_vec = reshape(Bstack, N*N, p);
V_D_raw = V_vec * E_kl * V_vec.'; 

%% 6. Global Matrix Assembly (GPU + Triplet)
fprintf('Assembling Global Matrices (GPU)...\n');
if parallel.gpu.GPUDevice.isAvailable
    g = gpuDevice; reset(g);
else
    error('GPU not found, please run on a GPU-enabled machine.');
end

G_M1 = gpuArray(M1); G_T1 = gpuArray(T1_raw); G_Vs = gpuArray(V_s);
G_VD_raw = gpuArray(V_D_raw); G_indices = gpuArray(indices); 

gpu_chunk_size = 2000; % Chunk size for 2E system
num_chunks = ceil(N_states / gpu_chunk_size);
offset = max_mode + 1;
sparsity_threshold = 1e-13; 

est_nnz = min(N_states * 800, N_states^2);
I_M = zeros(est_nnz, 1); J_M = zeros(est_nnz, 1); V_M = zeros(est_nnz, 1);
I_H = zeros(est_nnz, 1); J_H = zeros(est_nnz, 1); V_H = zeros(est_nnz, 1);
idx_M = 1; idx_H = 1;

for k = 1:num_chunks
    s_idx = (k-1)*gpu_chunk_size + 1;
    e_idx = min(k*gpu_chunk_size, N_states);
    current_rows = s_idx:e_idx;
    
    g_m_bra = G_indices(current_rows, 1) + offset;
    g_n_bra = G_indices(current_rows, 2) + offset;
    
    g_m_ket = G_indices(:, 1) + offset;
    g_n_ket = G_indices(:, 2) + offset;
    
    get_1D = @(Mat, br, kt) Mat(br + (kt.' - 1) * N);
    
    M_m1m2 = get_1D(G_M1, g_m_bra, g_m_ket); M_n1n2 = get_1D(G_M1, g_n_bra, g_n_ket);
    T_m1m2 = get_1D(G_T1, g_m_bra, g_m_ket); T_n1n2 = get_1D(G_T1, g_n_bra, g_n_ket);
    V_m1m2 = get_1D(G_Vs, g_m_bra, g_m_ket); V_n1n2 = get_1D(G_Vs, g_n_bra, g_n_ket);
    
    % --- Physics for 2-Electron System (Direct Tensor Product, No Anti-Symmetry enforced here since opposite spins) ---
    Mg_local = M_m1m2 .* M_n1n2;
    T_local = (T_m1m2 .* M_n1n2 + M_m1m2 .* T_n1n2) * (1 / (2 * rc^2));
    V_s_local = (V_m1m2 .* M_n1n2 + M_m1m2 .* V_n1n2) * (-Z / rc);
    
    N2 = N^2;
    get_VD = @(A, B, C, D) G_VD_raw((A + (C.' - 1)*N) + ((B + (D.' - 1)*N) - 1) * N2);
    V_ee_local = get_VD(g_m_bra, g_n_bra, g_m_ket, g_n_ket) * (1 / rc);
    
    Hg_local = T_local + V_s_local + V_ee_local;
    
    % Gather and Sparse Update
    M_cpu = gather(Mg_local); H_cpu = gather(Hg_local);
    [r_M, c_M] = find(abs(M_cpu) > sparsity_threshold);
    [r_H, c_H] = find(abs(H_cpu) > sparsity_threshold);
    
    if ~isempty(r_M)
        len = length(r_M);
        if idx_M+len-1 > length(I_M), I_M=[I_M; zeros(est_nnz,1)]; J_M=[J_M; zeros(est_nnz,1)]; V_M=[V_M; zeros(est_nnz,1)]; end
        I_M(idx_M:idx_M+len-1) = reshape(current_rows(r_M),[],1);
        J_M(idx_M:idx_M+len-1) = reshape(c_M,[],1);
        V_M(idx_M:idx_M+len-1) = reshape(M_cpu(sub2ind(size(M_cpu), r_M, c_M)),[],1);
        idx_M = idx_M + len;
    end
    if ~isempty(r_H)
        len = length(r_H);
        if idx_H+len-1 > length(I_H), I_H=[I_H; zeros(est_nnz,1)]; J_H=[J_H; zeros(est_nnz,1)]; V_H=[V_H; zeros(est_nnz,1)]; end
        I_H(idx_H:idx_H+len-1) = reshape(current_rows(r_H),[],1);
        J_H(idx_H:idx_H+len-1) = reshape(c_H,[],1);
        V_H(idx_H:idx_H+len-1) = reshape(H_cpu(sub2ind(size(H_cpu), r_H, c_H)),[],1);
        idx_H = idx_H + len;
    end
end

Mg = sparse(I_M(1:idx_M-1), J_M(1:idx_M-1), V_M(1:idx_M-1), N_states, N_states);
Hg = sparse(I_H(1:idx_H-1), J_H(1:idx_H-1), V_H(1:idx_H-1), N_states, N_states);
clear I_M J_M V_M I_H J_H V_H

% Force Perfect Symmetry to avoid imaginary eigenvalues from float truncation
Hg = (Hg + Hg') / 2; Mg = (Mg + Mg') / 2;

%% 7. Solve (Shift-and-Invert fix to prevent NaN at high mode!)
fprintf('Solving Generalized Eigenvalue Problem (Shift-and-Invert)...\n');
opts.p = 64; 
opts.maxit = 1e6; 
opts.tol = 1e-14; 
opts.issym = true;



[V, D] = eigs(Hg, Mg, 16, 'sa', opts);

fprintf('\nResults (Ground State):\nE_0 = %.16f a.u.\n', diag(D));

% --- Helpers ---
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