% =========================================================================
% Spectral Element Method for 1D Hydrogen Atom (Soft-Coulomb)
% System: 1 Electron
% Ansatz: Direct 1D Basis (No inter-electron interaction)
% =========================================================================

%% 1. Parameters
Z = 1;              % Nuclear charge for Hydrogen
rc = 15;            % Cutoff radius
max_mode = 64;     % Can easily push to 100+ for 1D
N = 2*max_mode + 1; % Size of 1D basis
nq_sub = 512;       % Quadrature points

fprintf('Initializing H Atom S-FEM Solver (1D)...\n');
fprintf('rc = %.1f, Basis Order = %d (Size: %d)\n', rc, max_mode, N);

%% 2. Load Coefficients (Only E_k needed for nuclear potential)
data_dir = fullfile(fileparts(mfilename('fullpath')), '..','..','..', 'data');
S1 = load(fullfile(data_dir, 'E_k_rc15.mat'),'coeffs');
alpha = S1.coeffs;
Jneed = size(alpha, 1);

%% 3. Precompute 1D Analytic Matrices (M_s and S_s)
fprintf('Constructing Analytic 1D Matrices...\n');
M1 = zeros(N);
T1_raw = zeros(N);
modes = -max_mode:max_mode;
idx_map = @(k) k + max_mode + 1;
absidx = @(k) abs(k);

for ii = 1:N
    k1 = modes(ii); i = idx_map(k1);
    for jj = 1:N
        k2 = modes(jj); j = idx_map(k2);
        
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

%% 4. Quadrature & Potential Setup
fprintf('Computing Potential Matrix...\n');
[x_ref, w_ref] = legendre_gauss(nq_sub);
s_pos = (x_ref + 1) / 2; w_pos = w_ref / 2;
s_neg = (x_ref - 1) / 2; w_neg = w_ref / 2;

pos_modes = modes(modes > 0); pos_idx_local = arrayfun(idx_map, pos_modes);
neg_modes = modes(modes < 0); neg_idx_local = arrayfun(idx_map, neg_modes);
zero_idx_local = idx_map(0);
np = numel(pos_modes); nn = numel(neg_modes);

Phi_pos_sub = zeros(np, nq_sub); xarg_pos = 2*s_pos - 1;
for ii = 1:np, Phi_pos_sub(ii, :) = legendreP_at_nodes(pos_modes(ii)-1, xarg_pos) - legendreP_at_nodes(pos_modes(ii)+1, xarg_pos); end
Phi_neg_sub = zeros(nn, nq_sub); xarg_neg = -1 - 2*s_neg;
for ii = 1:nn, Phi_neg_sub(ii, :) = legendreP_at_nodes(abs(neg_modes(ii))-1, xarg_neg) - legendreP_at_nodes(abs(neg_modes(ii))+1, xarg_neg); end
Phi0_pos = 1 - s_pos; Phi0_neg = 1 - abs(s_neg);


Wpos = w_pos(:)';
Wneg = w_neg(:)';

Bstack = zeros(N, N, Jneed);
for j_idx = 1:Jneed
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

V_s = reshape(reshape(Bstack, [], Jneed) * alpha(:), N, N);

%% 5. Global Assembly & Solve
fprintf('Assembling Global Matrices...\n');
Mg = M1;
Hg = T1_raw * (1 / (2 * rc^2)) + V_s * (-Z / rc);

% Ensure exact symmetry for numerical stability
Hg = (Hg + Hg') / 2; 
Mg = (Mg + Mg') / 2;

fprintf('Solving Eigenvalue Problem (Shift-and-Invert)...\n');
opts.p = 32;
opts.maxit = 1000; 
opts.tol = 1e-12;
opts.issym = true;




[V, D] = eigs(Hg, Mg, 16, 'sa', opts);
eigenvalue = diag(D);
fprintf('\nResults (Ground State):\nE_0 = %.16f a.u.\n', eigenvalue);

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