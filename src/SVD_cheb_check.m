clear; clc; close all;
data_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
E_Data = load(fullfile(data_dir, 'E_kl_SOG_rc40_1e-12.mat'));
E = E_Data.E_kl_total;
E = 0.5 * (E + E');
N = size(E,1)-1;
fprintf('Chebyshev Degree (N): %d\n', N);

[V, D_mat] = eig(E, 'vector');

[abs_eigvals, idx] = sort(abs(D_mat), 'descend');
sorted_eigvals = D_mat(idx); 
sorted_V = V(:, idx);        

figure('Name', 'EVD Analysis', 'Color', 'w');
semilogy(abs_eigvals, 'bo-', 'LineWidth', 1.5, 'MarkerSize', 4);
grid on;
title('Decay of Eigenvalues |\lambda_k|');
xlabel('Index k');
ylabel('Magnitude |\lambda_k|');
xlim([1, length(abs_eigvals)]);

truncation_tol = 1e-14; 
rank_r = find(abs_eigvals < truncation_tol, 1) - 1;

if isempty(rank_r)
    rank_r = length(abs_eigvals);
end

fprintf('\n=== EVD Truncation Analysis ===\n');
fprintf('Truncation Tolerance: %e\n', truncation_tol);
fprintf('Selected Rank r: %d (out of %d)\n', rank_r, N+1);
fprintf('Eigenvalue at truncation (|\lambda_{r+1}|): %e\n', abs_eigvals(rank_r+1));

V_r = sorted_V(:, 1:rank_r);      % Size: (N+1) x r
lambda_r = sorted_eigvals(1:rank_r); % Size: r x 1

numel = 256; 
r_c = 40;
r_grid = linspace(-r_c,r_c,numel);
[R1, R2] = meshgrid(r_grid, r_grid);

rho_exact = 1 ./ sqrt(1 + (R1 - R2).^2);

T_mat = compute_cheb_matrix(r_grid./r_c, N); 

approx_full = T_mat * E * T_mat';

W = T_mat * V_r; 
W_scaled = W .* lambda_r.'; 
approx_evd = W_scaled * W'; 


err_full = abs(rho_exact - approx_full);
max_err_full = max(err_full, [], 'all');
rel_err_full = max(err_full ./ abs(rho_exact), [], 'all');

err_evd = abs(rho_exact - approx_evd);
max_err_evd = max(err_evd, [], 'all');
rel_err_evd = max(err_evd ./ abs(rho_exact), [], 'all');

err_trunc = abs(approx_full - approx_evd);
max_err_trunc = max(err_trunc, [], 'all');

fprintf('\n=== Error Analysis (Grid Size: %dx%d) ===\n', numel, numel);
fprintf('1. Full Rank Chebyshev Approximation:\n');
fprintf('   Max Absolute Error: %e\n', max_err_full);
fprintf('   Max Relative Error: %e\n', rel_err_full);

fprintf('2. Rank-%d EVD Approximation (Fast Assembly):\n', rank_r);
fprintf('   Max Absolute Error: %e\n', max_err_evd);
fprintf('   Max Relative Error: %e\n', rel_err_evd);

fprintf('3. Truncation Error (Full - Reduced):\n');
fprintf('   Max Difference: %e\n', max_err_trunc);

figure('Name', 'EVD Approximation Error', 'Color', 'w', 'Position', [100, 100, 1400, 400]);

% Plot 1: Exact
subplot(1, 4, 1);
imagesc(r_grid, r_grid, rho_exact);
title('Exact Soft Coulomb');
axis square; colorbar; xlabel('r1'); ylabel('r2');

% Plot 2: EVD Approx
subplot(1, 4, 2);
imagesc(r_grid, r_grid, approx_evd);
title(sprintf('Rank-%d EVD Assembly', rank_r));
axis square; colorbar; xlabel('r1'); ylabel('r2');

% Plot 3: Absolute Error (EVD)
subplot(1, 4, 3);
imagesc(r_grid, r_grid, log10(err_evd + eps)); 
title('Log10 Abs Error (EVD)');
axis square; colorbar; xlabel('r1'); ylabel('r2');
clim([-16, -2]); 

% Plot 4: Relative Error (EVD)
subplot(1, 4, 4);
imagesc(r_grid, r_grid, log10(err_evd ./ abs(rho_exact) + eps));
title('Log10 Rel Error (EVD)');
axis square; colorbar; xlabel('r1'); ylabel('r2');

sgtitle(sprintf('EVD Low-Rank Assembly (Rank r=%d)', rank_r));

function T_mat = compute_cheb_matrix(x, N)
    x = x(:); 
    m = length(x);
    T_mat = zeros(m, N + 1);
    T_mat(:, 1) = 1;          
    if N >= 1
        T_mat(:, 2) = x;     
    end
    for k = 2:N
        T_mat(:, k+1) = 2 * x .* T_mat(:, k) - T_mat(:, k-1);
    end
end
