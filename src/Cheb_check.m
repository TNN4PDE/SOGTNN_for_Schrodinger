%% load 
data_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'data');
E_Data = load(fullfile(data_dir, 'E_kl_SOG_rc20_1e-12.mat'));
E = E_Data.E_kl;
N = size(E,1)-1;
numel = 1024;
r_c = 20;
r   = linspace(-r_c,r_c,numel);
[R1,R2] = meshgrid(r,r);
rho = 1 ./ sqrt(1 + (R1 - R2).^2);  

Tx_test = compute_cheb_matrix(r./r_c, N); 
Ty_test = compute_cheb_matrix(r./r_c, N);

approximation = Tx_test * E * Ty_test';

error_surface = abs(rho - approximation);
max_absolute_err = max(error_surface, [], 'all');
fprintf('Max Absolute Error: %e\n', max_absolute_err);
max_relative_err = max(error_surface ./ rho, [], 'all');
fprintf('Max Relative Error: %e\n', max_relative_err);

figure('Name', 'Vectorized Chebyshev Expansion', 'Color', 'w', 'Position', [100, 100, 1200, 400]);

subplot(1, 4, 1);
surf(R1, R2, rho, 'EdgeColor', 'none');
title('Exact Gaussian Kernel');
xlabel('x'); ylabel('y'); axis tight;  colorbar;

subplot(1, 4, 2);
surf(R1, R2,approximation , 'EdgeColor', 'none');
title('Chebyshev Approximation');
xlabel('x'); ylabel('y'); axis tight;  colorbar;

subplot(1, 4, 3);
surf(R1, R2, error_surface, 'EdgeColor', 'none');
title('Error Surface');
xlabel('x'); ylabel('y'); axis tight;  colorbar;

subplot(1, 4, 4);
surf(R1, R2, error_surface./rho, 'EdgeColor', 'none');
title('Error Surface');
xlabel('x'); ylabel('y'); axis tight;  colorbar;


function T_mat = compute_cheb_matrix(x, N)

    x = x(:); 
    m = length(x);
    T_mat = zeros(m, N + 1);
    T_mat(:, 1) = 1;          % T0
    if N >= 1
        T_mat(:, 2) = x;      % T1
    end
    for k = 2:N
        T_mat(:, k+1) = 2 * x .* T_mat(:, k) - T_mat(:, k-1);
    end
end
