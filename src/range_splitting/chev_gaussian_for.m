%%  load
data_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'data');
sData = load(fullfile(data_dir, 's_wbt_rc20_1e-12.mat'));
wData = load(fullfile(data_dir, 'w_wbt_rc20_1e-12.mat'));
s = sData.s_wbt;   
w = wData.w_wbt;
%% Projection
N_list = [1124,1124,984,824,768,656,518,472, 448, 400, 348, 308, 256, 232,192, 164, 128, 108, 96, 78, 68, 60, 52,44,36,30,24,32];
r_c = 20;  
num_intervals = 64; 
points_per_interval = 512; 
num_points_test = 256;
x_test = linspace(-r_c, r_c, num_points_test);
y_test = linspace(-r_c, r_c, num_points_test);
[X_test, Y_test] = meshgrid(x_test, y_test);
rho = 1./ sqrt(1+(X_test-Y_test).^2);
if isempty(gcp('nocreate'))
    parpool;
end
E_kl = zeros(N_list(1)+1,N_list(1)+1);
error = zeros(num_points_test,num_points_test);
for index = 1:1
    N = N_list(index);
    M = N;
    sl = s(index);
    wl = w(index);
    fprintf('Processing wl %d, wl = %.12e...\n', index, wl);
    fprintf('Processing sl %d, sl = %.12e...\n', index, sl);
    theta_edges = linspace(-pi/2, pi/2, num_intervals + 1);
    [gl_nodes, gl_weights] = get_gauss_legendre(points_per_interval);
    E_matrix = zeros(N+1, N+1);
    total_blocks = num_intervals^2;
    scale_factors = ones(N+1, N+1) * (4 / pi^2);
    scale_factors(1, :) = 2 / pi^2;
    scale_factors(:, 1) = 2 / pi^2;
    scale_factors(1, 1) = 1 / pi^2;
    parfor block_idx = 1:total_blocks
        [row_idx, col_idx] = ind2sub([num_intervals, num_intervals], block_idx);
        t1_start = theta_edges(col_idx); t1_end = theta_edges(col_idx+1);
        t2_start = theta_edges(row_idx); t2_end = theta_edges(row_idx+1);
        theta_x = (t1_end - t1_start)/2 * gl_nodes + (t1_end + t1_start)/2;
        w_x = (t1_end - t1_start)/2 * gl_weights; 
        theta_y = (t2_end - t2_start)/2 * gl_nodes + (t2_end + t2_start)/2;
        w_y = (t2_end - t2_start)/2 * gl_weights;
        x_local = sin(theta_x);
        y_local = sin(theta_y);
        [XX_local, YY_local] = meshgrid(x_local, y_local);
        [WX, WY] = meshgrid(w_x, w_y);
        W_local = WX .* WY;
%         kernel_vals = wl * exp(-(XX_local - YY_local).^2 * sl * r_c^2 );
        kernel_vals = 1 ./ sqrt((1/r_c)^2+(XX_local-YY_local).^2);
        Tx_mat = compute_cheb_matrix(x_local, N);
        Ty_mat = compute_cheb_matrix(y_local, N);
        E_partial = Ty_mat' * (kernel_vals .* W_local) * Tx_mat;
        E_matrix = E_matrix + E_partial;
    end
    E_final = E_matrix .* scale_factors;
    E_final = (E_final + E_final') ./ 2;
    [K_grid, L_grid] = meshgrid(0:N, 0:N); 
    mask = (K_grid + L_grid <= M) & (mod(K_grid + L_grid, 2) == 0);
    E_final = E_final .* mask'; 
    E_final = E_final';
    E_index = zeros(N_list(1)+1,N_list(1)+1);
    E_index(1:N_list(index)+1 , 1:N_list(index)+1) = E_final;
    E_kl = E_kl + E_index;
    Tx_test = compute_cheb_matrix(x_test(:)./r_c, N); % Size: num_test x (N+1)
    Ty_test = compute_cheb_matrix(y_test(:)./r_c, N);
    approximation = Tx_test * E_final * Ty_test';
%     gaussian_kernel_exact = wl * exp(-(X_test - Y_test).^2 * sl);
    gaussian_kernel_exact = 1 ./ sqrt((1/r_c)^2+(X_test - Y_test).^2);
    error_surface = abs(gaussian_kernel_exact - approximation);
    max_err = max(error_surface, [], 'all');
    fprintf('Max Error: %e\n', max_err);
    rel_error_surface = abs(gaussian_kernel_exact - approximation)./gaussian_kernel_exact;
    rel_max_err = max(rel_error_surface, [], 'all');
    fprintf('Rel Error: %e\n', rel_max_err);
    plot_results(X_test, Y_test, gaussian_kernel_exact, approximation, error_surface);
end


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

function [x, w] = get_gauss_legendre(n)

    beta = .5 ./ sqrt(1-(2*(1:n-1)).^(-2));
    T = diag(beta,1) + diag(beta,-1);
    [V, D] = eig(T);
    x = diag(D); 
    [x, idx] = sort(x); 
    w = 2 * V(1,idx).^2; 
    w = w(:); 
end

function plot_results(X, Y, Exact, Approx, Error)
    figure('Name', 'Vectorized Chebyshev Expansion', 'Color', 'w', 'Position', [100, 100, 1200, 400]);
    
    subplot(1, 3, 1);
    surf(X, Y, Exact, 'EdgeColor', 'none');
    title('Exact Gaussian Kernel');
    xlabel('x'); ylabel('y'); axis tight;  colorbar;
    
    subplot(1, 3, 2);
    surf(X, Y, Approx, 'EdgeColor', 'none');
    title('Chebyshev Approximation');
    xlabel('x'); ylabel('y'); axis tight;  colorbar;
    
    subplot(1, 3, 3);
    surf(X, Y, Error, 'EdgeColor', 'none');
    title('Error Surface');
    xlabel('x'); ylabel('y'); axis tight;  colorbar;
end
