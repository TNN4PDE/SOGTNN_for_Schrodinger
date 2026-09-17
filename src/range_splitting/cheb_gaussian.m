%% Chebyshev two-var expansion (vectorized, CPU optimized)
clear; clc; close all;

N = 28;                
b = 1.22749083347315613;
l = -2;         
delta = 2 * b^(2*l); 
num_points = 8192;     

nodes = cos((2*(0:num_points-1)+1) * pi / (2 * num_points))';  % (num_points x 1)

x = linspace(-1, 1, 100);
y = linspace(-1, 1, 100);
[X, Y] = meshgrid(x, y);   
gaussian_kernel = exp(-(X - Y).^2 / delta);
T_nodes = chebyshev_values(nodes, N);    
T_x = chebyshev_values(x(:), N);        
T_y = chebyshev_values(y(:), N);        
D = nodes - nodes.';        
G = exp(-(D.^2) / delta);
S_raw = T_nodes' * (G * T_nodes);    % (N+1 x N+1)

norm_factor = zeros(N+1, N+1);
[k_idx, l_idx] = meshgrid(0:N, 0:N); 
k_idx = k_idx'; l_idx = l_idx';

mask_M = (k_idx + l_idx) <= N;
mask_parity = mod(k_idx + l_idx, 2) == 0;

norm = zeros(N+1, N+1);
is_k0 = (k_idx == 0);
is_l0 = (l_idx == 0);
norm(is_k0 & is_l0) = 1;
norm(xor(is_k0, is_l0)) = 2;
norm(~is_k0 & ~is_l0) = 4;

% apply masks: only where both mask_M and parity even keep norm, else zero
norm = norm .* mask_M .* mask_parity;

% Finally construct E matrix (size (N+1 x N+1))
E = (norm .* S_raw) / (num_points^2);   % equals norm * (1/num_points^2) * S_raw

% T_x: (num_points x (N+1)), E: ((N+1) x (N+1)), T_y': ((N+1) x num_points)
approximation = T_x * E * (T_y');       % result (num_points x num_points)

error_surface = abs(gaussian_kernel - approximation);

figure;
subplot(1,2,1);
surf(X, Y, gaussian_kernel, 'EdgeColor', 'none'); hold on;
xlabel('x'); ylabel('y'); 
title('Gaussian Kernel');colorbar;view(2); grid on;

subplot(1,2,2);
surf(X, Y, error_surface, 'EdgeColor', 'none');
xlabel('x'); ylabel('y'); 
title('Error Surface'); colorbar;view(2); grid on;
% figure('Units','centimeters','Position',[5 5 16 6]);   
% 
% ax1 = axes('Units','centimeters','Position',[1 1 6 5]); % [left bottom width height]
% surf(X, Y, gaussian_kernel, 'EdgeColor','none');
% view(2); axis tight equal;          
% xlabel('x'); ylabel('y');
% title('Gaussian Kernel');
% colormap(ax1, parula); colorbar;
% 
% ax2 = axes('Units','centimeters','Position',[8 1 6 5]); % 7.2≈1+6+0.2 间隙仅0.2 cm
% surf(X, Y, error_surface, 'EdgeColor','none');
% view(2); axis tight equal;
% xlabel('x'); ylabel('y');
% title('Error Surface');
% colormap(ax2, parula); colorbar;
% 
% linkaxes([ax1 ax2],'xy');

function T = chebyshev_values(x_vec, N)
    x_vec = x_vec(:);                  % ensure column
    m = length(x_vec);
    T = zeros(m, N+1);
    T(:,1) = 1;                         % T_0
    if N >= 1
        T(:,2) = x_vec;                 % T_1
    end
    for n = 2:N
        T(:, n+1) = 2 .* (x_vec .* T(:, n)) - T(:, n-1);
    end
end
