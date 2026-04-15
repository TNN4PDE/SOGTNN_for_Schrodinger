%%  load
sData = load('s_wbt_rc20_1e-12.mat');  
wData = load('w_wbt_rc20_1e-12.mat');
s = sData.s_wbt;   
w = wData.w_wbt;
%% Projection
% ================= 配置参数 =================
N_list = [1124,1124,984,824,768,656,518,472, 448, 400, 348, 308, 256, 232,192, 164, 128, 108, 96, 78, 68, 60, 52,44,36,30,24,32];
r_c = 20;  
% 复化积分设置
num_intervals = 64;  % 将积分区间划分为多少段 (复化)
points_per_interval = 512; % 每段上的高斯点数
% 总积分点数 = num_intervals * points_per_interval

% 测试网格设置
num_points_test = 256;
x_test = linspace(-r_c, r_c, num_points_test);
y_test = linspace(-r_c, r_c, num_points_test);
[X_test, Y_test] = meshgrid(x_test, y_test);
rho = 1./ sqrt(1+(X_test-Y_test).^2);
% 开启并行池 (如果尚未开启)
if isempty(gcp('nocreate'))
    parpool;
end

E_kl = zeros(N_list(1)+1,N_list(1)+1);
error = zeros(num_points_test,num_points_test);
% ================= 主循环 =================
for index = 1:1
    N = N_list(index);
    M = N;
    sl = s(index);
    wl = w(index);
    fprintf('Processing wl %d, wl = %.12e...\n', index, wl);
    fprintf('Processing sl %d, sl = %.12e...\n', index, sl);

    % --- 1. 准备复化积分网格 (并行化准备) ---
    % 使用变量代换 x = sin(theta) 消除权函数奇异性
    % theta 区间 [-pi/2, pi/2]
    theta_edges = linspace(-pi/2, pi/2, num_intervals + 1);

    % 获取标准区间 [-1, 1] 上的高斯-勒让德节点和权重
    [gl_nodes, gl_weights] = get_gauss_legendre(points_per_interval);

    % 初始化系数矩阵 (Accumulator)
    E_matrix = zeros(N+1, N+1);

    % --- 2. 并行计算系数 (Parallelized Composite Integration) ---
    % 将二维积分区域划分为 num_intervals * num_intervals 个块
    % 我们将这些块展平以便 parfor 并行
    total_blocks = num_intervals^2;

    % 预计算切比雪夫系数的缩放因子矩阵
    % k=0 或 l=0 时系数为 2/pi^2，否则为 4/pi^2，(0,0)为 1/pi^2
    scale_factors = ones(N+1, N+1) * (4 / pi^2);
    scale_factors(1, :) = 2 / pi^2;
    scale_factors(:, 1) = 2 / pi^2;
    scale_factors(1, 1) = 1 / pi^2;

    % 使用 parfor 对积分区域块进行并行累加
    parfor block_idx = 1:total_blocks
        % 将线性索引转换为网格索引 (row_idx, col_idx)
        [row_idx, col_idx] = ind2sub([num_intervals, num_intervals], block_idx);

        % === A. 映射坐标与权重 ===
        % 当前块的 Theta 范围
        t1_start = theta_edges(col_idx); t1_end = theta_edges(col_idx+1);
        t2_start = theta_edges(row_idx); t2_end = theta_edges(row_idx+1);

        % 将标准GL节点映射到当前 theta 子区间
        % 变换公式: t = (b-a)/2 * xi + (a+b)/2
        theta_x = (t1_end - t1_start)/2 * gl_nodes + (t1_end + t1_start)/2;
        w_x = (t1_end - t1_start)/2 * gl_weights; % 包含雅可比行列式 (dt/dxi)

        theta_y = (t2_end - t2_start)/2 * gl_nodes + (t2_end + t2_start)/2;
        w_y = (t2_end - t2_start)/2 * gl_weights;

        % 变换回 x, y 空间: x = sin(theta)
        x_local = sin(theta_x);
        y_local = sin(theta_y);

        % 生成局部网格
        [XX_local, YY_local] = meshgrid(x_local, y_local);
        [WX, WY] = meshgrid(w_x, w_y);
        W_local = WX .* WY; % 组合权重 dtheta_x * dtheta_y

        % === B. 计算被积函数的核心部分 ===
        % 注意：由于变量代换 x=sin(theta)，dx = cos(theta) dtheta
        % 原积分含权 1/sqrt(1-x^2) = 1/cos(theta)
        % 两者抵消： (1/cos) * cos = 1
        % 因此我们只需要积分: K(sin(tx), sin(ty)) * Tn(sin(tx)) * Tm(sin(ty)) * dtx * dty

        % 计算高斯核
%         kernel_vals = wl * exp(-(XX_local - YY_local).^2 * sl * r_c^2 );
        kernel_vals = 1 ./ sqrt((1/r_c)^2+(XX_local-YY_local).^2);

        % 计算切比雪夫多项式矩阵 T_mat (Size: num_points x (N+1))
        % 使用 vectorize 版本的函数计算
        Tx_mat = compute_cheb_matrix(x_local, N);
        Ty_mat = compute_cheb_matrix(y_local, N);

        % === C. 向量化投影 (核心优化) ===
        % 局部系数贡献 = Ty' * (Weights .* Kernel) * Tx
        % 这是一个矩阵乘法操作，替代了所有内层循环
        E_partial = Ty_mat' * (kernel_vals .* W_local) * Tx_mat;

        % 累加到总系数矩阵
        E_matrix = E_matrix + E_partial;
    end

    % 应用缩放因子
    E_final = E_matrix .* scale_factors;
    E_final = (E_final + E_final') ./ 2;
    % 应用 k+l <= M 约束 和 奇偶性约束 (Masking)
    [K_grid, L_grid] = meshgrid(0:N, 0:N); % 注意: meshgrid(col, row) -> x对应列(l), y对应行(k)
    % 这里 E_final 的行对应 y(l)，列对应 x(k) 还是反过来？
    % 上面代码: Ty' * ... * Tx. Ty对应行(y方向), Tx对应列(x方向)。
    % 所以 E_final(l+1, k+1) 对应 l (row index), k (col index)
    % 但通常习惯 k 对应 x, l 对应 y。为了匹配原代码逻辑，我们转置一下或者注意索引
    % 原代码逻辑: E_kl 是标量。
    % 这里 E_final(row, col) -> row是l的阶数, col是k的阶数

    mask = (K_grid + L_grid <= M) & (mod(K_grid + L_grid, 2) == 0);
    E_final = E_final .* mask'; % 转置以匹配 (k,l) or (l,k)
    % 实际上因为核是对称的，E是转置对称的，影响不大，但为了严谨：
    % 此时 E_final(i, j) 对应 l=(i-1), k=(j-1)。
    % 我们转置它让行对应k，列对应l，方便后续处理
    E_final = E_final';
    E_index = zeros(N_list(1)+1,N_list(1)+1);
    E_index(1:N_list(index)+1 , 1:N_list(index)+1) = E_final;
    E_kl = E_kl + E_index;
    % 计算测试点上的切比雪夫矩阵
    Tx_test = compute_cheb_matrix(x_test(:)./r_c, N); % Size: num_test x (N+1)
    Ty_test = compute_cheb_matrix(y_test(:)./r_c, N);

    % 矩阵乘法重建: Approx = Tx * E * Ty'
    % 维度分析: (Nx * N+1) * (N+1 x N+1) * (N+1 x Ny) = (Nx * Ny)
    approximation = Tx_test * E_final * Ty_test';

    % 真实值
%     gaussian_kernel_exact = wl * exp(-(X_test - Y_test).^2 * sl);
    gaussian_kernel_exact = 1 ./ sqrt((1/r_c)^2+(X_test - Y_test).^2);

    % 误差
    error_surface = abs(gaussian_kernel_exact - approximation);
    max_err = max(error_surface, [], 'all');
    fprintf('Max Error: %e\n', max_err);
    rel_error_surface = abs(gaussian_kernel_exact - approximation)./gaussian_kernel_exact;
    rel_max_err = max(rel_error_surface, [], 'all');
    fprintf('Rel Error: %e\n', rel_max_err);
    % --- 4. 绘图 ---
    plot_results(X_test, Y_test, gaussian_kernel_exact, approximation, error_surface);
end


% ================= 辅助函数 =================

function T_mat = compute_cheb_matrix(x, N)
    % 向量化计算切比雪夫多项式
    % 输入 x: 向量 (m x 1)
    % 输出 T_mat: 矩阵 (m x N+1)，第 col 列是 T_{col-1}(x)
    
    x = x(:); % 确保列向量
    m = length(x);
    T_mat = zeros(m, N + 1);
    T_mat(:, 1) = 1;          % T0
    if N >= 1
        T_mat(:, 2) = x;      % T1
    end
    for k = 2:N
        % 递归公式: T_n = 2xT_{n-1} - T_{n-2}
        T_mat(:, k+1) = 2 * x .* T_mat(:, k) - T_mat(:, k-1);
    end
end

function [x, w] = get_gauss_legendre(n)
    % 生成 n 个高斯-勒让德节点和权重 (区间 [-1, 1])
    % 使用 Golub-Welsch 算法
    
    beta = .5 ./ sqrt(1-(2*(1:n-1)).^(-2));
    T = diag(beta,1) + diag(beta,-1);
    [V, D] = eig(T);
    x = diag(D); 
    [x, idx] = sort(x); % 排序节点
    w = 2 * V(1,idx).^2; % 权重
    w = w(:); % 列向量
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
