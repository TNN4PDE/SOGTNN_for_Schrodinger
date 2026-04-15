% MATLAB代码：SOG展开和误差图像
clear; clc; close all;
format long;
% 参数设置
% b = 1.22749083347315613; %(1e-10)
% b = 1.288632687697091; %(1e-8)
% b = 1.394021712971884; %(1e-6)
% b = 1.187800189600461; %(1e-12)
% b = 1.157860260063399; %(1e-14)
b = 1.618405221009676;
sigma = 1 ; % 高斯宽度
r_c = 13;
rmin =sqrt(1);
rmax =sqrt(1+4*r_c^2);
r = logspace(log10(rmin), log10(rmax), 100000); % r 的取值范围，避免 r=0

% 计算近似值
approx = zeros(size(r));
for ell = -15:220 % 选择 ell 的范围
    % 计算高斯项
    gaussian_term = (1 / b^ell) .* exp(-0.5 * (r / (b^ell * sigma)).^2);
    approx = approx + gaussian_term; % 累加高斯项
end

% 正则化常数
normalization_constant = (2 * log(b)) / sqrt(2 * pi * sigma^2);
approx = normalization_constant * approx;

% 真实值
exact = 1 ./ r;

% 计算误差
error = (approx - exact)./exact;
% 绘图
figure;
% subplot(2, 1, 1);
% plot(r, exact, 'b-', 'LineWidth', 2); hold on;
% plot(r, approx, 'r--', 'LineWidth', 2);
% xlabel('Distance (r)');
% ylabel('Potential');
% legend('Real Potential (1/r)', 'Ewald Potential');
% title('Ewald Expansion of 1/r');
% grid on;
% 
% subplot(2, 1, 2);
plot(r, error, 'k-', 'LineWidth', 2);
xlabel('r');
ylabel('Error');
title('Error between Coulomb kernel and SOG decomposition');
grid on;
