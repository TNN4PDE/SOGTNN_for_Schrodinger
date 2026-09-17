clear; clc; close all;
format long;
% b = 1.22749083347315613; %(1e-10)
% b = 1.288632687697091; %(1e-8)
% b = 1.394021712971884; %(1e-6)
% b = 1.187800189600461; %(1e-12)
% b = 1.157860260063399; %(1e-14)
% b = 1.618405221009676;
b = 1.4;
sigma = 1 ;
rmin = 1e-16;
rmax = 2*sqrt(3);
% r_c = 13;
% rmin =sqrt(1);
% rmax =sqrt(1+4*r_c^2);
r = logspace(log10(rmin), log10(rmax), 100000); 
approx = zeros(size(r));
for ell = -1000:60 
    gaussian_term = (1 / b^ell) .* exp(-0.5 * (r / (b^ell * sigma)).^2);
    approx = approx + gaussian_term; 
end
normalization_constant = (2 * log(b)) / sqrt(2 * pi * sigma^2);
approx = normalization_constant * approx;
exact = 1 ./ r;
error = (approx - exact)./exact;
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
