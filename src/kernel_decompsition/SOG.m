clear; clc; close all;
format long;

% b = 1.22749083347315613; %(1e-10)
% b = 1.288632687697091; %(1e-8)
% b = 1.394021712971884; %(1e-6)
% b = 1.187800189600461; %(1e-12)
% b = 1.157860260063399; %(1e-14)
b = 1.618405221009676;
sigma = 1 ; 
r_c = 13;
rmin =sqrt(1);
rmax =sqrt(1+4*r_c^2);
r = logspace(log10(rmin), log10(rmax), 100000); % r 的取值范围，避免 r=0


approx = zeros(size(r));
for ell = -15:220 
    gaussian_term = (1 / b^ell) .* exp(-0.5 * (r / (b^ell * sigma)).^2);
    approx = approx + gaussian_term; 
end

normalization_constant = (2 * log(b)) / sqrt(2 * pi * sigma^2);
approx = normalization_constant * approx;

exact = 1 ./ r;

error = (approx - exact)./exact;

figure;
plot(r, error, 'k-', 'LineWidth', 2);
xlabel('r');
ylabel('Error');
title('Error between Coulomb kernel and SOG decomposition');
grid on;
