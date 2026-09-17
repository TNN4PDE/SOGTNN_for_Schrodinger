%% Convergence Comparison: SlaterTNN vs SG (B to O Combined)
clear; clc; close all;

%% 1. Data Definition
% ==============================================================
% 1. B Atom
% ==============================================================
Dofs_B_SG = [465, 1484, 3902, 8912, 17593, 24447, 30150, 34270, 38832, 42002, 45458];
E_B_SG    = [-7.99521093, -8.55377778, -8.96533794, -9.27456372, -9.51030807, -9.60489820, -9.64985367, -9.67365858, -9.69093995, -9.70064626, -9.70994276];

Dofs_B_Slater = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64];
E_B_Slater    = [-9.783947908783265,  -9.805848815941845, -9.816788722963741, -9.819280136379456, -9.819603485224862, -9.819663642770207, -9.819700680497478, -9.819707388520092, -9.819710467266532, -9.819711578807377, -9.81971196743288, -9.819712286746151];

% ==============================================================
% 2. C Atom
% ==============================================================
Dofs_C_SG = [355, 1392, 4430, 12140, 20913, 28751, 35535];
E_C_SG    = [-10.26279833, -11.02976477, -11.64952223, -12.17510617, -12.48447905, -12.69397924, -12.96004599];

Dofs_C_Slater = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80];
E_C_Slater    = [-13.284878571886463, -13.301612297833806, -13.319443053146268, -13.328881475141872,  -13.330597681465173 ,  -13.331123431575906, -13.331342350075479, -13.331426854122506, -13.331501387945936, -13.331521835053339, -13.331531914589261, -13.331535811485601, -13.33153932289391, -13.331541019272963, -13.331542394279513];

% ==============================================================
% 3. N Atom
% ==============================================================
Dofs_N_SG = [63, 408, 1741, 3831, 7895, 14496, 20587, 25846];
E_N_SG    = [-11.97623127, -13.21446513, -14.07484211, -14.65130709, -15.13690796, -15.47978324, -15.72175956, -15.87873531];

Dofs_N_Slater = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 88];
E_N_Slater    = [-17.228411185221063, -17.263883614699566, -17.279856378258284, -17.2887542036768, -17.290524569867813, -17.291044345378687, -17.291349257057135, -17.29143126075642, -17.29155112206973, -17.29159523922079, -17.29161164594694, -17.291626263873585, -17.291632590182065, -17.291638453031435, -17.291641432341283, -17.29164413978363];

% ==============================================================
% 4. O Atom
% ==============================================================
Dofs_O_SG = [120, 645, 2527, 5189, 8058, 11516, 15094, 16742, 21642, 24866];
E_O_SG    = [-15.66078963, -16.62891085, -17.70240748, -18.19877819, -18.54340767, -18.86264007, -19.15830339, -19.39531557, -19.90845491, -21.33644962];

Dofs_O_Slater = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 60, 72, 84];
E_O_Slater    = [-21.651305017372596, -21.67550346288669, -21.68795557329471, -21.695078608509203, -21.69682328740639,  -21.69768768247736,  -21.69806816670705, -21.69865715812662,  -21.699159350273742, -21.6993375751852,  -21.699422092528295, -21.699498117781904, -21.699531323690625, -21.69954723999314];

%% 2. Extract Benchmarks (Extrapolated from SlaterTNN)
options = optimset('TolX', 1e-15, 'Display', 'off');
lb_delta = -25; ub_delta = -2;


fit_B = @(delta) get_r2_mixed(delta, Dofs_B_Slater(:), E_B_Slater(:));
bench_B = E_B_Slater(end) - 10^(fminbnd(fit_B, lb_delta, ub_delta, options));


fit_C = @(delta) get_r2_mixed(delta, Dofs_C_Slater(:), E_C_Slater(:));
bench_C = E_C_Slater(end) - 10^(fminbnd(fit_C, lb_delta, ub_delta, options));


fit_N = @(delta) get_r2_mixed(delta, Dofs_N_Slater(:), E_N_Slater(:));
bench_N = E_N_Slater(end) - 10^(fminbnd(fit_N, lb_delta, ub_delta, options));


fit_O = @(delta) get_r2_mixed(delta, Dofs_O_Slater(:), E_O_Slater(:));
bench_O = E_O_Slater(end) - 10^(fminbnd(fit_O, lb_delta, ub_delta, options));

%% 3. Calculate Relative Tolerances
err_B_Slater = abs(E_B_Slater - bench_B) / abs(bench_B);
err_B_SG     = abs(E_B_SG - bench_B) / abs(bench_B);

err_C_Slater = abs(E_C_Slater - bench_C) / abs(bench_C);
err_C_SG     = abs(E_C_SG - bench_C) / abs(bench_C);

err_N_Slater = abs(E_N_Slater - bench_N) / abs(bench_N);
err_N_SG     = abs(E_N_SG - bench_N) / abs(bench_N);

err_O_Slater = abs(E_O_Slater - bench_O) / abs(bench_O);
err_O_SG     = abs(E_O_SG - bench_O) / abs(bench_O);

%% 4. Visualization Setup
figure('Color', 'w', 'Position', [150, 150, 750, 550]);
hold on;


color_B = '#77AC30';
color_C = '#4DBEEE';
color_N = '#A2142F';
color_O = '#000000';

%% --- Plot Data ---

loglog(Dofs_B_Slater, err_B_Slater, '<-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_B, 'Color', color_B, 'DisplayName', 'B (SlaterTNN)');
loglog(Dofs_B_SG, err_B_SG, '<--', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_B, 'Color', color_B, 'DisplayName', 'B (SGCI)');


loglog(Dofs_C_Slater, err_C_Slater, '>-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_C, 'Color', color_C, 'DisplayName', 'C (SlaterTNN)');
loglog(Dofs_C_SG, err_C_SG, '>--', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_C, 'Color', color_C, 'DisplayName', 'C (SGCI)');


loglog(Dofs_N_Slater, err_N_Slater, 'p-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_N, 'Color', color_N, 'DisplayName', 'N (SlaterTNN)');
loglog(Dofs_N_SG, err_N_SG, 'p--', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_N, 'Color', color_N, 'DisplayName', 'N (SGCI)');


loglog(Dofs_O_Slater, err_O_Slater, 'h-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_O, 'Color', color_O, 'DisplayName', 'O (SlaterTNN)');
loglog(Dofs_O_SG, err_O_SG, 'h--', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_O, 'Color', color_O, 'DisplayName', 'O (SGCI)');

%% 5. Formatting
title('\textbf{Convergence: B, C, N, and O Atoms}', 'Interpreter', 'latex', 'FontSize', 15);
xlabel('\textbf{Degrees of Freedom (Dofs)}', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('\textbf{Relative Tolerance} $\boldsymbol{\epsilon_{rel}}$', 'Interpreter', 'latex', 'FontSize', 14);


legend('Location', 'southwest', 'FontSize', 11, 'NumColumns', 2);


xlim([1, 100000]); 
ylim([1e-12, 1e-1]);
set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'Box', 'on', 'XMinorTick', 'on', 'YMinorTick', 'on');
grid on; grid minor;
hold off;

%% Helper Function (For Benchmark Extraction)
function neg_r2 = get_r2_mixed(delta, N, E_raw)
    E_inf = E_raw(end) - 10^(delta); 
    err = E_raw - E_inf;
    if any(err <= 0)
        neg_r2 = 1e6; 
        return; 
    end 
    y = log10(err);
    X = [ones(size(N)), log10(N), N];
    b = X \ y;
    y_pred = X * b;
    r2 = 1 - sum((y - y_pred).^2) / sum((y - mean(y)).^2);
    neg_r2 = -r2;
end