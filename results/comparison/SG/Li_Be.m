%% Convergence Comparison: SlaterTNN vs SG (Li & Be Combined)
clear; clc; close all;

%% 1. Data Definition
% --- Li Atom (SGCI) ---
E_Li_SG = [-4.1455661856708934, -4.1929343397681018, -4.2076114218418219, -4.2100928548632108, -4.2104905140932960, -4.2105282482924080, -4.2105310672895673, -4.2105314854963032,-4.2105315828834016,-4.2105316110462319];
Dofs_Li_SG = [1561, 3353, 6704, 12540, 21747, 33574, 41836, 48168,53464,57912];

% --- Li Atom (SlaterTNN) ---
Dofs_Li_Slater = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40];
E_Li_Slater    = [ -4.195551084800674, -4.207140941420632,-4.21028395157562, -4.210526524350029, -4.2105311503810565, -4.210531548706509, -4.210531609586058, -4.210531622959367, -4.210531633905815, -4.210531636859215,  -4.210531641596987, -4.210531645922184];

% --- Be Atom (SGCI) ---
E_Be_SG = [-6.1575102733, -6.4148551171, -6.5840238139, -6.6978644058, -6.7393385832, ...
     -6.7574806678, -6.7666240817, -6.7732543576, -6.7775300791, -6.7802612569, -6.7815434763];
Dofs_Be_SG = [1209, 3198, 7424, 15463, 22611, 28333, 33522, 38970, 45194, 51414, 56380];

% --- Be Atom (SlaterTNN) ---
Dofs_Be_Slater = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 60];
E_Be_Slater    = [-6.739449617040396, -6.77328786473805, -6.782456563852316, -6.784887978868687, -6.785045279321682, -6.785070769478306, -6.785074823161318, -6.785076262696509, -6.785077053689453, -6.785077506093637, -6.785077642129568, -6.78507776168674, -6.785077911491615, -6.785077940670421, -6.785077942563639];

%% 2. Extract Benchmarks (Extrapolated from SlaterTNN)
options = optimset('TolX', 1e-15, 'Display', 'off');
lb_delta = -25; ub_delta = -2;


fit_Li = @(delta) get_r2_mixed(delta, Dofs_Li_Slater(:), E_Li_Slater(:));
best_delta_Li = fminbnd(fit_Li, lb_delta, ub_delta, options);
bench_Li = E_Li_Slater(end) - 10^(best_delta_Li);


fit_Be = @(delta) get_r2_mixed(delta, Dofs_Be_Slater(:), E_Be_Slater(:));
best_delta_Be = fminbnd(fit_Be, lb_delta, ub_delta, options);
bench_Be = E_Be_Slater(end) - 10^(best_delta_Be);

%% 3. Calculate Relative Tolerances
err_Li_Slater = abs(E_Li_Slater - bench_Li) / abs(bench_Li);
err_Li_SG     = abs(E_Li_SG - bench_Li) / abs(bench_Li);

err_Be_Slater = abs(E_Be_Slater - bench_Be) / abs(bench_Be);
err_Be_SG     = abs(E_Be_SG - bench_Be) / abs(bench_Be);

%% 4. Visualization Setup
figure('Color', 'w', 'Position', [200, 150, 650, 500]);
hold on;


color_Li = '#EDB120';
color_Be = '#7E2F8E';

%% --- Plot Data ---


loglog(Dofs_Li_Slater, err_Li_Slater, 'o-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_Li, 'Color', color_Li, 'DisplayName', 'Li (SlaterTNN)');


loglog(Dofs_Li_SG, err_Li_SG, 'o--', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_Li, 'Color', color_Li, 'DisplayName', 'Li (SGCI)');



loglog(Dofs_Be_Slater, err_Be_Slater, '^-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_Be, 'Color', color_Be, 'DisplayName', 'Be (SlaterTNN)');


loglog(Dofs_Be_SG, err_Be_SG, '^--', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_Be, 'Color', color_Be, 'DisplayName', 'Be (SGCI)');

%% 5. Formatting
title('\textbf{Convergence: Li and Be Atoms}', 'Interpreter', 'latex', 'FontSize', 15);
xlabel('\textbf{Degrees of Freedom (Dofs)}', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('\textbf{Relative Tolerance} $\boldsymbol{\epsilon_{rel}}$', 'Interpreter', 'latex', 'FontSize', 14);


legend('Location', 'southwest', 'FontSize', 11);


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