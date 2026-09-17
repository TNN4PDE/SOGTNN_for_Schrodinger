%% Ultimate Convergence Comparison: SlaterTNN vs SGCI (2x2 Panels)
% Panel 1: H & He  |  Panel 2: Li & Be
% Panel 3: B & C   |  Panel 4: N & O
clear; clc; close all;

%% ========================================================================
% 1. Data Definition (All 8 Atoms)
% ========================================================================
% --- 1. H Atom ---
Dofs_H_Slater = [1, 4, 8, 12]; 
E_H_Slater  = [-0.6697771379861136, -0.6697771381397745, -0.6697771381870984, -0.6697771382065866];
bench_H = -0.6697771382138; 
Dofs_H_SG = [5, 9, 17, 25, 33, 41, 49];
E_H_SG = [-0.6407167120137900, -0.6498845317224443,-0.6696857678761953,-0.6697653012175548,-0.6697771021273720,-0.6697771317452265,-0.6697771381899646];

% --- 2. He Atom ---
Dofs_He_Slater = [1, 2, 4, 8, 12, 16, 20];
E_He_Slater = [-2.2242095525123955, -2.236487893369299, -2.238214088352951, -2.2382575859531224, -2.2382577957675394, -2.2382578211908895, -2.2382578223930367];
bench_He = -2.23825782410; 
Dofs_He_SG = [49, 113, 265, 605, 1185, 2005, 3305, 5177];
E_He_SG = [-2.1722879685459398, -2.2218291367215839, -2.2348952918863554, -2.2377789303937510, -2.2382226966813152, -2.2382559230385080, -2.2382577749953572, -2.2382578234027513];

% --- 3. Li Atom ---
Dofs_Li_Slater = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40];
E_Li_Slater    = [ -4.195551084800674, -4.207140941420632,-4.21028395157562, -4.210526524350029, -4.2105311503810565, -4.210531548706509, -4.210531609586058, -4.210531622959367, -4.210531633905815, -4.210531636859215,  -4.210531641596987, -4.210531645922184];
Dofs_Li_SG = [1561, 3353, 6704, 12540, 21747, 33574, 41836, 48168,53464,57912];
E_Li_SG = [-4.1455661856708934, -4.1929343397681018, -4.2076114218418219, -4.2100928548632108, -4.2104905140932960, -4.2105282482924080, -4.2105310672895673, -4.2105314854963032,-4.2105315828834016,-4.2105316110462319];

% --- 4. Be Atom ---
Dofs_Be_Slater = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 60];
E_Be_Slater    = [-6.739449617040396, -6.77328786473805, -6.782456563852316, -6.784887978868687, -6.785045279321682, -6.785070769478306, -6.785074823161318, -6.785076262696509, -6.785077053689453, -6.785077506093637, -6.785077642129568, -6.78507776168674, -6.785077911491615, -6.785077940670421, -6.785077942563639];
Dofs_Be_SG = [1209, 3198, 7424, 15463, 22611, 28333, 33522, 38970, 45194, 51414, 56380];
E_Be_SG = [-6.1575102733, -6.4148551171, -6.5840238139, -6.6978644058, -6.7393385832, -6.7574806678, -6.7666240817, -6.7732543576, -6.7775300791, -6.7802612569, -6.7815434763];

% --- 5. B Atom ---
Dofs_B_Slater = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64];
E_B_Slater    = [-9.783947908783265,  -9.805848815941845, -9.816788722963741, -9.819280136379456, -9.819603485224862, -9.819663642770207, -9.819700680497478, -9.819707388520092, -9.819710467266532, -9.819711578807377, -9.81971196743288, -9.819712286746151];
Dofs_B_SG = [465, 1484, 3902, 8912, 17593, 24447, 30150, 34270, 38832, 42002, 45458];
E_B_SG    = [-7.99521093, -8.55377778, -8.96533794, -9.27456372, -9.51030807, -9.60489820, -9.64985367, -9.67365858, -9.69093995, -9.70064626, -9.70994276];

% --- 6. C Atom ---
Dofs_C_Slater = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80];
E_C_Slater    = [-13.284878571886463, -13.301612297833806, -13.319443053146268, -13.328881475141872,  -13.330597681465173 ,  -13.331123431575906, -13.331342350075479, -13.331426854122506, -13.331501387945936, -13.331521835053339, -13.331531914589261, -13.331535811485601, -13.33153932289391, -13.331541019272963, -13.331542394279513];
Dofs_C_SG = [355, 1392, 4430, 12140, 20913, 28751, 35535];
E_C_SG    = [-10.26279833, -11.02976477, -11.64952223, -12.17510617, -12.48447905, -12.69397924, -12.96004599];

% --- 7. N Atom ---
Dofs_N_Slater = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 72, 80, 88];
E_N_Slater    = [-17.228411185221063, -17.263883614699566, -17.279856378258284, -17.2887542036768, -17.290524569867813, -17.291044345378687, -17.291349257057135, -17.29143126075642, -17.29155112206973, -17.29159523922079, -17.29161164594694, -17.291626263873585, -17.291632590182065, -17.291638453031435, -17.291641432341283, -17.29164413981273];
Dofs_N_SG = [63, 408, 1741, 3831, 7895, 14496, 20587, 25846];
E_N_SG    = [-11.97623127, -13.21446513, -14.07484211, -14.65130709, -15.13690796, -15.47978324, -15.72175956, -15.87873531];

% --- 8. O Atom ---
Dofs_O_Slater = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 60, 72, 84];
E_O_Slater    = [-21.651305017372596, -21.67550346288669, -21.68795557329471, -21.695078608509203, -21.69682328740639,  -21.69768768247736,  -21.69806816670705, -21.69865715812662,  -21.699159350273742, -21.6993375751852,  -21.699422092528295, -21.699498117781904, -21.69953286629327, -21.69955377175065];
Dofs_O_SG = [120, 645, 2527, 5189, 8058, 11516, 15094, 16742, 21642, 24866];
E_O_SG    = [-15.66078963, -16.62891085, -17.70240748, -18.19877819, -18.54340767, -18.86264007, -19.15830339, -19.39531557, -19.90845491, -21.33644962];

%% ========================================================================
% 2. Extract Benchmarks via fminbnd (Extrapolated from SlaterTNN)
% ========================================================================
options = optimset('TolX', 1e-15, 'Display', 'off');
lb_delta = -25; ub_delta = -2;

% Li to O require extrapolation
bench_Li = E_Li_Slater(end) - 10^(fminbnd(@(d) get_r2_mixed(d, Dofs_Li_Slater(:), E_Li_Slater(:)), lb_delta, ub_delta, options));
bench_Be = E_Be_Slater(end) - 10^(fminbnd(@(d) get_r2_mixed(d, Dofs_Be_Slater(:), E_Be_Slater(:)), lb_delta, ub_delta, options));
bench_B  = E_B_Slater(end)  - 10^(fminbnd(@(d) get_r2_mixed(d, Dofs_B_Slater(:), E_B_Slater(:)), lb_delta, ub_delta, options));
bench_C  = E_C_Slater(end)  - 10^(fminbnd(@(d) get_r2_mixed(d, Dofs_C_Slater(:), E_C_Slater(:)), lb_delta, ub_delta, options));
bench_N  = E_N_Slater(end)  - 10^(fminbnd(@(d) get_r2_mixed(d, Dofs_N_Slater(:), E_N_Slater(:)), lb_delta, ub_delta, options));
bench_O  = E_O_Slater(end)  - 10^(fminbnd(@(d) get_r2_mixed(d, Dofs_O_Slater(:), E_O_Slater(:)), lb_delta, ub_delta, options));

%% ========================================================================
% 3. Calculate Relative Tolerances
% ========================================================================
err_H_Slater  = abs(E_H_Slater - bench_H) / abs(bench_H);   err_H_SG  = abs(E_H_SG - bench_H) / abs(bench_H);
err_He_Slater = abs(E_He_Slater - bench_He) / abs(bench_He); err_He_SG = abs(E_He_SG - bench_He) / abs(bench_He);
err_Li_Slater = abs(E_Li_Slater - bench_Li) / abs(bench_Li); err_Li_SG = abs(E_Li_SG - bench_Li) / abs(bench_Li);
err_Be_Slater = abs(E_Be_Slater - bench_Be) / abs(bench_Be); err_Be_SG = abs(E_Be_SG - bench_Be) / abs(bench_Be);
err_B_Slater  = abs(E_B_Slater - bench_B) / abs(bench_B);    err_B_SG  = abs(E_B_SG - bench_B) / abs(bench_B);
err_C_Slater  = abs(E_C_Slater - bench_C) / abs(bench_C);    err_C_SG  = abs(E_C_SG - bench_C) / abs(bench_C);
err_N_Slater  = abs(E_N_Slater - bench_N) / abs(bench_N);    err_N_SG  = abs(E_N_SG - bench_N) / abs(bench_N);
err_O_Slater  = abs(E_O_Slater - bench_O) / abs(bench_O);    err_O_SG  = abs(E_O_SG - bench_O) / abs(bench_O);

%% ========================================================================
% 4. Master Visualization Setup
% ========================================================================
figure('Color', 'w', 'Position', [100, 50, 1100, 800]);


color_H  = '#0072BD';
color_He = '#D95319';
color_Li = '#EDB120';
color_Be = '#7E2F8E';
color_B  = '#77AC30';
color_C  = '#4DBEEE';
color_N  = '#A2142F';
color_O  = '#000000';

%% ---------------------------------------------------------
% Panel 1: H & He (Top Left)
% ---------------------------------------------------------
subplot(2, 2, 1); hold on;
loglog(Dofs_H_Slater, err_H_Slater, 'o-', 'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', color_H, 'Color', color_H, 'DisplayName', 'H (SlaterTNN)');
loglog(Dofs_H_SG, err_H_SG, 'o--', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', color_H, 'Color', color_H, 'DisplayName', 'H (SGCI)');
loglog(Dofs_He_Slater, err_He_Slater, '^-', 'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', color_He, 'Color', color_He, 'DisplayName', 'He (SlaterTNN)');
loglog(Dofs_He_SG, err_He_SG, '^--', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', color_He, 'Color', color_He, 'DisplayName', 'He (SGCI)');

title('\textbf{H and He Atoms}', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('\textbf{Relative Tolerance} $\boldsymbol{\epsilon_{rel}}$', 'Interpreter', 'latex', 'FontSize', 13);
legend('Location', 'southwest', 'FontSize', 10);
xlim([1, 10000]); ylim([1e-12, 1e-1]);
set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on', 'XMinorTick', 'on', 'YMinorTick', 'on');
grid on; grid minor; hold off;

%% ---------------------------------------------------------
% Panel 2: Li & Be (Top Right)
% ---------------------------------------------------------
subplot(2, 2, 2); hold on;
loglog(Dofs_Li_Slater, err_Li_Slater, 'o-', 'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', color_Li, 'Color', color_Li, 'DisplayName', 'Li (SlaterTNN)');
loglog(Dofs_Li_SG, err_Li_SG, 'o--', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', color_Li, 'Color', color_Li, 'DisplayName', 'Li (SGCI)');
loglog(Dofs_Be_Slater, err_Be_Slater, '^-', 'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', color_Be, 'Color', color_Be, 'DisplayName', 'Be (SlaterTNN)');
loglog(Dofs_Be_SG, err_Be_SG, '^--', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', color_Be, 'Color', color_Be, 'DisplayName', 'Be (SGCI)');

title('\textbf{Li and Be Atoms}', 'Interpreter', 'latex', 'FontSize', 14);
legend('Location', 'southwest', 'FontSize', 10);
xlim([1, 100000]); ylim([1e-12, 1e-1]);
set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on', 'XMinorTick', 'on', 'YMinorTick', 'on');
grid on; grid minor; hold off;

%% ---------------------------------------------------------
% Panel 3: B & C (Bottom Left)
% ---------------------------------------------------------
subplot(2, 2, 3); hold on;
loglog(Dofs_B_Slater, err_B_Slater, 'o-', 'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', color_B, 'Color', color_B, 'DisplayName', 'B (SlaterTNN)');
loglog(Dofs_B_SG, err_B_SG, 'o--', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', color_B, 'Color', color_B, 'DisplayName', 'B (SGCI)');
loglog(Dofs_C_Slater, err_C_Slater, '^-', 'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', color_C, 'Color', color_C, 'DisplayName', 'C (SlaterTNN)');
loglog(Dofs_C_SG, err_C_SG, '^--', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', color_C, 'Color', color_C, 'DisplayName', 'C (SGCI)');

title('\textbf{B and C Atoms}', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('\textbf{Degrees of Freedom (DoFs)}', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('\textbf{Relative Tolerance} $\boldsymbol{\epsilon_{rel}}$', 'Interpreter', 'latex', 'FontSize', 13);
legend('Location', 'southwest', 'FontSize', 10);
xlim([1, 100000]); ylim([1e-12, 1e-1]);
set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on', 'XMinorTick', 'on', 'YMinorTick', 'on');
grid on; grid minor; hold off;

%% ---------------------------------------------------------
% Panel 4: N & O (Bottom Right)
% ---------------------------------------------------------
subplot(2, 2, 4); hold on;
loglog(Dofs_N_Slater, err_N_Slater, 'o-', 'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', color_N, 'Color', color_N, 'DisplayName', 'N (SlaterTNN)');
loglog(Dofs_N_SG, err_N_SG, 'o--', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', color_N, 'Color', color_N, 'DisplayName', 'N (SGCI)');
loglog(Dofs_O_Slater, err_O_Slater, '^-', 'LineWidth', 2.5, 'MarkerSize', 7, 'MarkerFaceColor', color_O, 'Color', color_O, 'DisplayName', 'O (SlaterTNN)');
loglog(Dofs_O_SG, err_O_SG, '^--', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', color_O, 'Color', color_O, 'DisplayName', 'O (SGCI)');

title('\textbf{N and O Atoms}', 'Interpreter', 'latex', 'FontSize', 14);
xlabel('\textbf{Degrees of Freedom (DoFs)}', 'Interpreter', 'latex', 'FontSize', 13);
legend('Location', 'southwest', 'FontSize', 10);
xlim([1, 100000]); ylim([1e-12, 1e-1]);
set(gca, 'FontSize', 11, 'LineWidth', 1.2, 'Box', 'on', 'XMinorTick', 'on', 'YMinorTick', 'on');
grid on; grid minor; hold off;

%% ========================================================================
% 5. Global Figure Enhancements
% ========================================================================

sgtitle('\textbf{Ultimate Convergence Comparison: SlaterTNN vs SGCI}', ...
    'Interpreter', 'latex', 'FontSize', 18, 'FontWeight', 'bold');


set(gcf, 'Units', 'normalized', 'Position', [0.1, 0.1, 0.85, 0.8]);

%% ========================================================================
% 6. Optional: Export High-Resolution Figure
% ========================================================================
% exportgraphics(gcf, 'Convergence_Comparison_SlaterTNN_vs_SGCI.png', ...
%     'Resolution', 300, 'BackgroundColor', 'white');

%% ========================================================================
% 7. Helper Function: Mixed Convergence Rate Estimator
% ========================================================================
function r2 = get_r2_mixed(delta, dofs, energies)





    dofs = dofs(:);
    energies = energies(:);
    

    N = min(5, length(dofs));
    x = log(dofs(end-N+1:end));
    y = log(abs(energies(end-N+1:end) - (energies(end) + 10^delta)));
    

    if length(x) < 2 || any(isnan(y)) || any(isinf(y))
        r2 = -inf;
        return;
    end
    
    p = polyfit(x, y, 1);
    y_fit = polyval(p, x);
    

    SS_res = sum((y - y_fit).^2);
    SS_tot = sum((y - mean(y)).^2);
    
    if SS_tot < eps
        r2 = -inf;
    else
        r2 = 1 - SS_res / SS_tot;
    end
    

    r2 = -r2;
end

% %% ========================================================================
% % 8. Optional: Print Summary Table
% % ========================================================================
% fprintf('\n========== Convergence Summary ==========\n');
% fprintf('%-4s %-12s %-12s %-12s %-12s\n', 'Atom', 'SlaterTNN(DoF)', 'SlaterTNN(Err)', 'SGCI(DoF)', 'SGCI(Err)');
% fprintf('%-4s %-12s %-12s %-12s %-12s\n', '----', '------------', '------------', '---------', '---------');
% 
% atoms = {'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O'};
% data = {Dofs_H_Slater, err_H_Slater, Dofs_H_SG, err_H_SG;
%         Dofs_He_Slater, err_He_Slater, Dofs_He_SG, err_He_SG;
%         Dofs_Li_Slater, err_Li_Slater, Dofs_Li_SG, err_Li_SG;
%         Dofs_Be_Slater, err_Be_Slater, Dofs_Be_SG, err_Be_SG;
%         Dofs_B_Slater, err_B_Slater, Dofs_B_SG, err_B_SG;
%         Dofs_C_Slater, err_C_Slater, Dofs_C_SG, err_C_SG;
%         Dofs_N_Slater, err_N_Slater, Dofs_N_SG, err_N_SG;
%         Dofs_O_Slater, err_O_Slater, Dofs_O_SG, err_O_SG};
% 
% for i = 1:8

%     s_dof = data{i,1}(end);
%     s_err = data{i,2}(end);

%     g_dof = data{i,3}(end);
%     g_err = data{i,4}(end);
%     
%     fprintf('%-4s %-12.0f %-12.2e %-12.0f %-12.2e\n', ...
%         atoms{i}, s_dof, s_err, g_dof, g_err);
% end
% fprintf('=========================================\n');