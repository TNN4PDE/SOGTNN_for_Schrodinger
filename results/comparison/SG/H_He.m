%% Convergence Comparison: SlaterTNN vs SG (H & He Combined)
clear; clc; close all;

%% 1. Data Definition
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

%% 2. Calculate Relative Tolerances
err_H_Slater  = abs(E_H_Slater - bench_H) / abs(bench_H);
err_H_SG      = abs(E_H_SG - bench_H) / abs(bench_H);
err_He_Slater = abs(E_He_Slater - bench_He) / abs(bench_He);
err_He_SG     = abs(E_He_SG - bench_He) / abs(bench_He);

%% 3. Visualization Setup
figure('Color', 'w', 'Position', [200, 150, 650, 500]);
hold on;


color_H  = '#0072BD';
color_He = '#D95319';

% --- Plot Data ---


loglog(Dofs_H_Slater, err_H_Slater, 'o-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_H, 'Color', color_H, 'DisplayName', 'H (SlaterTNN)');


loglog(Dofs_H_SG, err_H_SG, 'o--', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_H, 'Color', color_H, 'DisplayName', 'H (SGCI)');



loglog(Dofs_He_Slater, err_He_Slater, '^-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_He, 'Color', color_He, 'DisplayName', 'He (SlaterTNN)');


loglog(Dofs_He_SG, err_He_SG, '^--', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', color_He, 'Color', color_He, 'DisplayName', 'He (SGCI)');

%% 4. Formatting
title('\textbf{Convergence: H and He Atoms}', 'Interpreter', 'latex', 'FontSize', 15);
xlabel('\textbf{Degrees of Freedom (Dofs)}', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('\textbf{Relative Tolerance} $\boldsymbol{\epsilon_{rel}}$', 'Interpreter', 'latex', 'FontSize', 14);


legend('Location', 'southwest', 'FontSize', 11);


xlim([1, 10000]); 
ylim([1e-12, 1e-1]);

set(gca, 'FontSize', 12, 'LineWidth', 1.2, 'Box', 'on', 'XMinorTick', 'on', 'YMinorTick', 'on');
hold off;