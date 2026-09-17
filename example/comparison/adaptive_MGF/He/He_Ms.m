%% JH_He_Table51_part3_coarsening.m
% Part 3 of the J.H.-style He Table 5.1 reproduction.
%
% Goal:
%   Apply J.H. coarsening step to the raw a-priori initial system
%       B_three^[0]
%   constructed in Part 2.
%
% J.H. definition:
%   Let B = {Phi_mu}_{mu=1}^M be an L2-normalized basis set and let
%       Psi(B) = sum_mu v_mu^B Phi_mu
%   be the lowest state in span(B).
%
%   B(nu) = { Phi_mu in B : |v_mu^B| >= 2^(-nu) }.
%
%   coarse_delta(B) is the smallest B(nu) such that
%       | E(B(nu)) - E(B) | <= delta.
%
% J.H. numerical parameter:
%   delta = 1/100 kcal/mol ~= 1.5936e-5 hartree.
%
% Input:
%   JH_He_Bthree0_raw_assembled_M550.mat
%
% Output:
%   JH_He_Bthree0_coarsened_Mxxx.mat

clear; clc; close all;
format long e;

%% ============================================================
% 0. Load Part 2 result
% =============================================================
files = dir('JH_He_Bthree0_raw_assembled_M*.mat');
if isempty(files)
    error('Cannot find JH_He_Bthree0_raw_assembled_M*.mat. Run Part 2 first.');
end
[~,ord] = max([files.datenum]);
inputFile = files(ord).name;

fprintf('\n============================================================\n');
fprintf('J.H. He Table 5.1 Part 3: coarsening B_three^[0]\n');
fprintf('Loading: %s\n', inputFile);
fprintf('============================================================\n\n');

load(inputFile, 'part2');

S_raw = part2.S2;
H_raw = part2.H2;
Hone_raw = part2.Hone;
Vee_raw = part2.Vee;
twoBasis_raw = part2.twoBasis;
coeff_raw = part2.coeff_raw;
E_raw = part2.E0_raw;

Mraw = numel(twoBasis_raw);

% J.H. delta = 1/100 kcal/mol.
hartree_per_kcalmol = 1/627.5094740631;
delta = 0.01 * hartree_per_kcalmol;

mass_tol = 1e-12;

fprintf('Raw system:\n');
fprintf('  M_raw = %d\n', Mraw);
fprintf('  E_raw = %.15f\n', E_raw);
fprintf('  delta = %.15e hartree = 1/100 kcal/mol\n\n', delta);

%% ============================================================
% 1. Correct block energy diagnostics for raw system
% =============================================================
diagRaw = compute_block_energy_contributions(coeff_raw, H_raw, twoBasis_raw);

fprintf('Raw block-energy diagnostics:\n');
fprintf('  Row/action contributions, sum exactly equals c^T H c:\n');
fprintf('    E_zero(row)   = %.15f\n', diagRaw.row.zero);
fprintf('    E_one(row)    = %.15f\n', diagRaw.row.one);
fprintf('    E_two_ud(row) = %.15f\n', diagRaw.row.two_ud);
fprintf('    sum(row)      = %.15f\n', diagRaw.row.total);
fprintf('  Diagonal-block self energies only, do NOT sum to total if cross-block couplings exist:\n');
fprintf('    E_zero(diag)  = %.15f\n', diagRaw.diag.zero);
fprintf('    E_one(diag)   = %.15f\n', diagRaw.diag.one);
fprintf('    E_two(diag)   = %.15f\n', diagRaw.diag.two_ud);
fprintf('    sum(diag)     = %.15f\n', diagRaw.diag.total);
fprintf('  Total c^T H c   = %.15f\n\n', real(coeff_raw' * H_raw * coeff_raw));

%% ============================================================
% 2. Coarsening scan over B(nu)
% =============================================================
coefAbs = abs(coeff_raw(:));
nuList = 0:80;

records = struct('nu',{},'thr',{},'M',{},'Mzero',{},'Mone',{},'Mtwo',{}, ...
                 'E',{},'dE',{},'keptDim',{},'idx',{});

fprintf('Scanning B(nu) = {|v_mu| >= 2^{-nu}} ...\n');
fprintf('   nu       threshold          M     Mzero   Mone   Mtwo        E(nu)              dE\n');

for inu = 1:numel(nuList)
    nu = nuList(inu);
    thr = 2^(-nu);

    idx = find(coefAbs >= thr);

    % Keep the zero/HF reference explicitly. In practice it is selected anyway,
    % but this avoids pathological loss of the rank-1 reference due to a
    % nearly cancelling coefficient representation.
    idx = unique([1; idx(:)], 'stable');

    Ssub = S_raw(idx,idx);
    Hsub = H_raw(idx,idx);

    [Esub, csub, keptDim] = solve_generalized_semidefinite(Hsub, Ssub, mass_tol);
    dE = Esub - E_raw;

    counts = count_blocks(twoBasis_raw(idx));

    records(inu).nu = nu;
    records(inu).thr = thr;
    records(inu).M = numel(idx);
    records(inu).Mzero = counts.zero;
    records(inu).Mone = counts.one_total;
    records(inu).Mtwo = counts.two_ud;
    records(inu).E = Esub;
    records(inu).dE = dE;
    records(inu).keptDim = keptDim;
    records(inu).idx = idx;

    fprintf('%5d   %.6e   %6d   %5d  %5d  %5d   %.15f   %.3e\n', ...
        nu, thr, numel(idx), counts.zero, counts.one_total, counts.two_ud, Esub, dE);
end

%% ============================================================
% 3. Select coarse_delta(B)
% =============================================================
valid = find([records.dE] <= delta + 1e-12 & [records.dE] >= -1e-10);

if isempty(valid)
    warning('No B(nu) satisfies dE <= delta. Selecting the largest candidate.');
    [~,bestPos] = max([records.M]);
    selected = records(bestPos);
else
    % coarse_delta: minimal cardinality among valid thresholded sets.
    [~,local] = min([records(valid).M]);
    selected = records(valid(local));
end

idxBest = selected.idx;
S0 = S_raw(idxBest,idxBest);
H0 = H_raw(idxBest,idxBest);
Hone0 = Hone_raw(idxBest,idxBest);
Vee0 = Vee_raw(idxBest,idxBest);
twoBasis0 = twoBasis_raw(idxBest);

[E0, coeff0, kept0] = solve_generalized_semidefinite(H0, S0, mass_tol);
counts0 = count_blocks(twoBasis0);
ener0 = compute_block_energy_contributions(coeff0, H0, twoBasis0);

fprintf('\n==================== SELECTED COARSE SYSTEM ====================\n');
fprintf('nu_selected = %d\n', selected.nu);
fprintf('threshold   = %.6e\n', selected.thr);
fprintf('M0          = %d\n', numel(idxBest));
fprintf('M_zero      = %d\n', counts0.zero);
fprintf('M_one       = %d\n', counts0.one_total);
fprintf('M_two_ud    = %d\n', counts0.two_ud);
fprintf('E0          = %.15f\n', E0);
fprintf('E_raw       = %.15f\n', E_raw);
fprintf('E0-E_raw    = %.6e hartree\n', E0 - E_raw);
fprintf('delta       = %.6e hartree\n', delta);
fprintf('keptDim     = %d / %d\n', kept0, numel(idxBest));
fprintf('\nTable-5.1-style row/action energy decomposition:\n');
fprintf('E_zero(row)   = %.15f\n', ener0.row.zero);
fprintf('E_one(row)    = %.15f\n', ener0.row.one);
fprintf('E_two_ud(row) = %.15f\n', ener0.row.two_ud);
fprintf('sum(row)      = %.15f\n', ener0.row.total);
fprintf('E0            = %.15f\n', E0);

fprintf('\nNeighborhood around selected candidate:\n');
for k = max(1, selected.nu-3):min(numel(records)-1, selected.nu+3)
    r = records(k+1); % because nu starts at 0
    marker = ' ';
    if r.nu == selected.nu
        marker = '*';
    end
    fprintf('%s nu=%2d | M=%4d | Mone=%4d | Mtwo=%4d | E=%.15f | dE=%.3e\n', ...
        marker, r.nu, r.M, r.Mone, r.Mtwo, r.E, r.dE);
end

%% ============================================================
% 4. Save coarsened system
% =============================================================
part3 = struct();
part3.inputFile = inputFile;
part3.delta = delta;
part3.hartree_per_kcalmol = hartree_per_kcalmol;
part3.records = records;
part3.selected = selected;
part3.idxBest = idxBest;
part3.twoBasis0 = twoBasis0;
part3.S0 = S0;
part3.H0 = H0;
part3.Hone0 = Hone0;
part3.Vee0 = Vee0;
part3.E0 = E0;
part3.coeff0 = coeff0;
part3.kept0 = kept0;
part3.counts0 = counts0;
part3.blockEnergy0 = ener0;
part3.raw = struct('M',Mraw,'E',E_raw,'counts',part2.counts);
part3.part2_params = part2.params;

saveName = sprintf('JH_He_Bthree0_coarsened_M%d_nu%d.mat', numel(idxBest), selected.nu);
save(saveName, 'part3', '-v7.3');

fprintf('\nSaved coarsened system: %s\n', saveName);
fprintf('Part 3 done. Next step: use this as B^[0] for Algorithm 4.1 adaptive refinement.\n');

%% ========================================================================
% Local functions
% ========================================================================

function [E0, coeff, nkeep] = solve_generalized_semidefinite(H, S, mass_tol)
    H = (H+H')/2;
    S = (S+S')/2;

    [U,D] = eig(S, 'vector');
    lam = real(D(:));
    [lam,ord] = sort(lam,'descend');
    U = U(:,ord);

    keep = lam > mass_tol * max(lam);
    nkeep = nnz(keep);

    X = U(:,keep) .* (1 ./ sqrt(lam(keep))).';

    Horth = X' * H * X;
    Horth = (Horth+Horth')/2;

    [Y,Eval] = eig(Horth);
    evals = real(diag(Eval));
    [evals,idx] = sort(evals,'ascend');

    y0 = Y(:,idx(1));
    E0 = evals(1);
    coeff = X * y0;
    coeff = coeff / sqrt(real(coeff' * S * coeff));
end

function counts = count_blocks(twoBasis)
    M = numel(twoBasis);
    blocks = strings(M,1);
    for i = 1:M
        blocks(i) = string(twoBasis(i).block);
    end

    counts = struct();
    counts.zero = nnz(blocks == "zero");
    counts.one_up = nnz(blocks == "one_up");
    counts.one_down = nnz(blocks == "one_down");
    counts.one_total = counts.one_up + counts.one_down;
    counts.two_ud = nnz(blocks == "two_ud");
    counts.total = M;
end

function out = compute_block_energy_contributions(c, H, twoBasis)
    M = numel(twoBasis);
    blocks = strings(M,1);
    for i = 1:M
        blocks(i) = string(twoBasis(i).block);
    end

    idxZero = find(blocks == "zero");
    idxOne = find(blocks == "one_up" | blocks == "one_down");
    idxTwo = find(blocks == "two_ud");

    out = struct();

    % Diagonal self-block contributions only. These omit cross-block terms.
    out.diag.zero = diag_energy(c,H,idxZero);
    out.diag.one = diag_energy(c,H,idxOne);
    out.diag.two_ud = diag_energy(c,H,idxTwo);
    out.diag.total = out.diag.zero + out.diag.one + out.diag.two_ud;

    % Row/action contributions:
    %   E_u = c_u^T H_{u,:} c
    % They sum exactly to c^T H c and correspond to decomposing Psi=sum_u Psi_u.
    out.row.zero = row_energy(c,H,idxZero);
    out.row.one = row_energy(c,H,idxOne);
    out.row.two_ud = row_energy(c,H,idxTwo);
    out.row.total = out.row.zero + out.row.one + out.row.two_ud;

    out.total = real(c' * H * c);
end

function e = diag_energy(c,H,idx)
    if isempty(idx)
        e = 0;
    else
        e = real(c(idx)' * H(idx,idx) * c(idx));
    end
end

function e = row_energy(c,H,idx)
    if isempty(idx)
        e = 0;
    else
        e = real(c(idx)' * H(idx,:) * c);
    end
end
