%% JH_LiGround_part1_HF_Q9_UHF_fast_centered_v3_staged.m
% Fast J.H.-style Li ground-state rank-1 HF reference, N=3, M_S=1/2.
%
% Ansatz used in J.H. Sec. 5.3 for Li:
%   Psi_HF = C A^(3,1/2)( psi_1^alpha \otimes psi_2^alpha \otimes psi_3^beta )
% where all three one-particle functions are s-type Gaussian contractions
% centered at the origin.  This script keeps the three occupied orbitals
% independent by default (open-shell UHF-like non-orthogonal determinant).
%
% Compared with the previous configurable version, the expensive parts are
% rewritten for the centered s-Gaussian case:
%   1) no general-center primitive loops;
%   2) only the five ERI tensors actually used by Li are built;
%   3) all one-/two-electron integrals and tensor contractions are vectorized;
%   4) all user controls remain at the top.
%
% Output:
%   JH_LiGround_HF_Q9_orbitals_fast.mat
%   also writes JH_LiGround_HF_Q9_orbitals.mat for downstream compatibility.

clear; clc; close all;
format long e;

%% ========================================================================
% 0. User controls
% ========================================================================
Z = 3;
Q = 9;
center = [0,0,0];
rngSeed = 31;
rng(rngSeed);

% Reference diagnostics from J.H. Table 5.13.
E_HF_JH_approx = -7.43271;          % \tilde{E}^{HF}_{tot} for Li in Table 5.13
E_HF_limit_ref  = -7.43273;          % HF limit diagnostic in Table 5.13
E_exact_ref     = -7.47806;          % exact diagnostic in Table 5.13

% ---------- Manual optimization controls ----------
numSigmaStarts   = 1;                % number of sigma initial guesses to use
sigmaStartIDs    = 1:numSigmaStarts; % e.g. [1 3 5]
maxOuterRestart  = 1;                % restarts per start
maxOuterIter     = 2000;              % fminsearch MaxIter for EACH solve
maxOuterFunEvals = 20000;            % fminsearch MaxFunEvals for EACH solve

% Inner coefficient alternating solver controls.
% For fixed sigmas, each occupied orbital coefficient vector is updated by
% an exact 9x9 generalized Rayleigh quotient.  8--15 iterations are usually
% enough; raising this improves strictness but costs time per outer eval.
innerAltMaxIter = 12;
innerAltTolE    = 5e-11;
innerMassTol    = 1e-12;

% Optimized variables are log(sigma).  Keep a broad but finite search box.
sigmaLower = 2e-4;
sigmaUpper = 1e2;

% Early stop relative to J.H. diagnostic HF value.  Set 0 to disable.
targetTolHF = 2e-6;

% Checkpoint / save names.
saveAfterEachSolve = true;
checkpointName = 'JH_LiGround_HF_Q9_orbitals_fast_staged_checkpoint.mat';
finalSaveNameFast = 'JH_LiGround_HF_Q9_orbitals_fast_staged.mat';
finalSaveNameCompat = 'JH_LiGround_HF_Q9_orbitals.mat';

% fminsearch settings.  Display='iter' is useful for debugging but adds console I/O.
optsOuter = optimset('Display','iter', ...
                     'MaxFunEvals', maxOuterFunEvals, ...
                     'MaxIter', maxOuterIter, ...
                     'TolX', 5e-9, ...
                     'TolFun', 5e-11);

% ---------- Staged optimizer controls ----------
% Stage A ties the alpha-core and beta-core primitive widths:
%   sigma_1^alpha == sigma_1^beta, leaving only 18 outer variables.
% This is very close to the physical Li doublet closed-core structure and
% gives fminsearch a much easier landscape.  Stage B releases the two core
% sigma sets and refines the full 27-variable UHF ansatz.
useTiedCorePrelude = true;
tiedMaxIter        = 2000;
tiedMaxFunEvals    = 18000;
tiedRestarts       = 1;
releaseFullUHF     = true;
fullReleaseIter    = maxOuterIter;
fullReleaseFunEvals= maxOuterFunEvals;

% After full release, do low-dimensional block refinements.  This often
% improves the last 1e-5 hartree because each fminsearch sees only 9 or
% 18 variables instead of the full 27-dimensional simplex.
useBlockRefinement = true;
blockSweeps        = 2;
blockMaxIter       = 1000;
blockMaxFunEvals   = 6000;

optsTied = optimset(optsOuter, 'MaxIter', tiedMaxIter, 'MaxFunEvals', tiedMaxFunEvals, 'TolX', 3e-9, 'TolFun', 3e-11);
optsRelease = optimset(optsOuter, 'MaxIter', fullReleaseIter, 'MaxFunEvals', fullReleaseFunEvals, 'TolX', 3e-9, 'TolFun', 3e-11);
optsBlock = optimset(optsOuter, 'MaxIter', blockMaxIter, 'MaxFunEvals', blockMaxFunEvals, 'TolX', 2e-9, 'TolFun', 2e-11);

fprintf('\n============================================================\n');
fprintf('STAGED fast centered-s Li ground-state non-orthogonal UHF reference\n');
fprintf('System: N=3, M_S=1/2, N_up=2, N_down=1, Z=%d, Q=%d\n', Z, Q);
fprintf('Ansatz: A_up[psi_1^alpha, psi_2^alpha] * psi_1^beta; all centered s-Gaussians.\n');
fprintf('Manual controls: starts=%d, restarts=%d, MaxIter=%d, MaxFunEvals=%d, innerMax=%d\n', ...
    numSigmaStarts, maxOuterRestart, maxOuterIter, maxOuterFunEvals, innerAltMaxIter);
fprintf('Staged optimizer: tiedPrelude=%d, releaseUHF=%d, blockRefine=%d, blockSweeps=%d\n', ...
    useTiedCorePrelude, releaseFullUHF, useBlockRefinement, blockSweeps);
fprintf('Diagnostic J.H. Table 5.13 tilde E_HF ~= %.8f hartree\n', E_HF_JH_approx);
fprintf('============================================================\n\n');

%% ========================================================================
% 1. Starts
% ========================================================================
startLibrary = make_sigma_starts_li_fast(Q);
if isempty(sigmaStartIDs)
    sigmaStartIDs = 1:min(numSigmaStarts, numel(startLibrary));
end
sigmaStartIDs = sigmaStartIDs(:).';
if max(sigmaStartIDs) > numel(startLibrary)
    error('Requested sigmaStartIDs exceed available startLibrary length=%d.', numel(startLibrary));
end

best = empty_best();
stopAllStarts = false;

%% ========================================================================
% 2. Staged outer optimization over log-sigmas
% ========================================================================
for is0 = 1:numel(sigmaStartIDs)
    sid = sigmaStartIDs(is0);
    sig0 = startLibrary{sid}(:);
    if numel(sig0) ~= 3*Q
        error('Start %d has length %d, expected %d.', sid, numel(sig0), 3*Q);
    end
    logsFull = log(sig0);

    [E0, ~, comps0] = li_hf_outer_objective_fast(logsFull, Q, Z, center, ...
        innerAltMaxIter, innerAltTolE, innerMassTol, sigmaLower, sigmaUpper);
    fprintf('\nStart %d/%d, library id %d: initial full-UHF E = %.15f | D_alpha=%.3e | n_beta=%.3e | innerIt=%d\n', ...
        is0, numel(sigmaStartIDs), sid, E0, comps0.Dalpha, comps0.n3, comps0.altIter);

    % --------------------------------------------------------------------
    % Stage A: tied-core 18D prelude
    % --------------------------------------------------------------------
    if useTiedCorePrelude
        y = full_logs_to_tied_logs(logsFull, Q);  % [core(9); valence(9)]
        Etied0 = tied_objective(y, Q, Z, center, innerAltMaxIter, innerAltTolE, innerMassTol, sigmaLower, sigmaUpper);
        fprintf('  Stage A tied-core start: E = %.15f\n', Etied0);

        for rr = 0:tiedRestarts
            if rr > 0
                fprintf('  Stage A tied-core restart %d/%d from current y.\n', rr, tiedRestarts);
            end
            [y, ~, exitflag, output] = fminsearch( ...
                @(yy) tied_objective(yy, Q, Z, center, innerAltMaxIter, innerAltTolE, innerMassTol, sigmaLower, sigmaUpper), ...
                y, optsTied);
            logsFull = tied_logs_to_full_logs(y, Q);
            [Etied, orbitals, comps] = li_hf_outer_objective_fast(logsFull, Q, Z, center, ...
                innerAltMaxIter, innerAltTolE, innerMassTol, sigmaLower, sigmaUpper);
            fprintf('  Stage A tied-core result rr=%d: E=%.15f | diff J.H.=%.3e | D_alpha=%.3e | s12=%.6e\n', ...
                rr, Etied, Etied - E_HF_JH_approx, comps.Dalpha, comps.s12);
            print_li_components(comps);
            if Etied < best.E
                best = update_best(best, Etied, logsFull, orbitals, comps, exitflag, output, sid, -100-rr);
            end
            if saveAfterEachSolve
                save_hf_checkpoint(checkpointName, best, logsFull, Etied, orbitals, comps, sid, -100-rr, exitflag, output, ...
                    Z, Q, center, E_HF_JH_approx, E_HF_limit_ref, E_exact_ref, innerAltMaxIter, innerAltTolE, innerMassTol, ...
                    numSigmaStarts, maxOuterRestart, maxOuterIter, maxOuterFunEvals, sigmaLower, sigmaUpper);
                fprintf('  checkpoint saved after tied-core stage: %s\n', checkpointName);
            end
            if targetTolHF > 0 && abs(Etied - E_HF_JH_approx) < targetTolHF
                fprintf('Reached J.H. diagnostic HF tolerance in tied-core prelude; still doing optional full/block refinement if enabled.\n');
            end
        end
    end

    % --------------------------------------------------------------------
    % Stage B: full 27D UHF release
    % --------------------------------------------------------------------
    if releaseFullUHF
        % A tiny deterministic perturbation prevents the released alpha-core
        % and beta-core vertices from being exactly degenerate, but keeps the
        % solution close to the tied-core minimum.
        pert = zeros(size(logsFull));
        pert(2*Q+1:3*Q) = 1e-3 * sin((1:Q).');
        logsFull = logsFull + pert;

        fprintf('  Stage B full-UHF release from E = %.15f\n', ...
            li_hf_outer_objective_fast(logsFull, Q, Z, center, innerAltMaxIter, innerAltTolE, innerMassTol, sigmaLower, sigmaUpper));
        [logsFull, ~, exitflag, output] = fminsearch( ...
            @(xx) li_hf_outer_objective_fast(xx, Q, Z, center, innerAltMaxIter, innerAltTolE, innerMassTol, sigmaLower, sigmaUpper), ...
            logsFull, optsRelease);
        [Echeck, orbitals, comps] = li_hf_outer_objective_fast(logsFull, Q, Z, center, ...
            innerAltMaxIter, innerAltTolE, innerMassTol, sigmaLower, sigmaUpper);
        fprintf('\n  Stage B full-UHF result: E=%.15f | diff J.H.=%.3e | D_alpha=%.3e | s12=%.6e | n_beta=%.6e | innerIt=%d\n', ...
            Echeck, Echeck - E_HF_JH_approx, comps.Dalpha, comps.s12, comps.n3, comps.altIter);
        print_li_components(comps);
        if Echeck < best.E
            best = update_best(best, Echeck, logsFull, orbitals, comps, exitflag, output, sid, 0);
        end
        if saveAfterEachSolve
            save_hf_checkpoint(checkpointName, best, logsFull, Echeck, orbitals, comps, sid, 0, exitflag, output, ...
                Z, Q, center, E_HF_JH_approx, E_HF_limit_ref, E_exact_ref, innerAltMaxIter, innerAltTolE, innerMassTol, ...
                numSigmaStarts, maxOuterRestart, maxOuterIter, maxOuterFunEvals, sigmaLower, sigmaUpper);
            fprintf('  checkpoint saved after full-UHF release: %s\n', checkpointName);
        end
    end

    % --------------------------------------------------------------------
    % Stage C: block coordinate simplex refinements
    % --------------------------------------------------------------------
    if useBlockRefinement
        logsFull = best.logsigma;
        for sw = 1:blockSweeps
            fprintf('\n  Stage C block refinement sweep %d/%d starts at E=%.15f\n', sw, blockSweeps, best.E);
            % The order is physically motivated: first valence, then beta-core,
            % then alpha-core, then a tied core-pair adjustment.
            blockList = {Q+1:2*Q, 2*Q+1:3*Q, 1:Q, [1:Q, 2*Q+1:3*Q]};
            blockNames = {'alpha-valence','beta-core','alpha-core','two-core-block'};
            for bb = 1:numel(blockList)
                ids = blockList{bb};
                z0 = logsFull(ids);
                fprintf('    refining block %-15s (%d vars) from E=%.15f\n', blockNames{bb}, numel(ids), best.E);
                [z, ~, exitflag, output] = fminsearch( ...
                    @(zz) block_objective(zz, logsFull, ids, Q, Z, center, innerAltMaxIter, innerAltTolE, innerMassTol, sigmaLower, sigmaUpper), ...
                    z0, optsBlock);
                logsTrial = logsFull; logsTrial(ids)=z;
                [Etrial, orbitals, comps] = li_hf_outer_objective_fast(logsTrial, Q, Z, center, ...
                    innerAltMaxIter, innerAltTolE, innerMassTol, sigmaLower, sigmaUpper);
                fprintf('      block result E=%.15f | gain=%.3e | diff J.H.=%.3e\n', Etrial, best.E-Etrial, Etrial-E_HF_JH_approx);
                if Etrial < best.E
                    logsFull = logsTrial;
                    best = update_best(best, Etrial, logsFull, orbitals, comps, exitflag, output, sid, sw*10+bb);
                    if saveAfterEachSolve
                        save_hf_checkpoint(checkpointName, best, logsFull, Etrial, orbitals, comps, sid, sw*10+bb, exitflag, output, ...
                            Z, Q, center, E_HF_JH_approx, E_HF_limit_ref, E_exact_ref, innerAltMaxIter, innerAltTolE, innerMassTol, ...
                            numSigmaStarts, maxOuterRestart, maxOuterIter, maxOuterFunEvals, sigmaLower, sigmaUpper);
                    end
                else
                    logsFull = best.logsigma;
                end
            end
        end
    end

    fprintf('\nCurrent best after start id %d: E=%.15f | diff J.H.=%.3e\n', ...
        sid, best.E, best.E - E_HF_JH_approx);
    if targetTolHF > 0 && best.E <= E_HF_JH_approx + targetTolHF
        fprintf('Global target tolerance reached; stopping remaining starts.\n');
        break;
    end
end

if isempty(best.logsigma)
    error('No valid Li HF solution was found. Try more starts/restarts/iterations.');
end

%% ========================================================================
% 3. Final reconstruction and save
% ========================================================================
[Efinal, orbitals, comps] = li_hf_outer_objective_fast(best.logsigma, Q, Z, center, ...
    innerAltMaxIter, innerAltTolE, innerMassTol, sigmaLower, sigmaUpper);
orbitals = sort_orbitals_by_sigma(orbitals);
[Efinal, orbitals, comps] = energy_from_stored_li_orbitals_fast(orbitals, Z, center);

fprintf('\n==================== FINAL Li GROUND-STATE HF REFERENCE ====================\n');
fprintf('E_HF_Q9_Li_ground = %.15f hartree\n', Efinal);
fprintf('J.H. Table 5.13 tilde HF = %.15f hartree\n', E_HF_JH_approx);
fprintf('diff                    = %.6e hartree\n', Efinal - E_HF_JH_approx);
fprintf('HF limit diagnostic      = %.15f hartree | diff = %.6e\n', E_HF_limit_ref, Efinal - E_HF_limit_ref);
fprintf('exact ref diagnostic     = %.15f hartree | corr gap approx = %.6e\n', E_exact_ref, Efinal - E_exact_ref);
fprintf('n1,n2,s12,Dalpha,n3      = %.15f %.15f %.15e %.15e %.15f\n', ...
    comps.n1, comps.n2, comps.s12, comps.Dalpha, comps.n3);
print_li_components(comps);

for p = 1:3
    fprintf('\nOrbital %d (%s) independent Q=%d primitive contraction:\n', p, orbitals(p).spinLabel, Q);
    fprintf(' q          coeff                  sigma\n');
    for q = 1:numel(orbitals(p).terms)
        sigma = 1 / sqrt(orbitals(p).terms(q).alpha);
        fprintf('%2d   %+.16e   %.16e\n', q, orbitals(p).terms(q).coef, sigma);
    end
end

hf = struct();
hf.system = 'Li_ground';
hf.description = 'Fast centered-s J.H.-style non-orthogonal UHF for Li ground state, N_up=2,N_down=1';
hf.Z = Z; hf.N = 3; hf.N_up = 2; hf.N_down = 1; hf.MS = 1/2; hf.Q = Q;
hf.center = center;
hf.E = Efinal;
hf.E_HF_JH_approx = E_HF_JH_approx;
hf.E_HF_limit_ref = E_HF_limit_ref;
hf.E_exact_ref = E_exact_ref;
hf.correlation_gap_approx = Efinal - E_exact_ref;
hf.orbitals = orbitals;
hf.comps = comps;
hf.best = best;
hf.params = struct('numSigmaStarts',numSigmaStarts,'sigmaStartIDs',sigmaStartIDs, ...
    'maxOuterRestart',maxOuterRestart,'maxOuterIter',maxOuterIter,'maxOuterFunEvals',maxOuterFunEvals, ...
    'innerAltMaxIter',innerAltMaxIter,'innerAltTolE',innerAltTolE,'innerMassTol',innerMassTol, ...
    'sigmaLower',sigmaLower,'sigmaUpper',sigmaUpper,'rngSeed',rngSeed, ...
    'implementation','centered_s_vectorized_minimal_eri_staged_tied_release_block', ...
    'useTiedCorePrelude',useTiedCorePrelude,'tiedMaxIter',tiedMaxIter,'tiedRestarts',tiedRestarts, ...
    'releaseFullUHF',releaseFullUHF,'useBlockRefinement',useBlockRefinement,'blockSweeps',blockSweeps);
hf.savedAt = datestr(now);
save(finalSaveNameFast, 'hf', '-v7.3');
save(finalSaveNameCompat, 'hf', '-v7.3');
fprintf('\nSaved: %s\nSaved compatibility copy: %s\n', finalSaveNameFast, finalSaveNameCompat);

try
    xgrid = linspace(-18,18,2500).';
    figure('Color','w'); hold on;
    for p = 1:3
        y = eval_orbital_on_x_axis(orbitals(p), xgrid);
        plot(xgrid, y, 'LineWidth', 1.2);
    end
    xlabel('x_1 [bohr]'); ylabel('\psi(x_1,0,0)');
    title(sprintf('Li ground-state HF orbitals, E=%.10f', Efinal));
    legend('\psi_1^{\alpha}','\psi_2^{\alpha}','\psi_1^{\beta}','Location','best'); grid on;
catch ME
    warning('Plot skipped: %s', ME.message);
end

%% ========================================================================
% Local functions
% ========================================================================
function starts = make_sigma_starts_li_fast(Q)
    starts = {};
    compact1 = logspace(log10(0.006), log10(1.10), Q).';
    compact2 = logspace(log10(0.010), log10(1.70), Q).';
    diffuse1 = logspace(log10(0.045), log10(18.0), Q).';
    diffuse2 = logspace(log10(0.070), log10(35.0), Q).';

    starts{end+1} = [compact1; diffuse1; compact1*1.03];
    starts{end+1} = [compact1*0.90; diffuse2; compact1*1.10];
    starts{end+1} = [compact2; diffuse1; compact2*1.02];

    if exist('JH_He_HF_Q9_orbital_v2.mat','file')
        try
            S = load('JH_He_HF_Q9_orbital_v2.mat','hf');
            sigHe = S.hf.sigma(:);
            if numel(sigHe) == Q
                starts{end+1} = [sigHe*sqrt(2/3); diffuse1; sigHe*sqrt(2/3)*1.02];
            end
        catch
        end
    end

    base = starts{1};
    for k = 1:6
        starts{end+1} = base .* exp(0.15 * randn(3*Q,1));
    end
end


function y = full_logs_to_tied_logs(logsFull,Q)
    core = 0.5*(logsFull(1:Q) + logsFull(2*Q+1:3*Q));
    val  = logsFull(Q+1:2*Q);
    y = [core(:); val(:)];
end

function logsFull = tied_logs_to_full_logs(y,Q)
    y = y(:);
    core = y(1:Q);
    val  = y(Q+1:2*Q);
    logsFull = [core; val; core];
end

function E = tied_objective(y,Q,Z,center,innerMaxIter,innerTolE,massTol,sigmaLower,sigmaUpper)
    logsFull = tied_logs_to_full_logs(y,Q);
    E = li_hf_outer_objective_fast(logsFull,Q,Z,center,innerMaxIter,innerTolE,massTol,sigmaLower,sigmaUpper);
end

function E = block_objective(z,logsFull,ids,Q,Z,center,innerMaxIter,innerTolE,massTol,sigmaLower,sigmaUpper)
    logsTrial = logsFull;
    logsTrial(ids) = z(:);
    E = li_hf_outer_objective_fast(logsTrial,Q,Z,center,innerMaxIter,innerTolE,massTol,sigmaLower,sigmaUpper);
end

function [E, orbitals, comps] = li_hf_outer_objective_fast(logsigma, Q, Z, center, innerMaxIter, innerTolE, massTol, sigmaLower, sigmaUpper)
    orbitals = [];
    comps = empty_li_comps();
    logsigma = logsigma(:);
    if numel(logsigma) ~= 3*Q || any(~isfinite(logsigma))
        E = 1e30; return;
    end
    low = log(sigmaLower); high = log(sigmaUpper);
    penalty = 0;
    if any(logsigma < low),  penalty = penalty + 1e6*sum((logsigma(logsigma<low)-low).^2); end
    if any(logsigma > high), penalty = penalty + 1e6*sum((logsigma(logsigma>high)-high).^2); end
    logsigma = min(max(logsigma, low), high);

    sigma1 = exp(logsigma(1:Q));
    sigma2 = exp(logsigma(Q+1:2*Q));
    sigma3 = exp(logsigma(2*Q+1:3*Q));
    alpha = {1./sigma1.^2, 1./sigma2.^2, 1./sigma3.^2};

    try
        ints = build_li_integrals_centered_fast(alpha, Z);
        [coeffs, Eraw, comps] = solve_li_coefficients_alt_fast(ints, innerMaxIter, innerTolE, massTol);
        orbitals = make_li_orbitals_from_alpha(alpha, coeffs, center);
    catch ME
        E = 1e25 + penalty;
        comps.failMessage = ME.message;
        return;
    end

    if comps.Dalpha < 1e-8
        penalty = penalty + 1e4 * (log(max(comps.Dalpha,realmin)) - log(1e-8))^2;
    end
    if comps.n3 < 1e-8
        penalty = penalty + 1e4 * (log(max(comps.n3,realmin)) - log(1e-8))^2;
    end
    E = Eraw + penalty;
end

function ints = build_li_integrals_centered_fast(alpha, Z)
    a1 = alpha{1}(:); a2 = alpha{2}(:); a3 = alpha{3}(:);
    [ints.S11,ints.H11] = one_mats_centered(a1,a1,Z);
    [ints.S22,ints.H22] = one_mats_centered(a2,a2,Z);
    [ints.S33,ints.H33] = one_mats_centered(a3,a3,Z);
    [ints.S12,ints.H12] = one_mats_centered(a1,a2,Z);

    ints.S11 = symm(ints.S11); ints.H11 = symm(ints.H11);
    ints.S22 = symm(ints.S22); ints.H22 = symm(ints.H22);
    ints.S33 = symm(ints.S33); ints.H33 = symm(ints.H33);

    % Only five Coulomb tensors are needed by the Li alpha-alpha-beta energy.
    ints.G1122 = eri_centered(a1,a1,a2,a2); % J12
    ints.G1221 = eri_centered(a1,a2,a2,a1); % K12
    ints.G1133 = eri_centered(a1,a1,a3,a3);
    ints.G2233 = eri_centered(a2,a2,a3,a3);
    ints.G1233 = eri_centered(a1,a2,a3,a3);
end

function [S,H] = one_mats_centered(a,b,Z)
    [A,B] = ndgrid(a(:), b(:));
    p = A + B;
    Nprod = (A.*B).^(3/4) / (pi^(3/2));
    S = Nprod .* (2*pi./p).^(3/2);
    T = 0.5 .* A .* B .* (3./p) .* S;
    Ven = -Z .* Nprod .* (4*pi./p);
    H = T + Ven;
end

function G = eri_centered(a,b,c,d)
    [A,B,C,D] = ndgrid(a(:), b(:), c(:), d(:));
    p = A + B;
    q = C + D;
    Nprod = (A.*B.*C.*D).^(3/4) / (pi^3);
    G = Nprod .* (8*sqrt(2)*pi^(5/2)) ./ (p .* q .* sqrt(p+q));
end

function [coeffs, E, comps] = solve_li_coefficients_alt_fast(ints, maxIter, tolE, massTol)
    coeffs = initial_li_coefficients_fast(ints, massTol);
    Eprev = inf;
    comps = empty_li_comps();
    for it = 1:maxIter
        for p = 1:3
            [A,B] = build_orbital_update_AB_fast(p, coeffs, ints);
            [~, cp, ok] = solve_lowest_gen(A, B, massTol);
            if ~ok, error('Coefficient update failed for orbital %d.', p); end
            coeffs{p} = normalize_with_S(cp, get_Spp(ints,p));
            coeffs{p} = stabilize_sign(coeffs{p}, p);
        end
        comps = li_energy_comps_fast(ints, coeffs);
        E = comps.E;
        if isfinite(Eprev) && abs(E-Eprev) < tolE*max(1,abs(E))
            comps.altIter = it;
            return;
        end
        Eprev = E;
    end
    comps.altIter = maxIter;
end

function coeffs = initial_li_coefficients_fast(ints, massTol)
    coeffs = cell(1,3);
    [~,V1] = solve_all_gen(ints.H11, ints.S11, massTol); coeffs{1} = normalize_with_S(real(V1(:,1)), ints.S11);
    [~,V2] = solve_all_gen(ints.H22, ints.S22, massTol);
    if size(V2,2) >= 2, coeffs{2} = normalize_with_S(real(V2(:,2)), ints.S22);
    else, coeffs{2} = normalize_with_S(real(V2(:,1)), ints.S22); end
    [~,V3] = solve_all_gen(ints.H33, ints.S33, massTol); coeffs{3} = normalize_with_S(real(V3(:,1)), ints.S33);
end

function [A,B] = build_orbital_update_AB_fast(p, coeffs, ints)
    Q = numel(coeffs{p});
    A = zeros(Q,Q); B = zeros(Q,Q);
    Ndiag = zeros(Q,1); Ddiag = zeros(Q,1);
    for i = 1:Q
        test = coeffs; e = zeros(Q,1); e(i)=1; test{p}=e;
        [N,D] = li_energy_num_den_fast(ints,test);
        A(i,i)=N; B(i,i)=D; Ndiag(i)=N; Ddiag(i)=D;
    end
    for i = 1:Q
        for j = i+1:Q
            test = coeffs; e = zeros(Q,1); e([i j])=1; test{p}=e;
            [N,D] = li_energy_num_den_fast(ints,test);
            Aij = 0.5*(N-Ndiag(i)-Ndiag(j));
            Bij = 0.5*(D-Ddiag(i)-Ddiag(j));
            A(i,j)=Aij; A(j,i)=Aij; B(i,j)=Bij; B(j,i)=Bij;
        end
    end
    A=symm(A); B=symm(B);
end

function [N,D] = li_energy_num_den_fast(ints, coeffs)
    comps = li_energy_comps_fast(ints, coeffs);
    D = comps.normDen;
    N = comps.E * D;
end

function comps = li_energy_comps_fast(ints, coeffs)
    c1=coeffs{1}; c2=coeffs{2}; c3=coeffs{3};
    n1  = real(c1' * ints.S11 * c1);
    n2  = real(c2' * ints.S22 * c2);
    n3  = real(c3' * ints.S33 * c3);
    s12 = real(c1' * ints.S12 * c2);
    Dalpha = n1*n2 - s12^2;
    normDen = Dalpha * n3;
    if Dalpha <= 0 || n3 <= 0 || ~isfinite(normDen)
        error('Invalid Li HF norm: Dalpha=%g, n3=%g.', Dalpha, n3);
    end
    h11 = real(c1' * ints.H11 * c1);
    h22 = real(c2' * ints.H22 * c2);
    h33 = real(c3' * ints.H33 * c3);
    h12 = real(c1' * ints.H12 * c2);

    J12   = c4(ints.G1122,c1,c1,c2,c2);
    K12   = c4(ints.G1221,c1,c2,c2,c1);
    V1133 = c4(ints.G1133,c1,c1,c3,c3);
    V2233 = c4(ints.G2233,c2,c2,c3,c3);
    V1233 = c4(ints.G1233,c1,c2,c3,c3);

    E_alpha_one = (n2*h11 + n1*h22 - 2*s12*h12) / Dalpha;
    E_beta_one  = h33 / n3;
    E_aa        = (J12 - K12) / Dalpha;
    E_ab        = (n2*V1133 + n1*V2233 - 2*s12*V1233) / (Dalpha*n3);
    E = E_alpha_one + E_beta_one + E_aa + E_ab;

    comps = empty_li_comps();
    comps.E = real(E); comps.normDen = real(normDen);
    comps.n1=real(n1); comps.n2=real(n2); comps.n3=real(n3); comps.s12=real(s12); comps.Dalpha=real(Dalpha);
    comps.h11=real(h11); comps.h22=real(h22); comps.h33=real(h33); comps.h12=real(h12);
    comps.J12=real(J12); comps.K12=real(K12); comps.V1133=real(V1133); comps.V2233=real(V2233); comps.V1233=real(V1233);
    comps.E_alpha_one=real(E_alpha_one); comps.E_beta_one=real(E_beta_one); comps.E_aa=real(E_aa); comps.E_ab=real(E_ab);
end

function val = c4(T,a,b,c,d)
    X = T .* reshape(a,[],1,1,1) .* reshape(b,1,[],1,1) .* reshape(c,1,1,[],1) .* reshape(d,1,1,1,[]);
    val = real(sum(X(:)));
end

function S = get_Spp(ints,p)
    switch p
        case 1, S=ints.S11;
        case 2, S=ints.S22;
        case 3, S=ints.S33;
    end
end

function c = stabilize_sign(c,p)
    if p == 2
        [~,im] = max(abs(c)); if c(im)<0, c=-c; end
    else
        if sum(c)<0, c=-c; end
    end
end

function orbitals = make_li_orbitals_from_alpha(alpha, coeffs, center)
    labels = {'alpha_core','alpha_valence','beta_core'};
    spins  = {'alpha','alpha','beta'};
    orbitals = repmat(struct('terms',[],'spin','','spinLabel','','orbitalType','s','role',''),3,1);
    for p=1:3
        ap = alpha{p}(:);
        terms = repmat(struct('coef',0,'alpha',0,'center',[0 0 0]), numel(ap),1);
        for q=1:numel(ap)
            terms(q).coef = coeffs{p}(q);
            terms(q).alpha = ap(q);
            terms(q).center = center;
        end
        orbitals(p).terms=terms; orbitals(p).spin=spins{p}; orbitals(p).spinLabel=labels{p};
        orbitals(p).orbitalType='s'; orbitals(p).role=labels{p};
    end
end

function [Efinal, orbitals, comps] = energy_from_stored_li_orbitals_fast(orbitals, Z, center)
    alpha = cell(1,3); coeffs = cell(1,3);
    for p=1:3
        terms = orbitals(p).terms;
        alpha{p} = arrayfun(@(t) t.alpha, terms(:));
        coeffs{p} = arrayfun(@(t) t.coef, terms(:));
    end
    ints = build_li_integrals_centered_fast(alpha, Z);
    comps = li_energy_comps_fast(ints, coeffs);
    Efinal = comps.E;
end

function orbitals = sort_orbitals_by_sigma(orbitals)
    for p=1:numel(orbitals)
        terms = orbitals(p).terms;
        sig = arrayfun(@(t) 1/sqrt(t.alpha), terms(:));
        [~,ord] = sort(sig,'ascend');
        orbitals(p).terms = terms(ord);
    end
end

function [E,c,ok] = solve_lowest_gen(A,B,massTol)
    A=symm(A); B=symm(B);
    [U,d] = eig(B,'vector'); d=real(d(:));
    [d,ord]=sort(d,'descend'); U=U(:,ord);
    keep = d > massTol*max(abs(d));
    if nnz(keep)<1, E=inf; c=[]; ok=false; return; end
    X = U(:,keep) .* (1./sqrt(d(keep))).';
    Aorth = symm(X'*A*X);
    [Y,D2] = eig(Aorth);
    vals=real(diag(D2)); [E,idx]=min(vals);
    c = real(X*Y(:,idx));
    ok = isfinite(E) && all(isfinite(c));
end

function [vals,vecs] = solve_all_gen(A,B,massTol)
    A=symm(A); B=symm(B);
    [U,d] = eig(B,'vector'); d=real(d(:));
    [d,ord]=sort(d,'descend'); U=U(:,ord);
    keep = d > massTol*max(abs(d));
    X = U(:,keep) .* (1./sqrt(d(keep))).';
    Aorth = symm(X'*A*X);
    [Y,D2]=eig(Aorth); vals=real(diag(D2));
    [vals,idx]=sort(vals,'ascend'); vecs=real(X*Y(:,idx));
end

function c = normalize_with_S(c,S)
    n = real(c'*S*c);
    if n<=0 || ~isfinite(n), error('normalize_with_S failed, n=%g.', n); end
    c = c./sqrt(n);
end

function A = symm(A)
    A = (A+A')/2;
end

function comps = empty_li_comps()
    comps = struct('E',NaN,'normDen',NaN,'n1',NaN,'n2',NaN,'n3',NaN,'s12',NaN,'Dalpha',NaN, ...
        'h11',NaN,'h22',NaN,'h33',NaN,'h12',NaN,'J12',NaN,'K12',NaN, ...
        'V1133',NaN,'V2233',NaN,'V1233',NaN,'E_alpha_one',NaN,'E_beta_one',NaN, ...
        'E_aa',NaN,'E_ab',NaN,'altIter',NaN,'failMessage','');
end

function best = empty_best()
    best = struct('E',inf,'logsigma',[],'orbitals',[],'comps',[], ...
        'exitflag',[],'output',[],'startID',[],'restartID',[]);
end

function best = update_best(best,E,logs,orbitals,comps,exitflag,output,startID,restartID)
    best.E=E; best.logsigma=logs; best.orbitals=orbitals; best.comps=comps;
    best.exitflag=exitflag; best.output=output; best.startID=startID; best.restartID=restartID;
end

function save_hf_checkpoint(checkpointName, best, logs, Echeck, orbitals, comps, startID, restartID, exitflag, output, ...
    Z, Q, center, E_HF_JH_approx, E_HF_limit_ref, E_exact_ref, innerAltMaxIter, innerAltTolE, innerMassTol, ...
    numSigmaStarts, maxOuterRestart, maxOuterIter, maxOuterFunEvals, sigmaLower, sigmaUpper)
    hf_checkpoint = struct();
    hf_checkpoint.best = best;
    hf_checkpoint.current = struct('logsigma',logs,'E',Echeck,'orbitals',orbitals,'comps',comps, ...
        'startID',startID,'restartID',restartID,'exitflag',exitflag,'output',output);
    hf_checkpoint.params = struct('Z',Z,'Q',Q,'center',center,'E_HF_JH_approx',E_HF_JH_approx, ...
        'E_HF_limit_ref',E_HF_limit_ref,'E_exact_ref',E_exact_ref, ...
        'innerAltMaxIter',innerAltMaxIter,'innerAltTolE',innerAltTolE,'innerMassTol',innerMassTol, ...
        'numSigmaStarts',numSigmaStarts,'maxOuterRestart',maxOuterRestart, ...
        'maxOuterIter',maxOuterIter,'maxOuterFunEvals',maxOuterFunEvals, ...
        'sigmaLower',sigmaLower,'sigmaUpper',sigmaUpper);
    hf_checkpoint.savedAt = datestr(now);
    save(checkpointName,'hf_checkpoint','-v7.3');
end

function print_li_components(comps)
    fprintf('  one-alpha = %.15f | one-beta = %.15f | V_aa = %.15f | V_ab = %.15f | sum = %.15f\n', ...
        comps.E_alpha_one, comps.E_beta_one, comps.E_aa, comps.E_ab, comps.E);
    fprintf('  h11=%.15f h22=%.15f h33=%.15f h12=%.15f\n', comps.h11, comps.h22, comps.h33, comps.h12);
    fprintf('  J12=%.15f K12=%.15f | V1133=%.15f V2233=%.15f V1233=%.15f\n', ...
        comps.J12, comps.K12, comps.V1133, comps.V2233, comps.V1233);
end

function y = eval_orbital_on_x_axis(orb,xgrid)
    y=zeros(size(xgrid));
    for q=1:numel(orb.terms)
        alpha=orb.terms(q).alpha; c=orb.terms(q).coef; N=(alpha/pi)^(3/4);
        y = y + c*N*exp(-0.5*alpha*xgrid.^2);
    end
end
