%% JH_BeGround_part1_HF_RHF_SCF_DIIS_outerSigma_v3_noDIISwarning.m
% Be ground-state HF by a more stable Roothaan RHF-SCF + DIIS solver.
%
% Motivation:
%   The previous Be tied-sigma UHF alternating-coefficient code is correct but
%   expensive and can become numerically unstable during the cheap search stage.
%   For Be ground state (closed-shell singlet), a better HF inner solver is the
%   standard closed-shell Roothaan SCF problem.  For every trial nonlinear sigma
%   vector, this code solves the fixed-basis RHF problem by diagonalizing the
%   Fock matrix with DIIS acceleration.  Therefore no determinant-denominator
%   Rayleigh collapse and no negative opposite-spin Coulomb artefact can be
%   accepted by the objective.
%
% Structure:
%   - Use two 9-point primitive libraries: core-like and valence-like.
%   - Combine them into an 18-primitive spatial AO basis for RHF-SCF.
%   - Optimize the 18 sigma values in staged fminsearch, exactly like the Li
%     staged style, but the coefficient problem is solved by SCF, not by
%     alternating four occupied spin orbitals.
%
% Output:
%   JH_BeGround_HF_Q9_RHF_SCF_DIIS_v3_noDIISwarning.mat
%   JH_BeGround_HF_Q9_orbitals.mat   % compatibility copy for coarsening/adaptive
%
% Important note:
%   The two occupied spatial orbitals saved here may each contain all 18
%   primitive functions.  This is the mathematically stable RHF solution in the
%   union basis.  If a later script strictly assumes exactly Q=9 terms per
%   orbital, set saveCompatibilityCopy=false and use this result only as a warm
%   start for the old tied-sigma code.  If later scripts simply consume
%   hf.orbitals(p).terms, this file is directly usable.

clear; clc; close all;
format long e;

%% ========================================================================
% 0. User controls
% ========================================================================
Z = 4;
Nelec = 4;
nocc = Nelec/2;
QperShell = 6;
nBasis = 2*QperShell;
center = [0 0 0];
rngSeed = 41; rng(rngSeed);

% Diagnostics from J.H. tables / known Be values.
E_HF_JH_approx = -14.57296;
E_HF_limit_ref = -14.57302;
E_exact_ref    = -14.66736;

% Staged optimization.  Start modest; SCF objective is much safer than the old
% alternating objective, so larger MaxIter can be used if desired.
numSigmaStarts = 1;
sigmaStartIDs = 1:numSigmaStarts;
preludeMaxIter      = 50000;
preludeMaxFunEvals  = 100000;
blockSweeps         = 2;
blockMaxIter        = 25000;
blockMaxFunEvals    = 70000;
finalPolishIter     = 20000;
finalPolishFunEvals = 80000;

usePrelude = true;
useBlockRefinement = true;
useFinalPolish = true;

% Nonlinear sigma bounds.  These are deliberately conservative to avoid
% near-linear dependence in the 18-AO union basis.
sigmaLower = 2e-4;
sigmaUpper = 1e2;
minEigSObjective = 1e-10;
condSMaxObjective = 1e13;
energyFloorObjective = E_exact_ref - 5e-3;
badPenalty = 1e12;
targetTolHF = 4e-6;

% SCF controls.
scf.maxIter = 45;
scf.tolE = 5e-10;
scf.tolP = 5e-8;
scf.diisStart = 2;
scf.diisMax = 6;
scf.damping = 0.05;      % mild damping before DIIS becomes active
scf.verbose = false;
scf.useWarmStart = true; % objective-stage warm start; final strict solve disables it
scf.warmStartRMS = 0.35;

% Printing / checkpoint controls.
printEveryObjective = false;
objectivePrintEvery = 100;
saveCheckpoints = false;
checkpointName = 'JH_BeGround_HF_RHF_SCF_DIIS_checkpoint.mat';
makePlot = false;
saveCompatibilityCopy = true;
finalSaveName = 'JH_BeGround_HF_Q9_RHF_SCF_DIIS_v3_noDIISwarning.mat';
finalSaveNameCompat = 'JH_BeGround_HF_Q9_orbitals.mat';

optsPrelude = optimset('Display','off', 'MaxFunEvals',preludeMaxFunEvals, ...
    'MaxIter',preludeMaxIter, 'TolX',3e-8, 'TolFun',3e-10);
optsBlock = optimset('Display','off', 'MaxFunEvals',blockMaxFunEvals, ...
    'MaxIter',blockMaxIter, 'TolX',2e-8, 'TolFun',2e-10);
optsPolish = optimset('Display','off', 'MaxFunEvals',finalPolishFunEvals, ...
    'MaxIter',finalPolishIter, 'TolX',1e-8, 'TolFun',1e-11);

fprintf('\n============================================================\n');
fprintf('Be ground-state HF: RHF-SCF-DIIS inner solver + staged sigma optimization\n');
fprintf('System: Be, closed shell, Z=%d, Nelec=%d, nocc=%d\n', Z, Nelec, nocc);
fprintf('Primitive basis: QperShell=%d core + %d valence = %d spatial AOs\n', QperShell, QperShell, nBasis);
fprintf('Downstream compatibility: saved hf.orbitals(1:4).terms are ordinary contracted Gaussians; each has %d terms.\n', nBasis);
fprintf('SCF: maxIter=%d, DIIS start=%d, DIIS max=%d, tolE=%.1e, tolP=%.1e\n', ...
    scf.maxIter, scf.diisStart, scf.diisMax, scf.tolE, scf.tolP);
fprintf('J.H. HF diagnostic ~= %.8f hartree\n', E_HF_JH_approx);
fprintf('============================================================\n\n');

%% ========================================================================
% 1. Starts and objective
% ========================================================================
startLibrary = make_sigma_starts_be_scf(QperShell);
sigmaStartIDs = sigmaStartIDs(:).';
if max(sigmaStartIDs) > numel(startLibrary)
    error('Requested sigmaStartIDs exceed available startLibrary length=%d.', numel(startLibrary));
end

best = empty_best_scf();
objective = @(x) be_rhf_scf_objective(x, QperShell, Z, nocc, center, scf, ...
    sigmaLower, sigmaUpper, minEigSObjective, condSMaxObjective, energyFloorObjective, ...
    badPenalty, printEveryObjective, objectivePrintEvery);

%% ========================================================================
% 2. Staged sigma optimization
% ========================================================================
for is0 = 1:numel(sigmaStartIDs)
    sid = sigmaStartIDs(is0);
    sig0 = startLibrary{sid}(:);
    if numel(sig0) ~= nBasis
        error('Start %d has length %d, expected %d.', sid, numel(sig0), nBasis);
    end
    logs = log(sig0);

    [Einit, solInit, okInit] = be_rhf_scf_evaluate(logs, QperShell, Z, nocc, center, scf, ...
        sigmaLower, sigmaUpper, minEigSObjective, condSMaxObjective, energyFloorObjective);
    fprintf('\nStart %d/%d, library id %d: E=%.15f | ok=%d | SCF it=%d | minEigS=%.3e condS=%.3e\n', ...
        is0, numel(sigmaStartIDs), sid, Einit, okInit, solInit.scfIter, solInit.minEigS, solInit.condS);
    if okInit && Einit < best.E
        best = update_best_scf(best, Einit, logs, solInit, sid, -1, struct('iterations',0,'funcCount',1,'message','initial'));
        maybe_save_checkpoint_scf(saveCheckpoints, checkpointName, best, sid, -1, Z, QperShell, nocc, scf);
    end

    if usePrelude
        fprintf('\n----- Stage A: fminsearch full-%dD sigma prelude -----\n', nBasis);
        t0 = tic;
        [logsA,~,exitflag,output] = fminsearch(objective, logs, optsPrelude);
        [EA, solA, okA] = be_rhf_scf_evaluate(logsA, QperShell, Z, nocc, center, scf, ...
            sigmaLower, sigmaUpper, minEigSObjective, condSMaxObjective, energyFloorObjective);
        fprintf('Stage A done: E=%.15f | ok=%d | diff J.H.=%.3e | SCF it=%d | time=%.1fs | it=%d eval=%d\n', ...
            EA, okA, EA-E_HF_JH_approx, solA.scfIter, toc(t0), output.iterations, output.funcCount);
        if okA && EA < best.E
            best = update_best_scf(best, EA, logsA, solA, sid, 0, output, exitflag);
        end
        maybe_save_checkpoint_scf(saveCheckpoints, checkpointName, best, sid, 0, Z, QperShell, nocc, scf);
        logs = best.logsigma;
    end

    if useBlockRefinement
        for sw = 1:blockSweeps
            fprintf('\n----- Stage B: block sweep %d/%d, current best E=%.15f -----\n', sw, blockSweeps, best.E);
            blockList = {QperShell+1:nBasis, 1:QperShell};
            blockNames = {'valence-sigma','core-sigma'};
            for bb = 1:numel(blockList)
                ids = blockList{bb};
                z0 = logs(ids);
                objBlock = @(z) block_objective_scf(z, logs, ids, objective);
                fprintf('\n----- fminsearch block %s (%d vars) -----\n', blockNames{bb}, numel(ids));
                t0 = tic;
                [z,~,exitflag,output] = fminsearch(objBlock, z0, optsBlock);
                trial = logs; trial(ids) = z(:);
                [Et, solT, okT] = be_rhf_scf_evaluate(trial, QperShell, Z, nocc, center, scf, ...
                    sigmaLower, sigmaUpper, minEigSObjective, condSMaxObjective, energyFloorObjective);
                fprintf('block %-13s done: E=%.15f | ok=%d | gain=%.3e | diff=%.3e | SCF it=%d | time=%.1fs | it=%d eval=%d\n', ...
                    blockNames{bb}, Et, okT, best.E-Et, Et-E_HF_JH_approx, solT.scfIter, toc(t0), output.iterations, output.funcCount);
                if okT && Et < best.E
                    logs = trial;
                    best = update_best_scf(best, Et, logs, solT, sid, sw*10+bb, output, exitflag);
                else
                    logs = best.logsigma;
                end
                maybe_save_checkpoint_scf(saveCheckpoints, checkpointName, best, sid, sw*10+bb, Z, QperShell, nocc, scf);
            end
        end
    end

    if useFinalPolish
        fprintf('\n----- Stage C: fminsearch final full-%dD polish -----\n', nBasis);
        logs = best.logsigma;
        t0 = tic;
        [logsC,~,exitflag,output] = fminsearch(objective, logs, optsPolish);
        [EC, solC, okC] = be_rhf_scf_evaluate(logsC, QperShell, Z, nocc, center, scf, ...
            sigmaLower, sigmaUpper, minEigSObjective, condSMaxObjective, energyFloorObjective);
        fprintf('Stage C done: E=%.15f | ok=%d | gain=%.3e | diff=%.3e | SCF it=%d | time=%.1fs | it=%d eval=%d\n', ...
            EC, okC, best.E-EC, EC-E_HF_JH_approx, solC.scfIter, toc(t0), output.iterations, output.funcCount);
        if okC && EC < best.E
            best = update_best_scf(best, EC, logsC, solC, sid, 900, output, exitflag);
        end
        maybe_save_checkpoint_scf(saveCheckpoints, checkpointName, best, sid, 900, Z, QperShell, nocc, scf);
    end

    fprintf('\nCurrent best after start id %d: E=%.15f | diff J.H.=%.3e | minEigS=%.3e condS=%.3e\n', ...
        sid, best.E, best.E-E_HF_JH_approx, best.sol.minEigS, best.sol.condS);
    if targetTolHF > 0 && best.E <= E_HF_JH_approx + targetTolHF
        fprintf('Target tolerance reached; stopping remaining starts.\n');
        break;
    end
end

if isempty(best.logsigma)
    error('No valid Be RHF-SCF solution was found.');
end

%% ========================================================================
% 3. Final strict reconstruction and save
% ========================================================================
scfFinal = scf;
scfFinal.useWarmStart = false;
if isfield(scfFinal,'initialP'), scfFinal = rmfield(scfFinal,'initialP'); end
scfFinal.maxIter = max(120, scf.maxIter);
scfFinal.tolE = min(5e-13, scf.tolE);
scfFinal.tolP = min(5e-11, scf.tolP);
[Efinal, sol, okFinal] = be_rhf_scf_evaluate(best.logsigma, QperShell, Z, nocc, center, scfFinal, ...
    sigmaLower, sigmaUpper, minEigSObjective, condSMaxObjective, energyFloorObjective);
if ~okFinal
    warning('Final strict SCF did not pass objective guards, but will still be saved for inspection.');
end
orbitals = make_spin_orbitals_from_scf_solution(sol, center);

fprintf('\n==================== FINAL Be RHF-SCF HF REFERENCE ====================\n');
fprintf('E_HF_Q9_Be_RHF_SCF = %.15f hartree\n', Efinal);
fprintf('J.H. HF diagnostic  = %.15f hartree | diff = %.6e\n', E_HF_JH_approx, Efinal-E_HF_JH_approx);
fprintf('HF limit diagnostic = %.15f hartree | diff = %.6e\n', E_HF_limit_ref, Efinal-E_HF_limit_ref);
fprintf('exact diagnostic    = %.15f hartree | HF-exact gap = %.6e\n', E_exact_ref, Efinal-E_exact_ref);
fprintf('SCF it=%d | dE=%.3e | dP=%.3e | minEigS=%.3e | condS=%.3e\n', ...
    sol.scfIter, sol.dE, sol.dP, sol.minEigS, sol.condS);
fprintf('E_one=%.15f | E_coul_exchange=%.15f\n', sol.E_one, sol.E_two);

for p = 1:4
    fprintf('\nOrbital %d (%s), terms=%d:\n', p, orbitals(p).spinLabel, numel(orbitals(p).terms));
    fprintf(' q          coeff                  sigma\n');
    for q = 1:numel(orbitals(p).terms)
        sig = 1/sqrt(orbitals(p).terms(q).alpha);
        fprintf('%2d   %+.16e   %.16e\n', q, orbitals(p).terms(q).coef, sig);
    end
end

hf = struct();
hf.system = 'Be_ground';
hf.description = 'Fast closed-shell Be RHF-SCF-DIIS HF reference; downstream-compatible contracted Gaussian orbitals';
hf.atom = 'Be'; hf.Z = Z; hf.N = Nelec; hf.N_up = 2; hf.N_down = 2; hf.MS = 0; hf.Ms = 0;
hf.Q = numel(orbitals(1).terms);              % actual terms per saved orbital: 18 by default
hf.QperShell = QperShell;                     % nonlinear sigma groups: 9 core-like + 9 valence-like
hf.nPrimitiveUnion = nBasis;
hf.center = center;
hf.method = 'Be_closed_shell_RHF_SCF_DIIS_outer_sigma_v2_fast_compatible';
hf.compatibility = struct('usableByFrameCoarsening',true, ...
    'orbitalsHaveTerms',true, 'termsCanBeMoreThanQperShell',true, ...
    'orbitalsOneAndThreeIdenticalSpatial',true, 'orbitalsTwoAndFourIdenticalSpatial',true);
hf.E = Efinal;
hf.E_HF_JH_approx = E_HF_JH_approx;
hf.E_HF_limit_ref = E_HF_limit_ref;
hf.E_exact_ref = E_exact_ref;
hf.correlation_gap_approx = Efinal - E_exact_ref;
hf.E_components = struct('E_one',sol.E_one,'E_two',sol.E_two);
hf.orbitals = orbitals;
hf.best = best;
hf.scfSolution = sol;
hf.params = struct('E_HF_JH_approx',E_HF_JH_approx,'E_HF_limit_ref',E_HF_limit_ref,'E_exact_ref',E_exact_ref, ...
    'QperShell',QperShell,'nPrimitiveUnion',nBasis,'sigmaLower',sigmaLower,'sigmaUpper',sigmaUpper, ...
    'minEigSObjective',minEigSObjective,'condSMaxObjective',condSMaxObjective,'scf',scfFinal, ...
    'preludeMaxIter',preludeMaxIter,'blockSweeps',blockSweeps,'blockMaxIter',blockMaxIter,'finalPolishIter',finalPolishIter);
hf.savedAt = datestr(now);

save(finalSaveName,'hf','-v7.3');
fprintf('\nSaved: %s\n', finalSaveName);
if saveCompatibilityCopy
    save(finalSaveNameCompat,'hf','-v7.3');
    fprintf('Saved compatibility copy: %s\n', finalSaveNameCompat);
else
    fprintf('Compatibility copy disabled.\n');
end

if makePlot
    try
        xgrid = linspace(-12,12,2500).';
        figure('Color','w'); hold on;
        for p = 1:2
            y = eval_orbital_on_x_axis(orbitals(p), xgrid);
            plot(xgrid, y, 'LineWidth', 1.2);
        end
        xlabel('x_1 [bohr]'); ylabel('\psi(x_1,0,0)');
        title(sprintf('Be RHF-SCF orbitals, E=%.10f', Efinal));
        legend('spatial core','spatial valence','Location','best'); grid on;
    catch ME
        warning('Plot skipped: %s', ME.message);
    end
end

%% ========================================================================
% Local functions
% ========================================================================
function starts = make_sigma_starts_be_scf(Q)
    starts = {};
    core1 = logspace(log10(0.0048), log10(0.82), Q).';
    val1  = logspace(log10(0.16),   log10(4.8), Q).';
    core2 = logspace(log10(0.0060), log10(1.05), Q).';
    val2  = logspace(log10(0.11),   log10(7.0), Q).';
    core3 = logspace(log10(0.0035), log10(0.70), Q).';
    val3  = logspace(log10(0.08),   log10(9.0), Q).';
    starts{end+1} = [core1; val1];
    starts{end+1} = [core2; val2];
    starts{end+1} = [core3; val3];
    if exist('JH_LiGround_HF_Q9_orbitals.mat','file')
        try
            S = load('JH_LiGround_HF_Q9_orbitals.mat','hf');
            sig1 = arrayfun(@(t) 1/sqrt(t.alpha), S.hf.orbitals(1).terms(:));
            sig2 = arrayfun(@(t) 1/sqrt(t.alpha), S.hf.orbitals(2).terms(:));
            if numel(sig1)==Q && numel(sig2)==Q
                starts{end+1} = [sig1(:)*0.70; sig2(:)*0.65];
                starts{end+1} = [sig1(:)*0.78; sig2(:)*0.58];
                starts{end+1} = [sig1(:)*0.65; sig2(:)*0.80];
            end
        catch
        end
    end
    base = starts{1};
    for k=1:3
        starts{end+1} = base .* exp(0.06*randn(2*Q,1));
    end
end

function E = block_objective_scf(z, logs, ids, obj)
    trial = logs; trial(ids) = z(:); E = obj(trial);
end

function Eobj = be_rhf_scf_objective(logsigma, Q, Z, nocc, center, scf, sigmaLower, sigmaUpper, minEigS, condSMax, energyFloor, badPenalty, doPrint, printEvery)
    persistent evalCount lastLogSigma lastP lastGoodE;
    if isempty(evalCount), evalCount = 0; end
    evalCount = evalCount + 1;

    scfObj = scf;
    warmUsed = false;
    if isfield(scf,'useWarmStart') && scf.useWarmStart && ~isempty(lastP) && ~isempty(lastLogSigma)
        dx = logsigma(:) - lastLogSigma(:);
        rmsStep = norm(dx) / sqrt(numel(dx));
        if rmsStep <= scf.warmStartRMS && all(size(lastP)==[2*Q,2*Q])
            scfObj.initialP = lastP;
            warmUsed = true;
        end
    end

    [E, sol, ok, penalty] = be_rhf_scf_evaluate_with_penalty(logsigma, Q, Z, nocc, center, scfObj, sigmaLower, sigmaUpper, minEigS, condSMax, energyFloor, badPenalty);
    if ok
        Eobj = E + penalty;
        lastLogSigma = logsigma(:);
        lastP = sol.P;
        lastGoodE = E;
    else
        Eobj = badPenalty + penalty;
    end

    if doPrint && printEvery > 0 && mod(evalCount, printEvery)==0
        if ok
            fprintf('  obj eval %6d | E=%.12f | SCF=%2d | warm=%d | minEigS=%.2e condS=%.2e\n', evalCount, E, sol.scfIter, warmUsed, sol.minEigS, sol.condS);
        else
            if isempty(lastGoodE), lastGoodE = NaN; end
            fprintf('  obj eval %6d | Eobj=%.3e | rawE=%.12f | lastGood=%.12f | SCF=%2d | warm=%d | GUARD\n', evalCount, Eobj, E, lastGoodE, sol.scfIter, warmUsed);
        end
    end
end

function [E, sol, ok] = be_rhf_scf_evaluate(logsigma, Q, Z, nocc, center, scf, sigmaLower, sigmaUpper, minEigS, condSMax, energyFloor)
    [E, sol, ok] = be_rhf_scf_evaluate_with_penalty(logsigma, Q, Z, nocc, center, scf, sigmaLower, sigmaUpper, minEigS, condSMax, energyFloor, 1e12);
end

function [E, sol, ok, penalty] = be_rhf_scf_evaluate_with_penalty(logsigma, Q, Z, nocc, center, scf, sigmaLower, sigmaUpper, minEigS, condSMax, energyFloor, badPenalty)
    %#ok<INUSD> center is included for interface consistency and saved orbitals.
    penalty = 0;
    logsigma = logsigma(:);
    if numel(logsigma) ~= 2*Q || any(~isfinite(logsigma))
        E = badPenalty; sol = empty_scf_solution(); ok = false; return;
    end
    low = log(sigmaLower); high = log(sigmaUpper);
    if any(logsigma < low), penalty = penalty + 1e6*sum((logsigma(logsigma<low)-low).^2); end
    if any(logsigma > high), penalty = penalty + 1e6*sum((logsigma(logsigma>high)-high).^2); end
    logsigma = min(max(logsigma,low),high);
    sigmas = exp(logsigma);
    alpha = 1 ./ sigmas.^2;
    try
        ints = build_centered_primitive_integrals_fast(alpha, Z);
        [E, sol] = rhf_scf_diis(ints.S, ints.H, ints.Jmat, ints.Kmat, nocc, scf);
        sol.sigmas = sigmas;
        sol.alpha = alpha;
    catch ME
        E = badPenalty; sol = empty_scf_solution(); sol.failMessage = ME.message; ok = false; return;
    end
    ok = isfinite(E) && sol.converged && sol.minEigS >= minEigS && sol.condS <= condSMax && E >= energyFloor;
end

function ints = build_centered_primitive_integrals_fast(alpha, Z)
    % All primitives are normalized centered s Gaussians exp(-alpha*r^2/2).
    % This routine returns pre-flattened Coulomb/exchange matrices for fast
    % repeated J/K builds during SCF.  Avoid permuting ERI inside every SCF step.
    alpha = alpha(:); n = numel(alpha);
    [A,B] = ndgrid(alpha, alpha);
    p = A+B;
    Nprod = (A.*B).^(3/4) / (pi^(3/2));
    S = Nprod .* (2*pi./p).^(3/2);
    T = 0.5 .* A .* B .* (3./p) .* S;
    Ven = -Z .* Nprod .* (4*pi./p);
    H = T + Ven;

    [A4,B4,C4,D4] = ndgrid(alpha, alpha, alpha, alpha);
    p4 = A4+B4; q4 = C4+D4;
    N4 = (A4.*B4.*C4.*D4).^(3/4)/(pi^3);
    ERI4 = N4 .* (8*sqrt(2)*pi^(5/2)) ./ (p4.*q4.*sqrt(p4+q4));
    Jmat = reshape(ERI4, n*n, n*n);
    Kmat = reshape(permute(ERI4, [1 3 2 4]), n*n, n*n);
    ints = struct('S',S,'H',H,'Jmat',Jmat,'Kmat',Kmat);
end

function [E, sol] = rhf_scf_diis(S, h, Jmat, Kmat, nocc, opts)
    n = size(S,1);
    S = 0.5*(S+S'); h = 0.5*(h+h');
    [U,dmat] = eig(S); d = real(diag(dmat));
    [d,ord] = sort(d,'descend'); U = U(:,ord);
    minEigS_local = min(d);
    condS_local = max(d) / max(minEigS_local, realmin);
    keep = d > max(d)*1e-12;
    if nnz(keep) < nocc, error('S-rank is smaller than number of occupied orbitals.'); end
    X = U(:,keep) * diag(1./sqrt(d(keep)));

    % Initial density.  During fminsearch the previous converged density is
    % a good predictor for nearby sigma points; otherwise use core Hamiltonian.
    if isfield(opts,'initialP') && isequal(size(opts.initialP), [n,n]) && all(isfinite(opts.initialP(:)))
        P = 0.5*(opts.initialP + opts.initialP.');
        Cnew = []; CoccNew = []; epsOcc = [];
    else
        Hp0 = X'*h*X; Hp0 = 0.5*(Hp0+Hp0');
        [Cp,eps] = eig(Hp0);
        [~,ord] = sort(real(diag(eps))); C = X*Cp(:,ord);
        Cocc = C(:,1:nocc); P = Cocc*Cocc.'; P = 0.5*(P+P');
        Cnew = C; CoccNew = Cocc; epsOcc = real(diag(eps));
    end

    Eold = inf; Flist = {}; Rlist = {};
    sol = empty_scf_solution();
    for it = 1:opts.maxIter
        G = build_rhf_G_fast(P, Jmat, Kmat);
        F = h + G; F = 0.5*(F+F');
        R = F*P*S - S*P*F;
        Flist{end+1} = F; Rlist{end+1} = R; %#ok<AGROW>
        if numel(Flist) > opts.diisMax
            Flist = Flist(end-opts.diisMax+1:end); Rlist = Rlist(end-opts.diisMax+1:end);
        end
        if it >= opts.diisStart && numel(Flist) >= 2
            Fuse = diis_extrapolate(Flist, Rlist);
        else
            Fuse = F;
        end
        Fp = X'*Fuse*X; Fp = 0.5*(Fp+Fp');
        [Cp,eps] = eig(Fp); [epsOcc,ord] = sort(real(diag(eps))); %#ok<ASGLU>
        Cnew = X*Cp(:,ord);
        CoccNew = Cnew(:,1:nocc);
        Pnew = CoccNew*CoccNew.'; Pnew = 0.5*(Pnew+Pnew');
        if it < opts.diisStart && opts.damping > 0
            Pnew = (1-opts.damping)*Pnew + opts.damping*P;
        end
        Gnew = build_rhf_G_fast(Pnew, Jmat, Kmat);
        E = 2*trace(Pnew*h) + trace(Pnew*Gnew);
        dE = E - Eold;
        dP = norm(Pnew-P,'fro')/max(1,norm(P,'fro'));
        if opts.verbose
            fprintf('    SCF %3d E=%.15f dE=%.3e dP=%.3e\n', it, E, dE, dP);
        end
        P = Pnew; Eold = E;
        if it > 1 && abs(dE) < opts.tolE*max(1,abs(E)) && dP < opts.tolP
            sol.converged = true; sol.scfIter = it; sol.C = Cnew; sol.Cocc = CoccNew;
            sol.P = P; sol.eps = epsOcc; sol.E = E; sol.dE = dE; sol.dP = dP;
            sol.E_one = 2*trace(P*h); sol.E_two = trace(P*Gnew);
            sol.minEigS = minEigS_local; sol.condS = condS_local; return;
        end
    end
    sol.converged = false; sol.scfIter = opts.maxIter; sol.C = Cnew; sol.Cocc = CoccNew;
    sol.P = P; sol.eps = epsOcc; sol.E = E; sol.dE = dE; sol.dP = dP;
    sol.E_one = 2*trace(P*h); sol.E_two = trace(P*Gnew);
    sol.minEigS = minEigS_local; sol.condS = condS_local;
end

function G = build_rhf_G_fast(P, Jmat, Kmat)
    n = size(P,1);
    Pv = P(:);
    J = reshape(Jmat * Pv, n, n);
    K = reshape(Kmat * Pv, n, n);
    G = 2*J - K;
    G = 0.5*(G+G');
end

function F = diis_extrapolate(Flist, Rlist)
    % Robust DIIS extrapolation.
    %
    % In the outer fminsearch, nearby sigma values often produce almost the
    % same SCF residual history.  The usual Pulay augmented matrix can then be
    % nearly singular; using B\rhs directly triggers MATLAB's
    % "matrix is close to singular" warning.  DIIS is only an accelerator, so
    % the safe behavior is: prune old/collinear residuals, add tiny ridge
    % regularization, and fall back to the newest Fock matrix if the DIIS
    % coefficients are still unreliable.

    maxCoefL1 = 50;       % reject wildly amplified extrapolations
    rcondMin  = 1e-12;    % stricter than MATLAB's warning threshold
    ridgeRel  = 1e-13;

    % Keep only finite residuals/Fock matrices.
    good = true(1,numel(Flist));
    for k = 1:numel(Flist)
        good(k) = all(isfinite(Flist{k}(:))) && all(isfinite(Rlist{k}(:)));
    end
    Flist = Flist(good); Rlist = Rlist(good);
    if numel(Flist) < 2
        F = Flist{end}; F = 0.5*(F+F'); return;
    end

    % Prune from the oldest side until the augmented system is safe.
    while numel(Flist) >= 2
        m = numel(Flist);
        Berr = zeros(m,m);
        for i = 1:m
            ri = Rlist{i};
            for j = i:m
                rj = Rlist{j};
                val = sum(ri(:).*rj(:));
                Berr(i,j) = val; Berr(j,i) = val;
            end
        end
        Berr = 0.5*(Berr+Berr');
        scale = max(1, max(abs(Berr(:))));

        B = zeros(m+1,m+1);
        B(1:m,1:m) = Berr + ridgeRel*scale*eye(m);
        B(end,1:m) = -1; B(1:m,end) = -1;
        rhs = zeros(m+1,1); rhs(end) = -1;

        rc = rcond(B);
        if isfinite(rc) && rc >= rcondMin
            coefFull = B \ rhs;
            coef = coefFull(1:m);
            if all(isfinite(coef)) && sum(abs(coef)) <= maxCoefL1
                F = zeros(size(Flist{1}));
                for i = 1:m
                    F = F + coef(i)*Flist{i};
                end
                F = 0.5*(F+F');
                return;
            end
        end

        % If unsafe, discard the oldest residual/Fock pair and retry.
        Flist = Flist(2:end);
        Rlist = Rlist(2:end);
    end

    % Last resort: no DIIS this iteration.  This is stable and warning-free.
    F = Flist{end};
    F = 0.5*(F+F');
end

function orbitals = make_spin_orbitals_from_scf_solution(sol, center)
    n = numel(sol.alpha);
    Cocc = sol.Cocc;
    labels = {'alpha-core','alpha-valence','beta-core','beta-valence'};
    spins  = {'alpha','alpha','beta','beta'};
    roles  = {'core','valence','core','valence'};
    spatial = {Cocc(:,1), Cocc(:,2), Cocc(:,1), Cocc(:,2)};
    orbitals = repmat(struct('terms',[],'spin','','spinLabel','','orbitalType','s','role',''),4,1);
    for p = 1:4
        terms = repmat(struct('coef',0,'alpha',0,'center',center), n, 1);
        cp = spatial{p};
        % Stable sign convention: largest coefficient positive for core/valence.
        [~,imax] = max(abs(cp));
        if cp(imax) < 0, cp = -cp; end
        for q = 1:n
            terms(q).coef = cp(q);
            terms(q).alpha = sol.alpha(q);
            terms(q).center = center;
        end
        orbitals(p).terms = terms;
        orbitals(p).spin = spins{p};
        orbitals(p).spinLabel = labels{p};
        orbitals(p).orbitalType = 's';
        orbitals(p).role = roles{p};
    end
end

function best = empty_best_scf()
    best = struct('E',inf,'logsigma',[],'sol',[],'startID',[],'stageID',[],'output',[],'exitflag',[]);
end

function best = update_best_scf(best,E,logsigma,sol,sid,stageID,output,exitflag)
    if nargin<8, exitflag=[]; end
    best.E = E; best.logsigma = logsigma(:); best.sol = sol; best.startID = sid; best.stageID = stageID;
    best.output = output; best.exitflag = exitflag;
end

function sol = empty_scf_solution()
    sol = struct('converged',false,'scfIter',0,'C',[],'Cocc',[],'P',[],'eps',[], ...
        'E',inf,'dE',inf,'dP',inf,'E_one',NaN,'E_two',NaN,'sigmas',[],'alpha',[], ...
        'minEigS',NaN,'condS',Inf,'failMessage','');
end

function maybe_save_checkpoint_scf(doSave, filename, best, sid, stageID, Z, QperShell, nocc, scf)
    if ~doSave, return; end
    chk = struct('best',best,'sid',sid,'stageID',stageID,'Z',Z,'QperShell',QperShell,'nocc',nocc,'scf',scf,'savedAt',datestr(now));
    save(filename,'chk','-v7.3');
end

function y = eval_orbital_on_x_axis(orb,x)
    y = zeros(size(x));
    for k=1:numel(orb.terms)
        a = orb.terms(k).alpha; c = orb.terms(k).coef;
        N = (a/pi)^(3/4);
        y = y + c*N*exp(-0.5*a*x.^2);
    end
end
