%% JH_HeTriplet_part1_HF_Q9_v2_nonorthUHF_singleStart_earlySave.m
% Robust rank-1 HF reference for J.H. lowest triplet state of He, (3S)He.
%
% Main correction relative to the first triplet HF draft:
%   Each spin-up one-particle function psi_1 and psi_2 has its OWN Q=9
%   s-type Gaussian contraction.  They are optimized as non-orthogonal
%   unrestricted HF orbitals.  The determinant is normalized by the Gram
%   determinant; the orbitals themselves are NOT forced to be orthogonal.
%
% Output:
%   JH_HeTriplet_HF_Q9_orbitals.mat
%
% This output keeps the same hf.orbitals(1:2).terms interface used by the
% triplet adaptive code.

clear; clc; close all;
format long e;

%% Parameters
Z = 2;
Q = 9;
center = [0,0,0];
rng(23);

E_HF_JH_approx = -2.17424;          % J.H. Table 5.8 approximate HF for (3S)He
E_exact_ref    = -2.175229378236791; % rounded to -2.17523 in J.H. Table 5.8

% Outer optimization controls.  This is 18-dimensional, but for fixed
% exponents the coefficients are solved by exact alternating generalized
% Rayleigh-quotient minimization.
optsOuter = optimset('Display','final', ...
                     'MaxFunEvals', 1.2e5, ...
                     'MaxIter', 1e4, ...
                     'TolX', 5e-11, ...
                     'TolFun', 5e-13);

maxOuterRestart = 1;

fprintf('\n============================================================\n');
fprintf('J.H. (3S)He rank-1 non-orthogonal UHF reference, Q=%d per orbital\n', Q);
fprintf('Target J.H. approximate HF energy ~= %.8f hartree\n', E_HF_JH_approx);
fprintf('Ansatz: normalized det[psi_1, psi_2], each psi_p has independent Q=9 s-Gaussians.\n');
fprintf('============================================================\n\n');

sigmaStarts = make_sigma_starts(Q);

best = struct('E',inf,'logsigma',[],'orbitals',[],'comps',[],'exitflag',[],'output',[]);

% Early-stop and checkpoint controls.
% targetTolHF=1e-6 is intentionally matched to the precision actually needed
% for Table 5.8-level agreement: -2.17424.  Once this is reached, the code
% stops before extra restarts / extra starts.
targetTolHF = 1e-6;
saveAfterEachSolve = true;
checkpointName = 'JH_HeTriplet_HF_Q9_orbitals_checkpoint.mat';
stopAllStarts = false;

for is = 1:numel(sigmaStarts)
    logs0 = log(sigmaStarts{is}(:));
    if numel(logs0) ~= 2*Q
        error('Internal start %d has wrong length.', is);
    end

    [E0, ~, comps0] = triplet_nonorth_energy_outer_logsigma(logs0, Q, Z, center);
    fprintf('\nStart %d/%d: initial E = %.15f | detGram=%.3e | altIter=%d\n', ...
        is, numel(sigmaStarts), E0, comps0.detGram, comps0.altIter);

    logs = logs0;

    % ------------------------------------------------------------
    % First fminsearch from this start.
    % ------------------------------------------------------------
    [logs, E, exitflag, output] = fminsearch( ...
        @(xx) triplet_nonorth_energy_outer_logsigma(xx, Q, Z, center), logs, optsOuter);

    [Echeck, orbitals, comps] = triplet_nonorth_energy_outer_logsigma(logs, Q, Z, center);
    fprintf('Finished start %d primary solve: E=%.15f | diff J.H.=%.3e | detGram=%.3e | s12=%.6e | altIt=%d\n', ...
        is, Echeck, Echeck - E_HF_JH_approx, comps.detGram, comps.s12, comps.altIter);
    fprintf('  h11=%.15f h22=%.15f h12=%.15f J=%.15f K=%.15f\n', ...
        comps.h11, comps.h22, comps.h12, comps.J12, comps.K12);

    if Echeck < best.E
        best.E = Echeck;
        best.logsigma = logs;
        best.orbitals = orbitals;
        best.comps = comps;
        best.exitflag = exitflag;
        best.output = output;
    end

    if saveAfterEachSolve
        save_hf_checkpoint(checkpointName, best, logs, Echeck, orbitals, comps, ...
            is, 0, exitflag, output, Z, Q, center, E_HF_JH_approx, E_exact_ref, targetTolHF);
        fprintf('  checkpoint saved after primary solve: %s\n', checkpointName);
    end

    if abs(Echeck - E_HF_JH_approx) < targetTolHF
        fprintf('Reached J.H. HF target accuracy after primary solve; skip restart and stop further starts.\n');
        stopAllStarts = true;
    end

    % ------------------------------------------------------------
    % Optional restart(s).  With maxOuterRestart=1, this is at most one.
    % Crucially, it is skipped if the primary solve has already reached the
    % requested Table 5.8 precision.
    % ------------------------------------------------------------
    if ~stopAllStarts
        for r = 1:maxOuterRestart
            fprintf('  restart %d from E = %.15f\n', r, E);
            [logs, E, exitflag, output] = fminsearch( ...
                @(xx) triplet_nonorth_energy_outer_logsigma(xx, Q, Z, center), logs, optsOuter);

            [Echeck, orbitals, comps] = triplet_nonorth_energy_outer_logsigma(logs, Q, Z, center);
            fprintf('Finished start %d restart %d: E=%.15f | diff J.H.=%.3e | detGram=%.3e | s12=%.6e | altIt=%d\n', ...
                is, r, Echeck, Echeck - E_HF_JH_approx, comps.detGram, comps.s12, comps.altIter);
            fprintf('  h11=%.15f h22=%.15f h12=%.15f J=%.15f K=%.15f\n', ...
                comps.h11, comps.h22, comps.h12, comps.J12, comps.K12);

            if Echeck < best.E
                best.E = Echeck;
                best.logsigma = logs;
                best.orbitals = orbitals;
                best.comps = comps;
                best.exitflag = exitflag;
                best.output = output;
            end

            if saveAfterEachSolve
                save_hf_checkpoint(checkpointName, best, logs, Echeck, orbitals, comps, ...
                    is, r, exitflag, output, Z, Q, center, E_HF_JH_approx, E_exact_ref, targetTolHF);
                fprintf('  checkpoint saved after restart %d: %s\n', r, checkpointName);
            end

            if abs(Echeck - E_HF_JH_approx) < targetTolHF
                fprintf('Reached J.H. HF target accuracy after restart %d; stop further starts.\n', r);
                stopAllStarts = true;
                break;
            end
        end
    end

    fprintf('Current best after start %d: E=%.15f | diff J.H.=%.3e\n\n', ...
        is, best.E, best.E - E_HF_JH_approx);

    if stopAllStarts
        break;
    end
end

if isempty(best.logsigma)
    error('No valid HF solution was found. Check starting sigmas and coefficient solver.');
end

% Sort primitive sigmas within each orbital only for storage/readability.
[~, orbitals, ~] = triplet_nonorth_energy_outer_logsigma(best.logsigma, Q, Z, center);
orbitals = sort_orbitals_by_sigma(orbitals);
[Efinal, orbitals, comps] = energy_from_stored_orbitals(orbitals, Z, center);

fprintf('\n==================== FINAL (3S)He HF REFERENCE v2 ====================\n');
fprintf('E_HF_Q9_triplet = %.15f hartree\n', Efinal);
fprintf('J.H. approx HF  = %.15f hartree\n', E_HF_JH_approx);
fprintf('diff            = %.6e hartree\n', Efinal - E_HF_JH_approx);
fprintf('exact ref       = %.15f hartree; correlation gap approx %.6e\n', E_exact_ref, Efinal - E_exact_ref);
fprintf('n1,n2,s12       = %.15f %.15f %.15e\n', comps.n1, comps.n2, comps.s12);
fprintf('detGram         = %.15e\n', comps.detGram);
fprintf('h11,h22,h12     = %.15f %.15f %.15f\n', comps.h11, comps.h22, comps.h12);
fprintf('J12,K12         = %.15f %.15f\n', comps.J12, comps.K12);

for p = 1:2
    fprintf('\nOrbital %d independent Q=%d primitive contraction:\n', p, Q);
    fprintf(' q          coeff                  sigma\n');
    for q = 1:Q
        fprintf('%2d   %+ .16e   %.16e\n', q, orbitals(p).coeff(q), orbitals(p).sigma(q));
    end
end

hf = struct();
hf.E = Efinal;
hf.E_HF_JH_approx = E_HF_JH_approx;
hf.E_exact_ref = E_exact_ref;
hf.Z = Z;
hf.Q = Q;
hf.center = center;
hf.orbitals = orbitals;
hf.comps = comps;
hf.description = 'J.H. (3S)He rank-1 non-orthogonal determinant; independent Q=9 s-type contractions for psi1 and psi2';
hf.params = struct('Z',Z,'Q',Q,'center',center, ...
                   'E_HF_JH_approx',E_HF_JH_approx, ...
                   'E_exact_ref',E_exact_ref, ...
                   'coefficientSolver','alternating generalized determinant Rayleigh quotient');
hf.best = best;

saveName = 'JH_HeTriplet_HF_Q9_orbitals.mat';
save(saveName, 'hf', '-v7.3');
fprintf('\nSaved: %s\n', saveName);

% Plot along x-axis for diagnostic.
xgrid = linspace(-10,10,1600).';
y1 = eval_orbital_on_x_axis(orbitals(1), xgrid);
y2 = eval_orbital_on_x_axis(orbitals(2), xgrid);
figure('Color','w');
plot(xgrid, y1, 'LineWidth', 1.5); hold on;
plot(xgrid, y2, 'LineWidth', 1.5);
xlabel('x_1 [bohr]'); ylabel('\psi(x_1,0,0)');
title(sprintf('(3S)He non-orthogonal rank-1 HF orbitals, E=%.10f', Efinal));
legend('\psi_1^{(3S)He}','\psi_2^{(3S)He}','Location','best'); grid on;

%% ========================================================================
% Checkpoint helper
% ========================================================================
function save_hf_checkpoint(checkpointName, best, logs, Echeck, orbitals, comps, ...
    startID, restartID, exitflag, output, Z, Q, center, E_HF_JH_approx, E_exact_ref, targetTolHF)
    hf_checkpoint = struct();
    hf_checkpoint.best = best;
    hf_checkpoint.current = struct();
    hf_checkpoint.current.logsigma = logs;
    hf_checkpoint.current.E = Echeck;
    hf_checkpoint.current.orbitals = orbitals;
    hf_checkpoint.current.comps = comps;
    hf_checkpoint.current.startID = startID;
    hf_checkpoint.current.restartID = restartID;
    hf_checkpoint.current.exitflag = exitflag;
    hf_checkpoint.current.output = output;
    hf_checkpoint.params = struct('Z',Z,'Q',Q,'center',center, ...
        'E_HF_JH_approx',E_HF_JH_approx,'E_exact_ref',E_exact_ref, ...
        'targetTolHF',targetTolHF);
    hf_checkpoint.savedAt = datestr(now);
    save(checkpointName, 'hf_checkpoint', '-v7.3');
end

%% ========================================================================
% Starts
% ========================================================================
function starts = make_sigma_starts(Q)
    starts = {};

    % Single deterministic start only.  The previous version added two
    % jittered starts, so maxOuterRestart=1 still led to three independent
    % starts and one restart inside each start.  For Table 5.8-level matching
    % (-2.17424), the deterministic start is sufficient.
    singletFile = 'JH_He_HF_Q9_orbital_v2.mat';
    if exist(singletFile,'file')
        try
            S = load(singletFile,'hf');
            sig1_ref = S.hf.sigma(:);
        catch
            sig1_ref = logspace(log10(0.015), log10(2.5), Q).';
        end
    else
        sig1_ref = logspace(log10(0.015), log10(2.5), Q).';
    end

    % psi_2 needs a more diffuse/nodal 2s-like contraction.
    starts{1} = [sig1_ref(:); logspace(log10(0.060), log10(18.0), Q).'];
end

%% ========================================================================
% Outer objective: independent exponents for psi1 and psi2.
% ========================================================================
function [E, orbitals, comps] = triplet_nonorth_energy_outer_logsigma(logsigma, Q, Z, center)
    logsigma = logsigma(:);
    if numel(logsigma) ~= 2*Q || any(~isfinite(logsigma))
        E = 1e20; orbitals = []; comps = empty_comps(); return;
    end

    low = log(2e-4);
    high = log(1e2);
    penalty = 0;
    if any(logsigma < low)
        penalty = penalty + 1e5*sum((logsigma(logsigma<low)-low).^2);
    end
    if any(logsigma > high)
        penalty = penalty + 1e5*sum((logsigma(logsigma>high)-high).^2);
    end
    logsigma = min(max(logsigma, low), high);

    sigma1 = exp(logsigma(1:Q));
    sigma2 = exp(logsigma(Q+1:end));
    alpha1 = 1 ./ sigma1.^2;
    alpha2 = 1 ./ sigma2.^2;

    terms1 = make_terms_from_alpha(alpha1, center);
    terms2 = make_terms_from_alpha(alpha2, center);

    try
        ints = build_two_orbital_integrals(terms1, terms2, Z, center);
        [c1,c2,Escf,comps] = solve_coefficients_nonorth_det(ints);
    catch ME
        % Bad exponent sets can make the projected determinant metric nearly
        % singular.  Return a smooth large value rather than aborting fminsearch.
        E = 1e18 + penalty;
        orbitals = [];
        comps = empty_comps();
        comps.failMessage = ME.message;
        return;
    end

    % Penalize nearly singular determinants.  This is not part of the energy,
    % but protects Nelder-Mead from collapsed orbital pairs.
    if comps.detGram < 1e-8
        penalty = penalty + 1e3*(log(1e-8) - log(max(comps.detGram,realmin)))^2;
    end

    E = Escf + penalty;

    orbitals = repmat(struct('coeff',[],'sigma',[],'alpha',[],'center',center,'terms',[],'kind','ref'), 2, 1);
    orbitals(1).coeff = c1; orbitals(1).sigma = sigma1; orbitals(1).alpha = alpha1;
    orbitals(2).coeff = c2; orbitals(2).sigma = sigma2; orbitals(2).alpha = alpha2;
    orbitals(1).center = center; orbitals(2).center = center;
    orbitals(1).terms = contracted_terms(terms1, c1); orbitals(1).kind = 'ref';
    orbitals(2).terms = contracted_terms(terms2, c2); orbitals(2).kind = 'ref';
end

function [E, orbitals, comps] = energy_from_stored_orbitals(orbitals, Z, center)
    terms1 = make_terms_from_alpha(orbitals(1).alpha, center);
    terms2 = make_terms_from_alpha(orbitals(2).alpha, center);
    ints = build_two_orbital_integrals(terms1, terms2, Z, center);
    c1 = orbitals(1).coeff(:); c2 = orbitals(2).coeff(:);
    comps = determinant_energy_comps(ints,c1,c2);
    E = comps.E;
    orbitals(1).terms = contracted_terms(terms1,c1);
    orbitals(2).terms = contracted_terms(terms2,c2);
end

function orbitals = sort_orbitals_by_sigma(orbitals)
    for p = 1:2
        [sig,ord] = sort(orbitals(p).sigma(:),'ascend');
        orbitals(p).sigma = sig;
        orbitals(p).alpha = orbitals(p).alpha(ord);
        orbitals(p).coeff = orbitals(p).coeff(ord);
    end
end

%% ========================================================================
% Coefficient minimization for fixed exponents.
% ========================================================================
function [c1,c2,E,comps] = solve_coefficients_nonorth_det(ints)
    Q = size(ints.S11,1);
    mass_tol = 1e-12;
    maxIt = 200;
    tolE = 2e-13;

    % Initial c1: lowest one-electron state in space 1.
    [V1,D1] = eig((ints.H11+ints.H11')/2, (ints.S11+ints.S11')/2);
    [~,ord1] = sort(real(diag(D1)),'ascend');
    c1 = real(V1(:,ord1(1)));
    c1 = normalize_with_S(c1, ints.S11);

    % Initial c2: projected one-electron state in space 2, avoiding collapse
    % with c1.  This produces a nodal/diffuse 2s-like start.
    c2 = update_second_orbital_oneelectron(ints,c1,mass_tol);
    if isempty(c2)
        [V2,D2] = eig((ints.H22+ints.H22')/2, (ints.S22+ints.S22')/2);
        [~,ord2] = sort(real(diag(D2)),'ascend');
        c2 = real(V2(:,ord2(min(2,Q))));
        c2 = normalize_with_S(c2, ints.S22);
    end

    comps = determinant_energy_comps(ints,c1,c2);
    Eold = comps.E;

    for it = 1:maxIt
        c1 = update_c1_given_c2(ints,c2,mass_tol);
        c1 = normalize_with_S(c1, ints.S11);

        c2 = update_c2_given_c1(ints,c1,mass_tol);
        c2 = normalize_with_S(c2, ints.S22);

        comps = determinant_energy_comps(ints,c1,c2);
        Enew = comps.E;
        if abs(Enew - Eold) < tolE
            break;
        end
        Eold = Enew;
    end

    comps.altIter = it;
    E = comps.E;

    % Orient signs for reproducible storage.
    if sum(c1) < 0, c1 = -c1; end
    if c2(find(abs(c2)==max(abs(c2)),1,'first')) < 0, c2 = -c2; end
    comps = determinant_energy_comps(ints,c1,c2);
    comps.altIter = it;
    E = comps.E;
end

function c2 = update_second_orbital_oneelectron(ints,c1,mass_tol)
    n1 = real(c1' * ints.S11 * c1);
    h11 = real(c1' * ints.H11 * c1);
    bS = ints.S12' * c1;
    bH = ints.H12' * c1;
    A = n1*ints.H22 + h11*ints.S22 - (bS*bH' + bH*bS');
    B = n1*ints.S22 - bS*bS';
    [~,c2,ok] = solve_min_gen(A,B,mass_tol);
    if ~ok, c2 = []; end
end

function c1 = update_c1_given_c2(ints,c2,mass_tol)
    n2 = real(c2' * ints.S22 * c2);
    h22 = real(c2' * ints.H22 * c2);
    bS = ints.S12 * c2;
    bH = ints.H12 * c2;
    [Jmat,Kmat] = JK_for_space1_given_c2(ints,c2);
    A = n2*ints.H11 + h22*ints.S11 - (bS*bH' + bH*bS') + Jmat - Kmat;
    B = n2*ints.S11 - bS*bS';
    [~,c1,ok] = solve_min_gen(A,B,mass_tol);
    if ~ok, error('update_c1_given_c2 failed: projected overlap matrix singular.'); end
end

function c2 = update_c2_given_c1(ints,c1,mass_tol)
    n1 = real(c1' * ints.S11 * c1);
    h11 = real(c1' * ints.H11 * c1);
    bS = ints.S12' * c1;
    bH = ints.H12' * c1;
    [Jmat,Kmat] = JK_for_space2_given_c1(ints,c1);
    A = n1*ints.H22 + h11*ints.S22 - (bS*bH' + bH*bS') + Jmat - Kmat;
    B = n1*ints.S22 - bS*bS';
    [~,c2,ok] = solve_min_gen(A,B,mass_tol);
    if ~ok, error('update_c2_given_c1 failed: projected overlap matrix singular.'); end
end

function [E,c,ok] = solve_min_gen(A,B,mass_tol)
    A = (A+A')/2; B = (B+B')/2;
    [U,d] = eig(B,'vector');
    d = real(d(:));
    [d,ord] = sort(d,'descend'); U = U(:,ord);
    if isempty(d) || max(d) <= 0
        E = inf; c = []; ok = false; return;
    end
    keep = d > mass_tol * max(d);
    if nnz(keep) < 1
        E = inf; c = []; ok = false; return;
    end
    X = U(:,keep) .* (1 ./ sqrt(d(keep))).';
    Aorth = X' * A * X; Aorth = (Aorth+Aorth')/2;
    [Y,D] = eig(Aorth);
    vals = real(diag(D));
    [E,idx] = min(vals);
    c = real(X * Y(:,idx));
    ok = isfinite(E) && all(isfinite(c));
end

function c = normalize_with_S(c,S)
    n = real(c' * S * c);
    if n <= 0 || ~isfinite(n)
        error('normalize_with_S failed.');
    end
    c = c ./ sqrt(n);
end

function [Jmat,Kmat] = JK_for_space1_given_c2(ints,c2)
    Q1 = size(ints.S11,1); Q2 = size(ints.S22,1);
    Jmat = zeros(Q1,Q1); Kmat = zeros(Q1,Q1);
    yy = c2(:)*c2(:).';
    for i = 1:Q1
        for j = 1:Q1
            sJ = 0; sK = 0;
            for a = 1:Q2
                for b = 1:Q2
                    yab = yy(a,b);
                    sJ = sJ + yab * ints.ERI1122(i,j,a,b); % (i j | a b)
                    sK = sK + yab * ints.ERI1221(i,a,b,j); % (i a | b j)
                end
            end
            Jmat(i,j) = sJ; Kmat(i,j) = sK;
        end
    end
    Jmat=(Jmat+Jmat')/2; Kmat=(Kmat+Kmat')/2;
end

function [Jmat,Kmat] = JK_for_space2_given_c1(ints,c1)
    Q1 = size(ints.S11,1); Q2 = size(ints.S22,1);
    Jmat = zeros(Q2,Q2); Kmat = zeros(Q2,Q2);
    xx = c1(:)*c1(:).';
    for a = 1:Q2
        for b = 1:Q2
            sJ = 0; sK = 0;
            for i = 1:Q1
                for j = 1:Q1
                    xij = xx(i,j);
                    sJ = sJ + xij * ints.ERI2211(a,b,i,j); % (a b | i j)
                    sK = sK + xij * ints.ERI2112(a,i,j,b); % (a i | j b)
                end
            end
            Jmat(a,b) = sJ; Kmat(a,b) = sK;
        end
    end
    Jmat=(Jmat+Jmat')/2; Kmat=(Kmat+Kmat')/2;
end

function comps = determinant_energy_comps(ints,c1,c2)
    n1 = real(c1' * ints.S11 * c1);
    n2 = real(c2' * ints.S22 * c2);
    s12 = real(c1' * ints.S12 * c2);
    h11 = real(c1' * ints.H11 * c1);
    h22 = real(c2' * ints.H22 * c2);
    h12 = real(c1' * ints.H12 * c2);
    D = n1*n2 - s12^2;

    J12 = contract4(ints.ERI1122,c1,c1,c2,c2);
    K12 = contract4(ints.ERI1221,c1,c2,c2,c1);

    E = (n2*h11 + n1*h22 - 2*s12*h12 + J12 - K12) / D;

    comps = struct();
    comps.E = E;
    comps.n1 = n1; comps.n2 = n2; comps.s12 = s12; comps.detGram = D;
    comps.h11 = h11; comps.h22 = h22; comps.h12 = h12;
    comps.J12 = J12; comps.K12 = K12;
    comps.altIter = NaN;
end

function val = contract4(T,a,b,c,d)
    val = 0;
    na = numel(a); nb = numel(b); nc = numel(c); nd = numel(d);
    for i = 1:na
        for j = 1:nb
            ab = a(i)*b(j);
            for k = 1:nc
                for l = 1:nd
                    val = val + ab*c(k)*d(l)*T(i,j,k,l);
                end
            end
        end
    end
    val = real(val);
end

%% ========================================================================
% Integral construction
% ========================================================================
function ints = build_two_orbital_integrals(terms1, terms2, Z, Rnuc)
    [S11,T11,Ven11] = one_particle_mats(terms1,terms1,Z,Rnuc);
    [S22,T22,Ven22] = one_particle_mats(terms2,terms2,Z,Rnuc);
    [S12,T12,Ven12] = one_particle_mats(terms1,terms2,Z,Rnuc);

    ints = struct();
    ints.S11 = (S11+S11')/2; ints.H11 = (T11+Ven11 + (T11+Ven11)')/2;
    ints.S22 = (S22+S22')/2; ints.H22 = (T22+Ven22 + (T22+Ven22)')/2;
    ints.S12 = S12; ints.H12 = T12 + Ven12;

    ints.ERI1122 = eri_tensor(terms1,terms1,terms2,terms2);
    ints.ERI1221 = eri_tensor(terms1,terms2,terms2,terms1);
    ints.ERI2211 = eri_tensor(terms2,terms2,terms1,terms1);
    ints.ERI2112 = eri_tensor(terms2,terms1,terms1,terms2);
end

function [S,T,Ven] = one_particle_mats(Aterms,Bterms,Z,Rnuc)
    na = numel(Aterms); nb = numel(Bterms);
    S = zeros(na,nb); T = zeros(na,nb); Ven = zeros(na,nb);
    for i = 1:na
        for j = 1:nb
            [S(i,j),T(i,j),Ven(i,j)] = one_particle_primitive( ...
                Aterms(i).alpha,Aterms(i).center,Bterms(j).alpha,Bterms(j).center,Z,Rnuc);
        end
    end
end

function T4 = eri_tensor(Aterms,Bterms,Cterms,Dterms)
    na=numel(Aterms); nb=numel(Bterms); nc=numel(Cterms); nd=numel(Dterms);
    T4 = zeros(na,nb,nc,nd);
    for a=1:na
        for b=1:nb
            for c=1:nc
                for d=1:nd
                    T4(a,b,c,d) = eri_primitive(Aterms(a).alpha,Aterms(a).center, ...
                        Bterms(b).alpha,Bterms(b).center,Cterms(c).alpha,Cterms(c).center, ...
                        Dterms(d).alpha,Dterms(d).center);
                end
            end
        end
    end
end

function terms = make_terms_from_alpha(alpha, center)
    Q = numel(alpha);
    terms = repmat(struct('coef',1,'alpha',0,'center',[0 0 0]), Q, 1);
    for q = 1:Q
        terms(q).coef = 1;
        terms(q).alpha = alpha(q);
        terms(q).center = center;
    end
end

function out = contracted_terms(primTerms, coeff)
    out = primTerms;
    for i = 1:numel(out)
        out(i).coef = coeff(i) * out(i).coef;
    end
end

function [S, T, Ven] = one_particle_primitive(alphaA, A, alphaB, B, Z, Rnuc)
    p = alphaA + alphaB;
    P = (alphaA*A + alphaB*B) ./ p;
    RAB2 = sum((A - B).^2);
    NA = (alphaA/pi)^(3/4); NB = (alphaB/pi)^(3/4);
    K = NA * NB * exp(-(alphaA*alphaB/(2*p)) * RAB2);
    S = K * (2*pi/p)^(3/2);
    T = 0.5 * alphaA * alphaB * (3/p - (alphaA*alphaB/p^2) * RAB2) * S;
    RP2 = sum((P - Rnuc).^2);
    Ven = -Z * K * (4*pi/p) * boys0(0.5 * p * RP2);
end

function val = eri_primitive(alphaA, A, alphaB, B, alphaC, C, alphaD, D)
    p = alphaA + alphaB; q = alphaC + alphaD;
    P = (alphaA*A + alphaB*B) ./ p;
    Qc = (alphaC*C + alphaD*D) ./ q;
    RAB2 = sum((A-B).^2); RCD2 = sum((C-D).^2); RPQ2 = sum((P-Qc).^2);
    NA = (alphaA/pi)^(3/4); NB = (alphaB/pi)^(3/4); NC = (alphaC/pi)^(3/4); ND = (alphaD/pi)^(3/4);
    Kab = NA * NB * exp(-(alphaA*alphaB/(2*p)) * RAB2);
    Kcd = NC * ND * exp(-(alphaC*alphaD/(2*q)) * RCD2);
    arg = (p*q/(2*(p+q))) * RPQ2;
    val = Kab * Kcd * (8*sqrt(2)*pi^(5/2)) / (p*q*sqrt(p+q)) * boys0(arg);
end

function F = boys0(t)
    if t < 1e-10
        F = 1 - t/3 + t^2/10 - t^3/42 + t^4/216 - t^5/1320;
    else
        F = 0.5 * sqrt(pi) / sqrt(t) * erf(sqrt(t));
    end
end

function comps = empty_comps()
    comps = struct('E',NaN,'n1',NaN,'n2',NaN,'s12',NaN,'detGram',NaN, ...
        'h11',NaN,'h22',NaN,'h12',NaN,'J12',NaN,'K12',NaN,'altIter',NaN);
end

function y = eval_orbital_on_x_axis(orb, xgrid)
    y = zeros(size(xgrid));
    for q = 1:numel(orb.terms)
        alpha = orb.terms(q).alpha;
        N = (alpha/pi)^(3/4);
        c = orb.terms(q).coef;
        y = y + c * N * exp(-0.5 * alpha * xgrid.^2);
    end
end
