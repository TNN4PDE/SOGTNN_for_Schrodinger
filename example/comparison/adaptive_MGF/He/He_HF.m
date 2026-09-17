%% JH_He_Table51_part1_HF_Q9_orbital_v2.m
% Improved Part 1 for J.H.-style He Table 5.1 reproduction.
%
% Purpose:
%   Construct the rank-1 approximate Hartree-Fock reference orbital for He:
%       psi_He(x) = sum_{q=1}^Q c_q G_{sigma_q}(x), Q=9.
%
% Main improvement over v1:
%   1) Do NOT sort sigma inside the objective. Sorting inside fminsearch makes
%      the objective non-smooth when two sigmas cross.
%   2) Use a short SCF/linear-coefficient refinement for fixed sigmas.
%   3) Optimize only log(sigmas) in the outer Nelder-Mead loop, while the
%      coefficients are determined variationally by SCF for each set of sigmas.
%
% This is still the same variational family as J.H. Eq. (4.36)/(4.37), but
% numerically much more stable than optimizing all coefficients and sigmas
% simultaneously with raw Nelder-Mead.

clear; clc; close all;
format long e;

%% Parameters
Z = 2;
Q = 9;
center = [0,0,0];
rng(1);

% J.H. Table 5.4 gives approximate rank-1 HF for He about -2.86166,
% while the HF limit is about -2.86168.
E_HF_JH_approx = -2.86166;
E_HF_limit = -2.861679995612;

maxOuterStarts = 8;
maxOuterRestart = 2;

optsOuter = optimset('Display','iter', ...
                     'MaxFunEvals', 8e4, ...
                     'MaxIter', 2e4, ...
                     'TolX', 1e-10, ...
                     'TolFun', 1e-12);

% Wider deterministic starts. These are sigmas, not exponents.
sigmaStarts = {
    logspace(log10(0.015), log10(8.0),  Q)
};

fprintf('\n============================================================\n');
fprintf('J.H.-style He rank-1 HF orbital v2, Q=%d\n', Q);
fprintf('Target J.H. approx E_HF ~= %.8f; HF limit ~= %.12f\n', ...
    E_HF_JH_approx, E_HF_limit);
fprintf('Outer variables: log(sigmas); coefficients solved by SCF.\n');
fprintf('============================================================\n\n');

best = struct('E',inf,'logsigma',[],'orbital',[],'comps',[]);

for is = 1:min(maxOuterStarts, numel(sigmaStarts))
    logs0 = log(sigmaStarts{is}(:));
    if is > 1
        logs0 = logs0 + 0.03*randn(size(logs0));
    end

    [E0, orb0, comps0] = he_energy_outer_logsigma(logs0, Q, Z, center);
    fprintf('\nStart %d: initial E = %.15f\n', is, E0);

    logs = logs0;
    [logs, E, exitflag, output] = fminsearch( ...
        @(xx) he_energy_outer_logsigma(xx, Q, Z, center), logs, optsOuter);

    for r = 1:maxOuterRestart
        fprintf('  restart %d from E = %.15f\n', r, E);
        [logs, E, exitflag, output] = fminsearch( ...
            @(xx) he_energy_outer_logsigma(xx, Q, Z, center), logs, optsOuter);
    end

    [Echeck, orb, comps] = he_energy_outer_logsigma(logs, Q, Z, center);

    fprintf('Finished start %d: E=%.15f | h=%.15f | J=%.15f | SCF it=%d\n', ...
        is, Echeck, comps.h, comps.J, comps.scfIter);
    fprintf('  deviation from J.H. approx target = %.3e\n', Echeck - E_HF_JH_approx);
    fprintf('  deviation from HF limit           = %.3e\n\n', Echeck - E_HF_limit);

    if Echeck < best.E
        best.E = Echeck;
        best.logsigma = logs;
        best.orbital = orb;
        best.comps = comps;
        best.exitflag = exitflag;
        best.output = output;
    end
end

% Sort only for output/storage, not during optimization.
orbital = sort_orbital_by_sigma(best.orbital);
[Efinal, orbital, comps] = he_energy_from_orbital(orbital, Z, center);

fprintf('\n==================== FINAL HF ORBITAL v2 ====================\n');
fprintf('E_HF_Q9          = %.15f hartree\n', Efinal);
fprintf('J.H. approx ref  = %.15f hartree\n', E_HF_JH_approx);
fprintf('HF limit ref     = %.15f hartree\n', E_HF_limit);
fprintf('diff to J.H.     = %.6e hartree\n', Efinal - E_HF_JH_approx);
fprintf('diff to HF limit = %.6e hartree\n', Efinal - E_HF_limit);
fprintf('one-electron h   = %.15f hartree\n', comps.h);
fprintf('Coulomb J        = %.15f hartree\n', comps.J);
fprintf('orbital norm     = %.15f\n', comps.norm);
fprintf('\n q          coeff_normalized              sigma\n');
for q = 1:Q
    fprintf('%2d   %+ .16e      %.16e\n', q, orbital.coeff(q), orbital.sigma(q));
end

hf = struct();
hf.E = Efinal;
hf.h = comps.h;
hf.J = comps.J;
hf.coeff = orbital.coeff;
hf.sigma = orbital.sigma;
hf.alpha = orbital.alpha;
hf.center = center;
hf.terms = orbital.terms;
hf.params = struct('Z',Z,'Q',Q,'center',center, ...
                   'E_HF_JH_approx',E_HF_JH_approx, ...
                   'E_HF_limit',E_HF_limit, ...
                   'description','J.H. Eq. (4.36)/(4.37), He Q=9 s-type; v2 SCF coefficients');
hf.best = best;

saveName = 'JH_He_HF_Q9_orbital_v2.mat';
save(saveName, 'hf', '-v7.3');
fprintf('\nSaved: %s\n', saveName);

xgrid = linspace(-8,8,1200).';
y = eval_orbital_on_x_axis(hf, xgrid);
figure('Color','w');
plot(xgrid, y, 'LineWidth', 1.5);
xlabel('x_1 [bohr]');
ylabel('\psi^{He}(x_1,0,0)');
title(sprintf('Q=9 rank-1 HF orbital for He, E=%.10f', Efinal));
grid on;

%% ========================================================================
% Objective over log(sigmas); coefficients solved by SCF.
% ========================================================================

function [E, orbital, comps] = he_energy_outer_logsigma(logsigma, Q, Z, center)
    logsigma = logsigma(:);
    if numel(logsigma) ~= Q || any(~isfinite(logsigma))
        E = 1e20; orbital = []; comps = empty_comps(); return;
    end

    % Soft bounds by penalty, not hard clipping. Avoids flat clipped regions.
    low = log(5e-4);
    high = log(5e1);
    penalty = 0;
    if any(logsigma < low)
        penalty = penalty + 1e4*sum((logsigma(logsigma<low)-low).^2);
    end
    if any(logsigma > high)
        penalty = penalty + 1e4*sum((logsigma(logsigma>high)-high).^2);
    end

    logsigma = min(max(logsigma, low), high);
    sigma = exp(logsigma);
    alpha = 1 ./ sigma.^2;

    terms = make_terms_from_alpha(alpha, center);
    ints = primitive_integrals(terms, Z, center);

    [c, Escf, compsScf] = scf_one_orbital(ints);

    orbital = struct();
    orbital.coeff = c;
    orbital.sigma = sigma;
    orbital.alpha = alpha;
    orbital.center = center;
    orbital.terms = terms;

    E = Escf + penalty;
    comps = compsScf;
end

function [c, E, comps] = scf_one_orbital(ints)
    S = ints.S;
    H = ints.Hcore;
    ERI = ints.ERI;
    Q = size(S,1);

    % Initial coefficient: lowest one-electron orbital.
    [V,D] = eig((H+H')/2, (S+S')/2);
    [~,idx] = min(real(diag(D)));
    c = real(V(:,idx));
    c = normalize_c(c,S);

    Eold = inf;
    mix = 0.35;
    maxIt = 300;
    tolE = 1e-13;

    for it = 1:maxIt
        Jmat = coulomb_matrix_from_c(ERI, c);
        F = H + Jmat;
        F = (F+F')/2;

        [V,D] = eig(F, (S+S')/2);
        evals = real(diag(D));
        [~,idx] = min(evals);
        cNew = real(V(:,idx));
        cNew = normalize_c(cNew,S);

        % Fix arbitrary sign for stable mixing.
        if cNew' * S * c < 0
            cNew = -cNew;
        end

        cMix = normalize_c((1-mix)*c + mix*cNew, S);
        c = cMix;

        [E, h, J] = energy_from_c(ints, c);

        if abs(E - Eold) < tolE
            break;
        end
        Eold = E;
    end

    comps = struct();
    comps.h = h;
    comps.J = J;
    comps.norm = real(c' * S * c);
    comps.scfIter = it;
end

function Jmat = coulomb_matrix_from_c(ERI, c)
    Q = numel(c);
    Jmat = zeros(Q,Q);
    for mu = 1:Q
        for nu = 1:Q
            s = 0;
            for la = 1:Q
                for si = 1:Q
                    s = s + c(la)*c(si)*ERI(mu,nu,la,si);
                end
            end
            Jmat(mu,nu) = s;
        end
    end
    Jmat = (Jmat+Jmat')/2;
end

function [E,h,J] = energy_from_c(ints, c)
    h = real(c' * ints.Hcore * c);
    J = 0;
    Q = numel(c);
    for a = 1:Q
        for b = 1:Q
            cab = c(a)*c(b);
            for cc = 1:Q
                for d = 1:Q
                    J = J + cab*c(cc)*c(d)*ints.ERI(a,b,cc,d);
                end
            end
        end
    end
    E = 2*h + J;
end

function c = normalize_c(c,S)
    n2 = real(c' * S * c);
    if ~isfinite(n2) || n2 <= 0
        c = nan(size(c));
    else
        c = c / sqrt(n2);
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

function ints = primitive_integrals(terms, Z, Rnuc)
    Q = numel(terms);
    S = zeros(Q,Q);
    T = zeros(Q,Q);
    Ven = zeros(Q,Q);
    for a = 1:Q
        for b = 1:Q
            [S(a,b), T(a,b), Ven(a,b)] = one_particle_primitive( ...
                terms(a).alpha, terms(a).center, ...
                terms(b).alpha, terms(b).center, Z, Rnuc);
        end
    end
    ERI = zeros(Q,Q,Q,Q);
    for a = 1:Q
        for b = 1:Q
            for c = 1:Q
                for d = 1:Q
                    ERI(a,b,c,d) = eri_primitive( ...
                        terms(a).alpha, terms(a).center, ...
                        terms(b).alpha, terms(b).center, ...
                        terms(c).alpha, terms(c).center, ...
                        terms(d).alpha, terms(d).center);
                end
            end
        end
    end
    ints = struct('S',S,'T',T,'Ven',Ven,'Hcore',T+Ven,'ERI',ERI);
end

function [E, orbital, comps] = he_energy_from_orbital(orbital, Z, center)
    terms = make_terms_from_alpha(orbital.alpha, center);
    ints = primitive_integrals(terms, Z, center);
    c = normalize_c(orbital.coeff(:), ints.S);
    orbital.coeff = c;
    orbital.terms = terms;
    [E,h,J] = energy_from_c(ints, c);
    comps = struct('h',h,'J',J,'norm',real(c'*ints.S*c));
end

function orbital = sort_orbital_by_sigma(orbital)
    [sigma,ord] = sort(orbital.sigma(:),'ascend');
    orbital.sigma = sigma;
    orbital.alpha = orbital.alpha(ord);
    orbital.coeff = orbital.coeff(ord);
    orbital.terms = make_terms_from_alpha(orbital.alpha, orbital.center);
end

function comps = empty_comps()
    comps = struct('h',NaN,'J',NaN,'norm',NaN,'scfIter',NaN);
end

%% ========================================================================
% Primitive integral formulas for L2-normalized Gaussians
% G_alpha = (alpha/pi)^(3/4) exp[-alpha |x-A|^2 / 2],
% with alpha = 1/sigma^2.
% ========================================================================

function [S, T, Ven] = one_particle_primitive(alphaA, A, alphaB, B, Z, Rnuc)
    p = alphaA + alphaB;
    P = (alphaA*A + alphaB*B) ./ p;
    RAB2 = sum((A - B).^2);

    NA = (alphaA/pi)^(3/4);
    NB = (alphaB/pi)^(3/4);

    K = NA * NB * exp(-(alphaA*alphaB/(2*p)) * RAB2);

    S = K * (2*pi/p)^(3/2);

    T = 0.5 * alphaA * alphaB * ...
        (3/p - (alphaA*alphaB/p^2) * RAB2) * S;

    RP2 = sum((P - Rnuc).^2);
    Ven = -Z * K * (4*pi/p) * boys0(0.5 * p * RP2);
end

function val = eri_primitive(alphaA, A, alphaB, B, alphaC, C, alphaD, D)
    p = alphaA + alphaB;
    q = alphaC + alphaD;

    P = (alphaA*A + alphaB*B) ./ p;
    Qc = (alphaC*C + alphaD*D) ./ q;

    RAB2 = sum((A-B).^2);
    RCD2 = sum((C-D).^2);
    RPQ2 = sum((P-Qc).^2);

    NA = (alphaA/pi)^(3/4);
    NB = (alphaB/pi)^(3/4);
    NC = (alphaC/pi)^(3/4);
    ND = (alphaD/pi)^(3/4);

    Kab = NA * NB * exp(-(alphaA*alphaB/(2*p)) * RAB2);
    Kcd = NC * ND * exp(-(alphaC*alphaD/(2*q)) * RCD2);

    arg = (p*q/(2*(p+q))) * RPQ2;
    val = Kab * Kcd * (8*sqrt(2)*pi^(5/2)) / ...
        (p*q*sqrt(p+q)) * boys0(arg);
end

function F = boys0(t)
    if t < 1e-10
        F = 1 - t/3 + t^2/10 - t^3/42 + t^4/216 - t^5/1320;
    else
        F = 0.5 * sqrt(pi) / sqrt(t) * erf(sqrt(t));
    end
end

function y = eval_orbital_on_x_axis(hf, xgrid)
    Q = numel(hf.sigma);
    y = zeros(size(xgrid));
    for q = 1:Q
        alpha = hf.alpha(q);
        N = (alpha/pi)^(3/4);
        y = y + hf.coeff(q) * N * exp(-0.5 * alpha * xgrid.^2);
    end
end
