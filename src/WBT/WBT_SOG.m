%% run_WBT_script.m
clear; close all; clc;
% b = 1.22749083347315613;
b = 1.18780018960046; %(1e-12)
% sigma=0.90802447499108738;
sigma = 1;
r_c = 20;
rmin = 1;
rmax = sqrt(1+1600);
% rmin = 1e-7 ;
% rmax = 1e5;
% rmax = 10;
% r = logspace(log10(1e-9), log10(1e7), 1e4);   
r=linspace(rmin,rmax,100000);
k1 = -12;
% k2 = -6;
approx1 = zeros(size(r));
% for ell = -86:(k1-1) 
%     gaussian_term = (2 * log(b)) / sqrt(2 * pi * sigma^2) * (1 / b^ell).* exp(-0.5 * (r / (b^ell * sigma)).^2);
%     approx1 = approx1 + gaussian_term; 
% end
% for ell = (k2+1):60 
%     gaussian_term = (2 * log(b)) / sqrt(2 * pi * sigma^2) * (1 / b^ell) .* exp(-0.5 * (r / (b^ell * sigma)).^2);
%     approx1 = approx1 + gaussian_term; 
% end

n = 216-k1+1;
w = zeros(n,1);
s = zeros(n,1);
for l = 1:n
    w(l) = (2 * log(b)) / sqrt(2 * pi * sigma^2) * (1 / b^(l+k1-1))*exp(-0.5 * (b^(l+k1-1) * sigma)^(-2));
    s(l) = 0.5 * (b^(l+k1-1) * sigma)^(-2);
end
s = s(:); w = w(:);
p = 28;                     
T = rmax^2;            
method = "WBT";

opt.alpha = 0.25;
opt.K = 5;                

A   = diag(-s);             % A = -diag(s)
B   = sqrt(w);
C   = B.';
AA  = s + s.';              % AA(i,j) = s_i + s_j
EA  = exp(-T * s);

P = B * B.';
Q = C' * C;

switch method
    case "classical"
        % Classical MR:
        P = P ./ AA;
        Q = Q ./ (AA.');
    case "TLBT"
        % Time-limited BT: w(t)=1
        P = P .* (1 - EA * EA')./(AA);
        Q = Q .* (1 - (EA * EA').')./(AA.');
    case "WBT"
        % Weighted BT: w(r)=(r+K)^(-alpha)
                % Weighted BT: w(r)=(r+K)^(-alpha)
        K = opt.K;
        alpha = opt.alpha;
        % we will compute P = (B*B') .* I  where
        % I_ij = \int_0^T exp(-(s_i+s_j)*t) * (t+K)^(-2*alpha) dt
        % use analytic closed-form where safe, otherwise use quadgk.

        nsub = length(s);   % note: in your script s is the current sub-vector
        base = (B * B.');   % base weights sqrt(wi*wj)

        AA = s + s.';       % already computed above
        T_local = T;        % T is x upper limit (r^2 if you use x=r^2)

        I = zeros(nsub, nsub);

        % threshold to decide whether exp(K * AA) will overflow or be unstable
        % exp(700) ~ 1e304, so choose 700 as safe threshold
        th = 700;

        % choose a tolerance for quadgk
        reltol = 1e-9;
        abstol = 1e-12;

        % Precompute matrices for speed
        % We'll compute only upper triangle and mirror
        % If you have Parallel Toolbox, you can replace outer loop by parfor
        for i = 1:nsub
            ai = AA(i,i); % just for small optimization
            for j = i:nsub
                aij = AA(i,j);
                safe = (K * aij) < th && isfinite(aij) && (aij > 0);
                if safe
                    % use analytic formula for alpha == 0.5 if that's the branch
                    if abs(alpha - 0.5) < 1e-14
                        % analytic: I = exp(aij*K) * (expint(aij*K) - expint(aij*(T_local+K)))
                        % compute gg0 and ggt using expint (should be stable for moderate args)
                        gg0 = expint(aij * K);
                        ggt = expint(aij * (T_local + K));
                        % compute I_ij:
                        Iij = exp(aij * K) * (gg0 - ggt);
                    else
                        % general alpha closed-form (original code uses gammainc)
                        a_param = 1 - 2*alpha;
                        u1 = aij * K;
                        u2 = aij * (T_local + K);
                        Q1 = gammainc(u1, a_param, 'upper');
                        Q2 = gammainc(u2, a_param, 'upper');
                        G1 = Q1 .* gamma(a_param);
                        G2 = Q2 .* gamma(a_param);
                        % EWA = exp(aij * K) might still be safe here by safe check
                        Iij = exp(aij * K) * (G1 - G2) * (aij^(2*alpha - 1));
                    end
                else
                    % unsafe -> fallback to robust numerical integration
                    integrand = @(t) exp(-aij * t) .* (t + K).^(-2*alpha);
                    % quadgk on [0, T_local]
                    % catch possible integration failure
                    try
                        val = quadgk(integrand, rmin^2, T_local, 'RelTol', reltol, 'AbsTol', abstol);
                    catch ME
                        % if quadgk fails, try with looser tol or break
                        warning('quadgk failed at i=%d j=%d, trying looser tol: %s', i, j, ME.message);
                        val = quadgk(integrand, rmin^2, T_local, 'RelTol', 1e-6, 'AbsTol', 1e-10);
                    end
                    Iij = val;
                end
                I(i,j) = Iij;
                I(j,i) = Iij; % symmetry
            end
        end

        % now form P and Q
        P = base .* I;
        % Q should carry sign(w_i)*sign(w_j)
        signs = sign(w);
        Q = (signs * signs.') .* P;

    otherwise
        error("Unknown method “%s”", method)
end

if any(isinf(P(:))) && any(isinf(Q(:)))
    error('Detected Inf in P or Q; exiting.');
end
if any(isnan(P(:))) || any(isnan(Q(:)))
    error('Detected NaN in P or Q; exiting.');
end

RP = chol(P + 1e-14 * eye(size(P)), 'lower');
RQ = chol(Q + 1e-14 * eye(size(Q)), 'lower');
S = RP;
L = RQ;
LL = S' * L;

[U, Sigma_mat, ~] = svd(LL);
Sigma = diag(Sigma_mat);
fprintf(Sigma(p)/Sigma(1));
Sigma_12 = diag(Sigma .^ (-1/2));
Trans = S * U * Sigma_12;
invT = inv(Trans);
At = invT * A * Trans;
Bt = invT * B;
Ct = C * Trans;

Ad = At(1:p,1:p);
Bd = Bt(1:p);
Cd = Ct(1:p);

[V_eig, D_eig] = eig(Ad);
s_wbt = -diag(D_eig);
B_mr = inv(V_eig) * Bd;
C_mr = Cd * V_eig;
w_wbt = B_mr .* (C_mr.');

x_col = x.'; 
y_original = exp(-x_col * s.') * w;
y_wbt     = exp(-x_col * s_wbt.') * w_wbt;
error = abs((y_wbt + approx1') - 1./sqrt(r'.^2+1));
rerror = error .* sqrt(r'.^2+1);
% error = abs((y_wbt + approx1') - 1./r');
% rerror = error .* r';
merror = max(error);
mrerror = max(rerror);
fprintf(merror);
fprintf(mrerror);   

disp('s_wbt (first few):'); disp(s_wbt(1:length(s_wbt)));
disp('w_wbt (first few):'); disp(w_wbt(1:length(w_wbt)));

figure;
loglog(r, rerror');    
xlabel('r'); ylabel('relative error');
title('WBT relative error (analytic branch)');
