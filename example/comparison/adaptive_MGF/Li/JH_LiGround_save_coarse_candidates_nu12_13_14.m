%% JH_LiGround_save_coarse_candidates_nu12_13_14.m
% Save multiple Li coarsened candidates from an already assembled raw system.
%
% Use this after running:
%   JH_LiGround_part2_coarsening_v3_canonicalNg.m
%
% It avoids rebuilding the raw B_three^[0] basis and avoids reassembling S/H.
% The script loads the raw system, applies the same coefficient-threshold
% coarsening rule B(nu)={|v_mu|>=2^{-nu}}, and saves nu=12,13,14 candidates
% in a format compatible with the later adaptive main program.

clear; clc; close all;
format long e;

%% User settings
rawFile = '';                 % leave empty to auto-detect latest v3canon raw file
nuCandidates = [12 13 14];    % candidates to save for adaptive initial-state comparison
mass_tol = 1e-10;
verboseEig = true;
E_ref_exact = -7.47806;

%% Locate raw file
if isempty(rawFile)
    F = dir('JH_LiGround_Bthree0_raw_assembled_v3canon_M*.mat');
    if isempty(F)
        error('Cannot find JH_LiGround_Bthree0_raw_assembled_v3canon_M*.mat in current folder.');
    end
    [~,ix] = max([F.datenum]);
    rawFile = F(ix).name;
end
fprintf('\n============================================================\n');
fprintf('Saving Li coarsened candidate matrices from raw file:\n  %s\n', rawFile);
fprintf('nuCandidates = [%s]\n', num2str(nuCandidates));
fprintf('============================================================\n\n');

load(rawFile, 'part2');
required = {'S','H','basis','coeff_raw','E_raw','oneFuncs','hf','params'};
for k = 1:numel(required)
    if ~isfield(part2, required{k})
        error('Raw file does not contain part2.%s. Please rerun v3 canonical coarsening script.', required{k});
    end
end

Sraw = part2.S;
Hraw = part2.H;
rawBasis = part2.basis;
cr = part2.coeff_raw;
Er = part2.E_raw;
coefAbs = abs(cr(:));

fprintf('Loaded raw: M=%d | Eraw=%.15f | nnz(S/H)=%.3e/%.3e\n', ...
    numel(rawBasis), Er, nnz(Sraw), nnz(Hraw));

candidateRecords = struct('nu',{},'thr',{},'M',{},'Mzero',{},'Mone',{},'Mtwo_ud',{},'Mtwo_uu',{},'Mthree',{}, ...
    'E',{},'dE',{},'err',{},'keptDim',{},'residual',{},'outFile',{});

for q = 1:numel(nuCandidates)
    nu = nuCandidates(q);
    thr = 2^(-nu);
    idx = find(coefAbs >= thr);
    idx = unique([1; idx(:)], 'stable');

    S0 = Sraw(idx,idx);
    H0 = Hraw(idx,idx);
    basis0 = rawBasis(idx);

    [E0, coeff0, kept0, solveInfo0] = solve_ground_projected(H0, S0, mass_tol, verboseEig, E_ref_exact);
    counts0 = count_blocks(basis0);
    ener0 = compute_block_energy_contributions(coeff0, H0, basis0);

    part3 = struct();
    part3.inputFile = rawFile;
    part3.hf = part2.hf;
    part3.params = part2.params;
    if isfield(part2.params,'delta'), part3.delta = part2.params.delta; end
    part3.nuBest = nu;
    part3.selected = struct('nu',nu,'thr',thr,'M',numel(idx),'Mzero',counts0.zero, ...
        'Mone',counts0.one_total,'Mtwo_ud',counts0.two_ud,'Mtwo_uu',counts0.two_uu, ...
        'Mthree',counts0.three,'E',E0,'dE',E0-Er,'err',abs(E0-E_ref_exact), ...
        'keptDim',kept0,'idx',idx);
    part3.idxBest = idx;
    part3.oneFuncs = part2.oneFuncs;
    part3.basis0 = basis0;
    part3.S0 = S0;
    part3.H0 = H0;
    part3.E0 = E0;
    part3.coeff0 = coeff0;
    part3.kept0 = kept0;
    part3.solveInfo0 = solveInfo0;
    part3.counts0 = counts0;
    part3.blockEnergy0 = ener0;
    part3.rawSummary = struct('Mraw',numel(rawBasis),'Eraw',Er,'assemblyTime',getfield_safe(part2,'assemblyTime',NaN));
    part3.candidateNote = sprintf('Saved by %s for adaptive initial-basis comparison.', mfilename);

    outFile = sprintf('JH_LiGround_Bthree0_coarsened_v3canon_M%d_nu%d_candidate.mat', counts0.total, nu);
    save(outFile, 'part3', '-v7.3');

    candidateRecords(q).nu = nu;
    candidateRecords(q).thr = thr;
    candidateRecords(q).M = counts0.total;
    candidateRecords(q).Mzero = counts0.zero;
    candidateRecords(q).Mone = counts0.one_total;
    candidateRecords(q).Mtwo_ud = counts0.two_ud;
    candidateRecords(q).Mtwo_uu = counts0.two_uu;
    candidateRecords(q).Mthree = counts0.three;
    candidateRecords(q).E = E0;
    candidateRecords(q).dE = E0 - Er;
    candidateRecords(q).err = abs(E0 - E_ref_exact);
    candidateRecords(q).keptDim = kept0;
    candidateRecords(q).residual = solveInfo0.residual;
    candidateRecords(q).outFile = string(outFile);

    fprintf('\nSaved candidate nu=%d:\n', nu);
    fprintf('  file       = %s\n', outFile);
    fprintf('  M          = %d | zero/one/two_ud/two_uu/three = %d/%d/%d/%d/%d\n', ...
        counts0.total, counts0.zero, counts0.one_total, counts0.two_ud, counts0.two_uu, counts0.three);
    fprintf('  E          = %.15f | E-Eraw = %.6e | residual = %.3e\n', E0, E0-Er, solveInfo0.residual);
    fprintf('  row energy = zero %.6f | one %.6f | two_ud %.6f | two_uu %.6e | three %.6e | sum %.15f\n', ...
        ener0.row.zero, ener0.row.one, ener0.row.two_ud, ener0.row.two_uu, ener0.row.three, ener0.row.total);
end

T = struct2table(candidateRecords);
writetable(T, 'JH_LiGround_Bthree0_saved_candidates_nu12_13_14.csv');
save('JH_LiGround_Bthree0_saved_candidates_nu12_13_14_index.mat', 'candidateRecords', 'rawFile', 'nuCandidates');

fprintf('\n==================== SAVED CANDIDATE SUMMARY ====================\n');
fprintf(' nu      M      E                    dEraw        Mone  Mud  Muu  M3\n');
for q = 1:numel(candidateRecords)
    r = candidateRecords(q);
    fprintf('%3d  %6d  %.15f  %.3e  %5d %4d %4d %4d\n', ...
        r.nu, r.M, r.E, r.dE, r.Mone, r.Mtwo_ud, r.Mtwo_uu, r.Mthree);
end
fprintf('\nSaved index CSV: JH_LiGround_Bthree0_saved_candidates_nu12_13_14.csv\n');
fprintf('Saved index MAT: JH_LiGround_Bthree0_saved_candidates_nu12_13_14_index.mat\n');

%% ========================================================================
% Local helpers
% ========================================================================
function v = getfield_safe(s, name, default)
    if isfield(s,name), v = s.(name); else, v = default; end
end

function counts = count_blocks(B)
    M = numel(B); blocks = strings(M,1);
    for i = 1:M, blocks(i)=string(B(i).block); end
    counts = struct();
    counts.zero = nnz(blocks=="zero");
    counts.one_total = nnz(blocks=="one");
    counts.two_ud = nnz(blocks=="two_ud");
    counts.two_uu = nnz(blocks=="two_uu");
    counts.three = nnz(blocks=="three");
    counts.total = M;
end

function [E0,coeff,nkeep,info] = solve_ground_projected(H,S,mass_tol,verbose,targetEnergy)
    if nargin<5, targetEnergy=NaN; end
    H=full(0.5*(H+H'));
    S=full(0.5*(S+S'));
    [U,D]=eig(S);
    d=real(diag(D));
    keep=d>mass_tol*max(d);
    if ~any(keep), error('Mass matrix projection removed all dimensions.'); end
    Uk=U(:,keep);
    dk=d(keep);
    X=Uk*diag(1./sqrt(dk));
    Hp=0.5*(X'*H*X + (X'*H*X)');
    [Y,Ediag]=eig(Hp);
    evals=real(diag(Ediag));
    [E0,pos]=min(evals);
    coeff=X*Y(:,pos);
    coeff=coeff/sqrt(real(coeff'*S*coeff));
    nkeep=sum(keep);
    res=norm(H*coeff - E0*S*coeff)/max(1,norm(H*coeff));
    info=struct('residual',res,'nkeep',nkeep,'targetEnergy',targetEnergy);
    if verbose
        fprintf('    projected solve: nkeep=%d/%d, E=%.15f, residual=%.3e\n', nkeep, size(S,1), E0, res);
    end
end

function be = compute_block_energy_contributions(c,H,B)
    be = empty_block_energy();
    blocks = strings(numel(B),1);
    for i=1:numel(B), blocks(i)=string(B(i).block); end
    be.row.zero = row_energy(c,H,blocks=="zero");
    be.row.one = row_energy(c,H,blocks=="one");
    be.row.two_ud = row_energy(c,H,blocks=="two_ud");
    be.row.two_uu = row_energy(c,H,blocks=="two_uu");
    be.row.three = row_energy(c,H,blocks=="three");
    be.row.total = real(c'*H*c);
    be.diag.zero = diag_energy(c,H,blocks=="zero");
    be.diag.one = diag_energy(c,H,blocks=="one");
    be.diag.two_ud = diag_energy(c,H,blocks=="two_ud");
    be.diag.two_uu = diag_energy(c,H,blocks=="two_uu");
    be.diag.three = diag_energy(c,H,blocks=="three");
    be.diag.total = be.diag.zero + be.diag.one + be.diag.two_ud + be.diag.two_uu + be.diag.three;
end

function be = empty_block_energy()
    fields = {'zero','one','two_ud','two_uu','three','total'};
    for k=1:numel(fields), row.(fields{k})=0; diagv.(fields{k})=0; end %#ok<AGROW>
    be=struct('row',row,'diag',diagv);
end

function e = row_energy(c,H,idx)
    e = real(c(idx)' * (H(idx,:) * c));
end

function e = diag_energy(c,H,idx)
    e = real(c(idx)' * (H(idx,idx) * c(idx)));
end
