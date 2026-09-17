%% JH_BeGround_save_coarse_candidates_nu12_13_14_from_existing_raw.m
% Reuse an already assembled Be raw/work matrix and save multiple coarsened
% initial systems for later adaptive runs.
%
% Intended use after a completed Be coarsening assembly such as:
%   JH_BeGround_BthreeOther0_raw_assembled_v3termFast_M4459.mat
%   JH_BeGround_BthreeOther0_raw_or_working_assembled_v4pairFast_M4459.mat
%
% This script DOES NOT reassemble S/H.  It only slices the existing raw
% matrices by coefficient thresholds |c_mu| >= 2^{-nu}, solves each selected
% subspace, and saves each candidate in the same part3-compatible format.

clear; clc;

%% ========================= user controls =========================
% Explicit filename is safest.  Leave empty to auto-discover the largest-M raw file.
rawFile = '';

candidateNuList = [12 13 14 15 16];

% Generalized eigensolve options for each selected subspace.
mass_tol = 1e-10;
verboseEig = false;

% Diagnostics only.  These values do not affect candidate construction.
E_ref_exact_default  = -14.66736;
E_table_kappa1      = -14.644213;
hartree_per_kcalmol = 1/627.5094740631;
delta               = hartree_per_kcalmol/100;

% Save both variable names for maximum downstream compatibility.
save_part3cand_alias = true;

%% ========================= load existing raw/work system =========================
if isempty(rawFile)
    rawFile = discover_existing_be_raw_file();
end

fprintf('\n============================================================\n');
fprintf('Save Be coarse candidates from existing raw/work matrix\n');
fprintf('Raw/work file: %s\n', rawFile);
fprintf('Candidate nu list:'); fprintf(' %d', candidateNuList); fprintf('\n');
fprintf('No matrix reassembly will be performed.\n');
fprintf('============================================================\n\n');

Sload = load(rawFile);
if isfield(Sload, 'part2')
    part2 = Sload.part2;
else
    error('File %s does not contain variable part2. Please choose a raw/work assembled file, not a final part3 file.', rawFile);
end

required = {'basis','S','H','coeff_raw','E_raw','oneFuncs'};
for k = 1:numel(required)
    if ~isfield(part2, required{k})
        error('part2.%s is missing in %s.', required{k}, rawFile);
    end
end

rawBasis = part2.basis(:).';
Sraw = part2.S;
Hraw = part2.H;
cr   = part2.coeff_raw(:);
Er   = part2.E_raw;
oneFuncs = part2.oneFuncs;

Mraw = numel(rawBasis);
if size(Sraw,1) ~= Mraw || size(Hraw,1) ~= Mraw || numel(cr) ~= Mraw
    error('Dimension mismatch: Mraw=%d, size(S)=%d, size(H)=%d, length(coeff_raw)=%d.', ...
        Mraw, size(Sraw,1), size(Hraw,1), numel(cr));
end

if isfield(part2, 'hf'), hf = part2.hf; else, hf = struct(); end
if isfield(part2, 'params'), params = part2.params; else, params = struct(); end
if isfield(params, 'E_ref_exact'), E_ref_exact = params.E_ref_exact; else, E_ref_exact = E_ref_exact_default; end
if isfield(params, 'delta'), delta = params.delta; end

rawCounts = count_blocks(rawBasis);
fprintf('Loaded raw/work system: M=%d | Mzero=%d | Mone=%d | Mud=%d | Muu=%d | Mthree=%d | Mother=%d\n', ...
    rawCounts.total, rawCounts.zero, rawCounts.one_total, rawCounts.two_ud, rawCounts.two_uu, rawCounts.three, rawCounts.other);
fprintf('Existing raw/work solve: E_raw=%.15f | delta=%.6e hartree\n', Er, delta);

%% ========================= save each candidate =========================
coefAbs = abs(cr(:));
records = repmat(empty_record(), 0, 1);
savedFiles = cell(numel(candidateNuList),1);

fprintf('\nCandidate scan from existing coefficients:\n');
fprintf('   nu       threshold          M   Mzero   Mone  Mtwo_ud Mtwo_uu Mthree Mother        E(nu)              E-Er\n');

for inu = 1:numel(candidateNuList)
    nu = candidateNuList(inu);
    thr = 2^(-nu);

    idx = find(coefAbs >= thr);
    idx = unique([1; idx(:)], 'stable');  % always retain the HF zero determinant

    S0 = Sraw(idx, idx);
    H0 = Hraw(idx, idx);
    basis0 = rawBasis(idx);

    [E0, coeff0, kept0, solveInfo0] = solve_ground_projected(H0, S0, mass_tol, verboseEig, E_ref_exact);
    counts0 = count_blocks(basis0);
    ener0 = compute_block_energy_contributions(coeff0, H0, basis0);

    selected = empty_record();
    selected.nu = nu;
    selected.thr = thr;
    selected.M = numel(idx);
    selected.Mzero = counts0.zero;
    selected.Mone = counts0.one_total;
    selected.Mtwo_ud = counts0.two_ud;
    selected.Mtwo_uu = counts0.two_uu;
    selected.Mthree = counts0.three;
    selected.Mother = counts0.other;
    selected.E = E0;
    selected.dE = E0 - Er;
    selected.err = abs(E0 - E_ref_exact);
    selected.keptDim = kept0;
    selected.idx = idx;
    records(end+1) = selected; %#ok<SAGROW>

    fprintf('%5d   %.6e   %6d   %5d  %5d  %7d %7d %6d %6d   %.15f   %.3e\n', ...
        nu, thr, selected.M, selected.Mzero, selected.Mone, selected.Mtwo_ud, selected.Mtwo_uu, ...
        selected.Mthree, selected.Mother, E0, selected.dE);

    part3 = struct();
    part3.inputFile = rawFile;
    part3.hf = hf;
    part3.params = params;
    part3.delta = delta;
    part3.hartree_per_kcalmol = hartree_per_kcalmol;
    part3.records = records;
    part3.selected = selected;
    part3.idxBest = idx;
    part3.nuBest = nu;
    part3.oneFuncs = oneFuncs;
    part3.basis0 = basis0;
    part3.S0 = S0;
    part3.H0 = H0;
    part3.selectedExactInfo = struct('used', false, ...
        'source', 'existing raw/work submatrix; no reassembly', ...
        'rawFile', rawFile);
    part3.E0 = E0;
    part3.coeff0 = coeff0;
    part3.kept0 = kept0;
    part3.solveInfo0 = solveInfo0;
    part3.counts0 = counts0;
    part3.blockEnergy0 = ener0;
    part3.rawSummary = make_raw_summary(part2, rawCounts, Er);

    part3cand = part3; %#ok<NASGU>
    outFile = sprintf('JH_BeGround_BthreeOther0_coarsened_fromExistingRaw_M%d_nu%d_candidate.mat', counts0.total, nu);
    if save_part3cand_alias
        save(outFile, 'part3', 'part3cand', '-v7.3');
    else
        save(outFile, 'part3', '-v7.3');
    end
    savedFiles{inu} = outFile;
    fprintf('        saved: %s\n', outFile);
end

%% ========================= save summary =========================
summary = struct();
summary.rawFile = rawFile;
summary.Eraw = Er;
summary.Mraw = Mraw;
summary.rawCounts = rawCounts;
summary.candidateNuList = candidateNuList;
summary.records = records;
summary.savedFiles = savedFiles;
summary.delta = delta;
summary.E_ref_exact = E_ref_exact;
summary.E_table_kappa1 = E_table_kappa1;

save('JH_BeGround_BthreeOther0_coarsened_fromExistingRaw_candidates_nu12_13_14_summary.mat', 'summary', '-v7.3');
write_candidate_csv(records, 'JH_BeGround_BthreeOther0_coarsened_fromExistingRaw_candidates_nu12_13_14.csv');

fprintf('\nSaved summary: JH_BeGround_BthreeOther0_coarsened_fromExistingRaw_candidates_nu12_13_14_summary.mat\n');
fprintf('Saved CSV    : JH_BeGround_BthreeOther0_coarsened_fromExistingRaw_candidates_nu12_13_14.csv\n');
fprintf('\nDone. Candidate files are adaptive-compatible part3 files.\n');

%% ========================================================================
% Local functions
% ========================================================================
function rawFile = discover_existing_be_raw_file()
    patterns = { ...
        'JH_BeGround_BthreeOther0_raw_or_working_assembled_v4pairFast_M*.mat', ...
        'JH_BeGround_BthreeOther0_raw_assembled_v4pairFast_M*.mat', ...
        'JH_BeGround_BthreeOther0_raw_assembled_v3termFast_M*.mat', ...
        'JH_BeGround_BthreeOther0_raw_assembled_v2fast_M*.mat', ...
        'JH_BeGround_BthreeOther0_raw_assembled_v1canon_M*.mat', ...
        'JH_BeGround_BthreeOther0_raw*_M*.mat'};

    all = struct([]);
    for p = 1:numel(patterns)
        d = dir(patterns{p});
        all = append_dir_unique(all, d);
    end
    if isempty(all)
        error(['No Be raw/work assembled matrix file found. Set rawFile manually, e.g. ', ...
               'rawFile = ''JH_BeGround_BthreeOther0_raw_assembled_v3termFast_M4459.mat'';']);
    end

    Mvals = zeros(numel(all),1);
    for i = 1:numel(all)
        Mvals(i) = extract_M_from_filename(all(i).name);
    end
    maxM = max(Mvals);
    idx = find(Mvals == maxM);
    if numel(idx) > 1
        [~,j] = max([all(idx).datenum]);
        idx = idx(j);
    end
    rawFile = all(idx).name;
end

function all = append_dir_unique(all, d)
    if isempty(d), return; end
    if isempty(all)
        all = d(:).';
        return;
    end
    names = {all.name};
    for k = 1:numel(d)
        if ~any(strcmp(names, d(k).name))
            all(end+1) = d(k); %#ok<AGROW>
            names{end+1} = d(k).name; %#ok<AGROW>
        end
    end
    all = all(:).';
end

function M = extract_M_from_filename(name)
    tok = regexp(name, '_M(\d+)\.mat$', 'tokens', 'once');
    if isempty(tok)
        tok = regexp(name, '_M(\d+)', 'tokens', 'once');
    end
    if isempty(tok)
        M = -1;
    else
        M = str2double(tok{1});
    end
end

function rec = empty_record()
    rec = struct('nu',NaN,'thr',NaN,'M',NaN,'Mzero',NaN,'Mone',NaN,'Mtwo_ud',NaN, ...
        'Mtwo_uu',NaN,'Mthree',NaN,'Mother',NaN,'E',NaN,'dE',NaN,'err',NaN,'keptDim',NaN,'idx',[]);
end

function counts = count_blocks(B)
    counts = struct('zero',0,'one_total',0,'two_ud',0,'two_uu',0,'three',0,'other',0,'total',numel(B));
    for i = 1:numel(B)
        blk = '';
        if isfield(B, 'block'), blk = char(string(B(i).block)); end
        switch blk
            case 'zero'
                counts.zero = counts.zero + 1;
            case 'one'
                counts.one_total = counts.one_total + 1;
            case 'two_ud'
                counts.two_ud = counts.two_ud + 1;
            case 'two_uu'
                counts.two_uu = counts.two_uu + 1;
            case 'three'
                counts.three = counts.three + 1;
            case 'other'
                counts.other = counts.other + 1;
        end
    end
end

function [E0, coeff, nkeep, info] = solve_ground_projected(H, S, mass_tol, verbose, targetEnergy)
    H = full((H + H')/2);
    S = full((S + S')/2);
    [U,D] = eig(S);
    d = real(diag(D));
    dmax = max(d);
    if ~isfinite(dmax) || dmax <= 0
        error('Mass matrix is not positive in projected solve: max eig(S)=%.3e.', dmax);
    end
    keep = d > mass_tol*dmax;
    nkeep = nnz(keep);
    if nkeep == 0
        error('All mass eigenmodes were removed. Increase numerical stability or lower mass_tol.');
    end
    if verbose
        fprintf('  mass eig: kept %d/%d | min kept=%.3e | max=%.3e\n', nkeep, numel(d), min(d(keep)), dmax);
    end
    X = U(:,keep) * diag(1 ./ sqrt(d(keep)));
    Hp = (X' * H * X); Hp = (Hp + Hp')/2;
    [Y,Eval] = eig(Hp);
    evals = real(diag(Eval));
    [E0,pos] = min(evals);
    coeff = X * Y(:,pos);
    coeff = coeff / sqrt(real(coeff' * S * coeff));
    res = norm(H*coeff - E0*S*coeff) / max(1, norm(H*coeff));
    info = struct('residual',res,'nkeep',nkeep,'massEigMinKept',min(d(keep)), ...
        'massEigMax',dmax,'targetEnergy',targetEnergy,'belowTarget',E0 < targetEnergy - 1e-4);
    if info.belowTarget
        warning('Projected energy %.15f is below target diagnostic %.15f. Check mass_tol/linear dependence.', E0, targetEnergy);
    end
end

function be = compute_block_energy_contributions(c, H, B)
    H = (H + H')/2;
    idxZero = block_mask(B, 'zero');
    idxOne  = block_mask(B, 'one');
    idxUD   = block_mask(B, 'two_ud');
    idxUU   = block_mask(B, 'two_uu');
    idx3    = block_mask(B, 'three');
    idxO    = block_mask(B, 'other');

    row.zero   = row_energy(c,H,idxZero);
    row.one    = row_energy(c,H,idxOne);
    row.two_ud = row_energy(c,H,idxUD);
    row.two_uu = row_energy(c,H,idxUU);
    row.three  = row_energy(c,H,idx3);
    row.other  = row_energy(c,H,idxO);
    row.total  = real(c' * H * c);

    diagv.zero   = diag_energy(c,H,idxZero);
    diagv.one    = diag_energy(c,H,idxOne);
    diagv.two_ud = diag_energy(c,H,idxUD);
    diagv.two_uu = diag_energy(c,H,idxUU);
    diagv.three  = diag_energy(c,H,idx3);
    diagv.other  = diag_energy(c,H,idxO);
    diagv.total  = diagv.zero + diagv.one + diagv.two_ud + diagv.two_uu + diagv.three + diagv.other;
    be = struct('row',row,'diag',diagv);
end

function idx = block_mask(B, name)
    idx = false(numel(B),1);
    for i = 1:numel(B)
        if isfield(B, 'block') && strcmp(char(string(B(i).block)), name)
            idx(i) = true;
        end
    end
end

function e = row_energy(c,H,idx)
    if ~any(idx), e = 0; return; end
    e = real(c(idx)' * (H(idx,:) * c));
end

function e = diag_energy(c,H,idx)
    if ~any(idx), e = 0; return; end
    e = real(c(idx)' * (H(idx,idx) * c(idx)));
end

function rawSummary = make_raw_summary(part2, rawCounts, Er)
    rawSummary = struct();
    rawSummary.Mraw = rawCounts.total;
    rawSummary.Eraw = Er;
    rawSummary.countsRaw = rawCounts;
    if isfield(part2, 'counts_full_raw'), rawSummary.countsFullRaw = part2.counts_full_raw; end
    if isfield(part2, 'counts_raw'), rawSummary.countsRawOriginal = part2.counts_raw; end
    if isfield(part2, 'assemblyTime'), rawSummary.assemblyTime = part2.assemblyTime; end
    if isfield(part2, 'preselectInfo'), rawSummary.preselectInfo = part2.preselectInfo; end
end

function write_candidate_csv(records, filename)
    fid = fopen(filename, 'w');
    if fid < 0
        warning('Could not open CSV file for writing: %s', filename);
        return;
    end
    fprintf(fid, 'nu,threshold,M,Mzero,Mone,Mtwo_ud,Mtwo_uu,Mthree,Mother,E,dE,err,keptDim\n');
    for i = 1:numel(records)
        r = records(i);
        fprintf(fid, '%d,%.16e,%d,%d,%d,%d,%d,%d,%d,%.16f,%.16e,%.16e,%d\n', ...
            r.nu, r.thr, r.M, r.Mzero, r.Mone, r.Mtwo_ud, r.Mtwo_uu, r.Mthree, r.Mother, ...
            r.E, r.dE, r.err, r.keptDim);
    end
    fclose(fid);
end
