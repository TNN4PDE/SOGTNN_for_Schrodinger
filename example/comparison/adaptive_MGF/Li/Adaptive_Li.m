%% JH_LiGround_part3_adaptive_v2_multiInit_debugged.m
% Li ground-state adaptive scheme for J.H. Table 5.9 reproduction.
% v2: debugged file discovery, row/column basis orientation, and robust coarse-file matching.
%
% This is Part 3 after:
%   Part 1: JH_LiGround_HF_Q9_orbitals.mat
%   Part 2: JH_LiGround_Bthree0_coarsened_v3canon_M*_nu*.mat
%
% Main design:
%   - N = 3, M_S = 1/2, N_up=2, N_down=1.
%   - Basis element represents A_alpha[f_a(x1) f_b(x2)] * f_c(x3), with
%     an L2-normalized 2-alpha determinant times a beta one-particle function.
%   - Adaptive growth follows Algorithm 4.1 idea: select important functions
%     by epsilon_kappa=2^(-kappa), then replace active one-particle frame
%     functions by N_g and N_g^+ neighbors.
%   - API supports multiple coarsened initial spaces, e.g. nu=12/13/14,
%     so one can compare adaptive trajectories from several B^[0].
%
% Notes:
%   This first Li adaptive version intentionally uses the canonical-Ng mapping
%   fixed in Part 2 v3.  It does not rebuild B_three^[0].  It loads each
%   saved coarse candidate and only refines from there.

clear; clc; close all;
format long e;

%% ============================================================
% 0. User-facing controls / API
% =============================================================
% Initial-space selection API:
%   initMode = 'nuList'   : discover files whose name contains _nu<nu>
%   initMode = 'explicit' : use targetCoarseFiles exactly
%   initMode = 'latest'   : use latest Li coarsened file only
initMode = 'nuList';
targetNuList = 12;
targetCoarseFiles = { ...
    % 'JH_LiGround_Bthree0_coarsened_v3canon_M508_nu12_candidate.mat'
    % 'JH_LiGround_Bthree0_coarsened_v3canon_M690_nu13_candidate.mat'
    % 'JH_LiGround_Bthree0_coarsened_v3canon_M812_nu14_candidate.mat'
};
runAllInitializations = true;

% Adaptive controls.  For a first debug, kappa_max=5 is recommended.
kappa_max = 10;
max_inner = 4;
max_basis_allowed = 25000;
updateMode = 'monotone_union';       % same practical mode used in He v11 / He triplet tests
importanceMode = 'partial_orthogonal'; % J.H.-style m1=1 partial orthogonal coefficient score

% Exact references / diagnostics.
E_ref_exact = -7.47806;          % J.H. Table 5.13 exact diagnostic for Li
E_ref_tilde_table = -7.47702;    % J.H. final tilde energy diagnostic
E_ref_HF_table = -7.43271;

% Solver and assembly controls.
mass_tol = 1e-10;
verboseEig = false;

assemblyOpts = struct();
assemblyOpts.mode = 'cpu_parfor_sparse';
assemblyOpts.rowBlockSize = 512;       % reduce to 32/64 if memory is tight
assemblyOpts.startPool = true;
assemblyOpts.dropTolS = 1e-13;
assemblyOpts.dropTolH = 1e-12;
assemblyOpts.screenERIByOverlap = false;
assemblyOpts.screenTolSForERI = 1e-13;
assemblyOpts.screenTolHCheapForERI = 1e-12;
assemblyOpts.forceDiagonal = true;
assemblyOpts.showRowProgress = true;
assemblyOpts.upperTriangle = true;
assemblyOpts.detScaleMode = 'l2_normalized';

solverOpts = struct();
solverOpts.projectedDenseLimit = 25000;  % exact mass-projection dense solve; lower if RAM is tight
solverOpts.rejectBelowExact = true;
solverOpts.energyLowerTol = 5e-4;        % Li adaptive should never go far below exact; guard serious bugs
solverOpts.exactLowerBound = E_ref_exact;

recordOpts = struct();
recordOpts.computeBlockEnergyEachInner = true;
recordOpts.saveRecordsEveryInner = true;
recordOpts.saveBasisEveryMRecord = true;
recordOpts.savePrefix = 'JH_LiGround_Table59';

checkpointOpts = struct();
checkpointOpts.saveBasisOnly = true;
checkpointOpts.dropMatrixAfterEachInner = true;

% Optional practical early stop.  Keep false for strict tests.
innerStop = struct();
innerStop.enable = false;
innerStop.minInner = 2;
innerStop.minAdded = 3000;
innerStop.minGrowthRel = 0.35;
innerStop.minAbsGain = 5e-7;
innerStop.minRelGain = 0.01;

%% ============================================================
% 1. Discover and run selected initial spaces
% =============================================================
coarseFiles = discover_li_coarse_files(initMode, targetNuList, targetCoarseFiles);
if isempty(coarseFiles)
    error('No Li coarsened initial files found. Save nu=12/13/14 candidates first.');
end
if ~runAllInitializations
    coarseFiles = coarseFiles(1);
end

fprintf('\n============================================================\n');
fprintf('J.H. Li Table 5.9 adaptive scheme, multi-initial API\n');
fprintf('initMode=%s | runAllInitializations=%d | kappa_max=%d | max_inner=%d\n', ...
    initMode, runAllInitializations, kappa_max, max_inner);
fprintf('Using %d initial coarse file(s):\n', numel(coarseFiles));
for i = 1:numel(coarseFiles)
    fprintf('  [%d] %s\n', i, coarseFiles(i).name);
end
fprintf('============================================================\n');

allRuns = struct('coarseFile',{},'finalFile',{},'records',{},'recordsByM',{});
for iRun = 1:numel(coarseFiles)
    coarseFile = coarseFiles(iRun).name;
    runTag = make_run_tag_from_file(coarseFile, iRun);
    fprintf('\n\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n');
    fprintf('Starting Li adaptive run %d/%d from %s\n', iRun, numel(coarseFiles), coarseFile);
    fprintf('runTag = %s\n', runTag);
    fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n');

    [records, recordsByM, finalFile] = run_one_li_adaptive( ...
        coarseFile, runTag, kappa_max, max_inner, max_basis_allowed, updateMode, importanceMode, ...
        E_ref_exact, E_ref_tilde_table, E_ref_HF_table, mass_tol, verboseEig, ...
        assemblyOpts, solverOpts, recordOpts, checkpointOpts, innerStop);

    allRuns(end+1).coarseFile = string(coarseFile); %#ok<SAGROW>
    allRuns(end).finalFile = string(finalFile);
    allRuns(end).records = records;
    allRuns(end).recordsByM = recordsByM;
end

save('JH_LiGround_Table59_adaptive_multiInit_summary.mat','allRuns','-v7.3');
fprintf('\nAll Li adaptive runs finished. Summary saved: JH_LiGround_Table59_adaptive_multiInit_summary.mat\n');

%% ========================================================================
% Main run function
% ========================================================================
function [records, recordsByM, finalFile] = run_one_li_adaptive( ...
    coarseFile, runTag, kappa_max, max_inner, max_basis_allowed, updateMode, importanceMode, ...
    E_ref_exact, E_ref_tilde_table, E_ref_HF_table, mass_tol, verboseEig, ...
    assemblyOpts, solverOpts, recordOpts, checkpointOpts, innerStop)

    Sload = load(coarseFile);
    if isfield(Sload,'part3')
        part3 = Sload.part3;
    else
        error('File %s does not contain part3.', coarseFile);
    end

    hf = part3.hf;
    params = part3.params;
    oneFuncs = part3.oneFuncs;
    basis = part3.basis0;
    % Candidate files saved by the coarsening script may store struct arrays
    % as column vectors because idx is a column vector.  Keep all adaptive
    % concatenations row-oriented to avoid dimension-mismatch in [basis,grow].
    oneFuncs = oneFuncs(:).';
    basis = basis(:).';

    % parameters from coarsening
    Zcharge = params.Zcharge;
    Rnuc = params.Rnuc;
    sigma0 = params.sigma0;
    cscale = params.cscale;
    L0 = params.L0;
    D = params.D;
    ZsetMode = params.ZsetMode;
    centerConvention = params.centerConvention;
    if isfield(params,'centerForNeighbors'), centerForNeighbors = params.centerForNeighbors; else, centerForNeighbors = 'actual'; end
    if isfield(params,'neighborMode'), neighborMode = params.neighborMode; else, neighborMode = 'cross'; end
    if isfield(params,'detScaleMode'), assemblyOpts.detScaleMode = params.detScaleMode; end

    oneKeyMap = containers.Map('KeyType','char','ValueType','double');
    for ii = 1:numel(oneFuncs), oneKeyMap(char(oneFuncs(ii).key)) = ii; end

    oneCache = reset_one_cache(Zcharge, Rnuc);
    records = repmat(empty_record(), kappa_max, 1);
    recordsByM = repmat(empty_m_record(), 0, 1);
    sampleID = 0;
    tGlobal = tic;

    fprintf('Loaded initial space: %s\n', coarseFile);
    c0 = count_blocks(basis);
    fprintf('Initial B^[0]: M=%d | Mone=%d | Mtwo_ud=%d | Mtwo_uu=%d | Mthree=%d | E0(coarse)=%.15f\n', ...
        c0.total, c0.one_total, c0.two_ud, c0.two_uu, c0.three, part3.E0);
    fprintf('J.H. Table 5.9 first row target: M=610, E=-7.471645, Mone=253, Mtwo_ud=190, Mtwo_uu=108, Mthree=58.\n');
    fprintf('Parameters: sigma=%g, c=%g, L=%d, neighbor=%s, update=%s, importance=%s\n\n', ...
        sigma0, cscale, L0, neighborMode, updateMode, importanceMode);

    for kappa = 1:kappa_max
        epsK = 2^(-kappa);
        fprintf('\n############################################################\n');
        fprintf('Li adaptive kappa = %d, epsilon = %.6e, runTag=%s\n', kappa, epsK, runTag);
        fprintf('############################################################\n');

        finalInfo = struct();
        changed = false;
        prevAccepted = struct('valid',false);

        for inner = 1:max_inner
            tInner = tic;
            M = numel(basis);
            counts = count_blocks(basis);
            fprintf('\n  inner %d | M=%d | Mone/two_ud/two_uu/three = %d / %d / %d / %d\n', ...
                inner, M, counts.one_total, counts.two_ud, counts.two_uu, counts.three);

            if M > max_basis_allowed
                fprintf('  STOP: M=%d exceeds max_basis_allowed=%d.\n', M, max_basis_allowed);
                break;
            end
            if M > solverOpts.projectedDenseLimit
                fprintf('  STOP: M=%d exceeds solverOpts.projectedDenseLimit=%d for dense projected solve.\n', ...
                    M, solverOpts.projectedDenseLimit);
                break;
            end

            ticAsm = tic;
            oneCache = update_one_cache(oneCache, oneFuncs, Zcharge, Rnuc);
            [S,H,~] = assemble_li_matrices(basis, oneCache, oneFuncs, assemblyOpts);
            tAsm = toc(ticAsm);
            nnzS = nnz(S); nnzH = nnz(H); sparseGB = estimate_sparse_pair_gb(S,H);
            fprintf('  assembly: %.2f s | nnz(S/H)=%.3e/%.3e | sparse est %.3f GB\n', ...
                tAsm, nnzS, nnzH, sparseGB);

            ticSol = tic;
            [E0, coeff, nkeep, solverInfo] = solve_ground_projected(H, S, mass_tol, verboseEig, E_ref_exact, solverOpts);
            tSol = toc(ticSol);
            err = abs(E0 - E_ref_exact);
            fprintf('  solve: %.2f s | E=%.15f | err(exact diag)=%.3e | kept=%d | residual=%.3e\n', ...
                tSol, E0, err, nkeep, solverInfo.residual);
            if solverOpts.rejectBelowExact && E0 < E_ref_exact - solverOpts.energyLowerTol
                warning('Energy %.12f is below diagnostic exact %.12f by %.3e; check basis/solver.', ...
                    E0, E_ref_exact, E_ref_exact-E0);
            end

            ticImp = tic;
            switch lower(importanceMode)
                case 'partial_orthogonal'
                    scores = partial_orthogonal_importance(coeff, S);
                case 'raw'
                    scores = abs(coeff);
                otherwise
                    error('Unknown importanceMode: %s', importanceMode);
            end
            important = scores >= epsK;
            nImp = nnz(important);
            scoreMax = max(scores);
            if any(scores>0), scoreMinNonzero = min(scores(scores>0)); else, scoreMinNonzero = NaN; end
            tImp = toc(ticImp);
            fprintf('  important=%d / %d | score max=%.3e, min(nonzero)=%.3e | importance %.2f s\n', ...
                nImp, M, scoreMax, scoreMinNonzero, tImp);

            ticBE = tic;
            if recordOpts.computeBlockEnergyEachInner
                blockEnergy = compute_block_energy_contributions(coeff, H, basis);
            else
                blockEnergy = empty_block_energy();
            end
            tBE = toc(ticBE);
            fprintf('  row energy: zero=%.6f one=%.6f two_ud=%.6f two_uu=%.3e three=%.3e sum=%.15f\n', ...
                blockEnergy.row.zero, blockEnergy.row.one, blockEnergy.row.two_ud, ...
                blockEnergy.row.two_uu, blockEnergy.row.three, blockEnergy.row.total);

            finalInfo = struct('E0',E0,'err',err,'coeff',coeff,'nkeep',nkeep,'solverInfo',solverInfo, ...
                'inner',inner,'M',M,'counts',counts,'blockEnergy',blockEnergy, ...
                'tAsm',tAsm,'tSol',tSol,'tImp',tImp,'tBE',tBE,'nnzS',nnzS,'nnzH',nnzH,'sparseGB',sparseGB);

            if innerStop.enable && inner >= innerStop.minInner && isfield(prevAccepted,'valid') && prevAccepted.valid
                addedFromPrev = M - prevAccepted.M;
                growthRel = addedFromPrev / max(prevAccepted.M,1);
                absGain = prevAccepted.err - err;
                relGain = absGain / max(prevAccepted.err,realmin);
                if addedFromPrev >= innerStop.minAdded && growthRel >= innerStop.minGrowthRel && ...
                   (absGain <= innerStop.minAbsGain || relGain <= innerStop.minRelGain)
                    fprintf('  inner-stop: added=%d growth=%.2f%% gain=%.3e rel=%.2f%%; rollback.\n', ...
                        addedFromPrev, 100*growthRel, absGain, 100*relGain);
                    basis = prevAccepted.basis;
                    oneFuncs = prevAccepted.oneFuncs;
                    oneKeyMap = prevAccepted.oneKeyMap;
                    oneCache = prevAccepted.oneCache;
                    finalInfo = prevAccepted.finalInfo;
                    changed = false;
                    break;
                end
            end

            ticGrow = tic;
            oldKeys = libasis_keys(basis);
            oldM = numel(basis);
            [growBasis, oneFuncs, oneKeyMap] = grow_important_li_basis( ...
                basis(important), oneFuncs, oneKeyMap, sigma0, cscale, D, ZsetMode, ...
                centerConvention, centerForNeighbors, neighborMode);
            switch lower(updateMode)
                case 'monotone_union'
                    candidate = [basis(:).', growBasis(:).'];
                case 'paper_reset'
                    candidate = growBasis(:).';
                otherwise
                    error('Unknown updateMode: %s', updateMode);
            end
            [basisNew, ~] = unique_libasis(candidate);
            newKeys = libasis_keys(basisNew);
            added = setdiff(newKeys, oldKeys);
            nGrowRaw = numel(candidate);
            nNewM = numel(basisNew);
            nAdded = numel(added);
            tGrow = toc(ticGrow);
            fprintf('  grown raw=%d | unique new M=%d | added=%d | grow %.2f s\n', ...
                nGrowRaw, nNewM, nAdded, tGrow);

            sampleID = sampleID + 1;
            tInnerNoSave = toc(tInner);
            [recordsByM, tSave, ~] = append_li_M_record_and_checkpoint( ...
                recordsByM, sampleID, kappa, inner, epsK, M, counts, E0, err, blockEnergy, ...
                nkeep, solverInfo, tAsm, tSol, tImp, tBE, tGrow, tInnerNoSave, toc(tGlobal), ...
                nImp, scoreMax, scoreMinNonzero, nGrowRaw, nNewM, nAdded, nnzS, nnzH, sparseGB, ...
                oneFuncs, basis, records, recordOpts, runTag, params, importanceMode, updateMode, mass_tol, assemblyOpts, solverOpts);
            fprintf('  M-record #%d | kappa.inner=%d.%d | M=%d | err=%.3e | tAsm=%.1fs tSol=%.1fs tGrow=%.1fs tSave=%.1fs | cum=%.1fs\n', ...
                sampleID, kappa, inner, M, err, tAsm, tSol, tGrow, tSave, toc(tGlobal));

            if checkpointOpts.dropMatrixAfterEachInner
                clear S H coeff scores important blockEnergy;
            end

            if nNewM == oldM || nAdded == 0
                fprintf('  inner closure reached for kappa=%d; keep M=%d.\n', kappa, oldM);
                changed = false;
                break;
            else
                changed = true;
                if innerStop.enable
                    prevAccepted.valid = true;
                    prevAccepted.M = M;
                    prevAccepted.err = err;
                    prevAccepted.basis = basis;
                    prevAccepted.oneFuncs = oneFuncs;
                    prevAccepted.oneKeyMap = oneKeyMap;
                    prevAccepted.oneCache = oneCache;
                    prevAccepted.finalInfo = finalInfo;
                end
                basis = basisNew(:).';
            end
        end

        if isempty(fieldnames(finalInfo))
            warning('No solved state for kappa=%d; stopping.', kappa);
            records = records(1:kappa-1);
            break;
        end

        records(kappa) = make_kappa_record(kappa, epsK, finalInfo);
        fprintf('\n>>> record kappa=%d | M=%d | E=%.15f | err=%.3e | Mone=%d | Mud=%d | Muu=%d | M3=%d | inner=%d\n', ...
            kappa, finalInfo.M, finalInfo.E0, finalInfo.err, finalInfo.counts.one_total, ...
            finalInfo.counts.two_ud, finalInfo.counts.two_uu, finalInfo.counts.three, finalInfo.inner);

        adaptiveState = struct('coarseFile',coarseFile,'runTag',runTag,'records',records,'recordsByM',recordsByM, ...
            'kappa',kappa,'oneFuncs',oneFuncs,'basis',basis,'params',params, ...
            'importanceMode',importanceMode,'updateMode',updateMode,'mass_tol',mass_tol, ...
            'assemblyOpts',assemblyOpts,'solverOpts',solverOpts);
        save(sprintf('%s_adaptive_checkpoint_%s_kappa%d_M%d.mat', recordOpts.savePrefix, runTag, kappa, numel(basis)), ...
            'adaptiveState','-v7.3');

        if ~changed
            % Continue to next kappa with same closed basis, as in He code.
        end
    end

    fprintf('\n==================== Li TABLE 5.9 STYLE SUMMARY: %s ====================\n', runTag);
    fprintf(' kappa      M        E                    err       Mone   Mtwo_ud Mtwo_uu Mthree    Ezero      Eone       Etwo_ud    Etwo_uu    Ethree\n');
    for k = 1:numel(records)
        r = records(k);
        if isnan(r.kappa), continue; end
        fprintf('%5d  %7d  %.15f  %.3e  %6d %8d %7d %6d  %9.3f %9.3f %10.3f %10.3e %10.3e\n', ...
            r.kappa, r.M, r.E, r.err, r.Mone, r.Mtwo_ud, r.Mtwo_uu, r.Mthree, ...
            r.Ezero, r.Eone, r.Etwo_ud, r.Etwo_uu, r.Ethree);
    end

    fprintf('\n==================== M-INDEXED INNER RECORDS: %s ====================\n', runTag);
    fprintf(' sample  k.i       M        E                    err        tAsm     tSol    tGrow   tInner  tCum\n');
    for q = 1:numel(recordsByM)
        r = recordsByM(q);
        fprintf('%6d  %2d.%02d  %7d  %.15f  %.3e  %8.1f %8.1f %8.1f %8.1f %8.1f\n', ...
            r.sampleID, r.kappa, r.inner, r.M, r.E, r.err, ...
            r.assemblyTime, r.solveTime, r.growTime, r.totalInnerTimeNoSave, r.cumulativeTime);
    end

    recordsByM_unique = unique_records_by_M_keep_best(recordsByM);
    finalFile = sprintf('%s_adaptive_final_%s_kappa%d_M%d.mat', recordOpts.savePrefix, runTag, numel(records), numel(basis));
    finalState = struct('coarseFile',coarseFile,'runTag',runTag,'records',records,'recordsByM',recordsByM, ...
        'recordsByM_unique',recordsByM_unique,'oneFuncs',oneFuncs,'basis',basis,'params',params, ...
        'importanceMode',importanceMode,'updateMode',updateMode,'mass_tol',mass_tol, ...
        'assemblyOpts',assemblyOpts,'solverOpts',solverOpts, 'E_ref_exact',E_ref_exact, ...
        'E_ref_tilde_table',E_ref_tilde_table, 'E_ref_HF_table', E_ref_HF_table);
    save(finalFile, 'finalState', '-v7.3');

    csvFile = sprintf('%s_M_records_%s.csv', recordOpts.savePrefix, runTag);
    try
        T = struct2table(recordsByM);
        writetable(T, csvFile);
        fprintf('Saved M-record CSV: %s\n', csvFile);
    catch ME
        warning('Could not write M-record CSV: %s', ME.message);
    end
    fprintf('Saved final adaptive state: %s\n', finalFile);
end

%% ========================================================================
% File discovery and records
% ========================================================================
function files = discover_li_coarse_files(initMode, targetNuList, targetCoarseFiles)
    % Robust coarse-file discovery.
    % IMPORTANT: initialize as struct([]), not [], otherwise MATLAB tries to
    % convert dir() structs into double arrays when doing files(end+1)=d.
    files = struct([]);

    mode = lower(char(string(initMode)));
    switch mode
        case 'explicit' 
            for i = 1:numel(targetCoarseFiles)
                f = targetCoarseFiles{i};
                if isstring(f), f = char(f); end
                if exist(f,'file')
                    d = dir(f);
                    files = append_dir_unique(files, d);
                else
                    warning('Explicit coarse file not found: %s', f);
                end
            end

        case 'nulist' 
            all = [dir('JH_LiGround_Bthree0_coarsened_v3canon_M*_nu*_candidate.mat'); ...
                   dir('JH_LiGround_Bthree0_coarsened_v3canon_M*_nu*.mat')];
            if isempty(all), return; end

            % Remove duplicate names while preserving latest copy.  This avoids
            % selecting both the selected-coarse file and candidate file if both
            % match the same nu.  Candidate files are preferred when available.
            all = unique_dir_by_name_keep_latest(all);

            for nu = targetNuList(:).'
                hit = struct([]);
                pat = sprintf('_nu%d', nu);
                for i = 1:numel(all)
                    if contains(all(i).name, pat)
                        hit = append_dir_unique(hit, all(i));
                    end
                end
                if isempty(hit)
                    warning('No Li coarsened file found for nu=%d.', nu);
                    continue;
                end

                % Prefer explicit candidate files; otherwise latest matching file.
                isCand = contains({hit.name}, '_candidate.mat');
                if any(isCand)
                    cand = hit(isCand);
                    [~,ord] = max([cand.datenum]);
                    d = cand(ord);
                else
                    [~,ord] = max([hit.datenum]);
                    d = hit(ord);
                end
                files = append_dir_unique(files, d);
            end

        case 'latest' 
            all = [dir('JH_LiGround_Bthree0_coarsened_v3canon_M*_nu*_candidate.mat'); ...
                   dir('JH_LiGround_Bthree0_coarsened_v3canon_M*_nu*.mat')];
            if isempty(all), return; end
            all = unique_dir_by_name_keep_latest(all);
            [~,ord] = max([all.datenum]);
            files = all(ord);

        otherwise
            error('Unknown initMode: %s', initMode);
    end

    files = files(:).';
end

function files = append_dir_unique(files, d)
    if isempty(d), return; end
    if isempty(files)
        files = d;
        files = files(:).';
        return;
    end
    names = {files.name};
    for k = 1:numel(d)
        if ~any(strcmp(names, d(k).name))
            files(end+1) = d(k); %#ok<AGROW>
            names{end+1} = d(k).name; %#ok<AGROW>
        end
    end
    files = files(:).';
end

function out = unique_dir_by_name_keep_latest(in)
    out = struct([]);
    if isempty(in), return; end
    names = unique({in.name}, 'stable');
    for k = 1:numel(names)
        idx = find(strcmp({in.name}, names{k}));
        [~,j] = max([in(idx).datenum]);
        out = append_dir_unique(out, in(idx(j)));
    end
    out = out(:).';
end

function tag = make_run_tag_from_file(fname, idx)
    tokNu = regexp(fname,'_nu(\d+)','tokens','once');
    tokM = regexp(fname,'_M(\d+)','tokens','once');
    if isempty(tokNu), nu = sprintf('run%d',idx); else, nu = sprintf('nu%s',tokNu{1}); end
    if isempty(tokM), mm = ''; else, mm = sprintf('_M%s',tokM{1}); end
    tag = sprintf('%s%s',nu,mm);
    tag = regexprep(tag,'[^A-Za-z0-9_]','_');
end

function r = empty_record()
    r = struct('kappa',NaN,'epsilon',NaN,'M',NaN,'Mzero',NaN,'Mone',NaN, ...
        'Mtwo_ud',NaN,'Mtwo_uu',NaN,'Mthree',NaN,'E',NaN,'err',NaN,'inner',NaN, ...
        'kept',NaN,'solverResidual',NaN,'Ezero',NaN,'Eone',NaN,'Etwo_ud',NaN, ...
        'Etwo_uu',NaN,'Ethree',NaN,'assemblyTime',NaN,'solveTime',NaN, ...
        'importanceTime',NaN,'blockEnergyTime',NaN,'cumulativeTime',NaN);
end

function r = empty_m_record()
    r = struct('sampleID',NaN,'kappa',NaN,'inner',NaN,'epsilon',NaN, ...
        'M',NaN,'Mzero',NaN,'Mone',NaN,'Mtwo_ud',NaN,'Mtwo_uu',NaN,'Mthree',NaN, ...
        'E',NaN,'err',NaN,'Ezero',NaN,'Eone',NaN,'Etwo_ud',NaN,'Etwo_uu',NaN,'Ethree',NaN, ...
        'kept',NaN,'solverResidual',NaN,'assemblyTime',NaN,'solveTime',NaN, ...
        'importanceTime',NaN,'blockEnergyTime',NaN,'growTime',NaN,'saveTime',NaN, ...
        'totalInnerTimeNoSave',NaN,'totalInnerTimeWithSave',NaN,'cumulativeTime',NaN, ...
        'nImportant',NaN,'scoreMax',NaN,'scoreMinNonzero',NaN, ...
        'nGrowRaw',NaN,'nNewM',NaN,'nAdded',NaN,'nnzS',NaN,'nnzH',NaN, ...
        'sparseGB_est',NaN,'basisFile',"");
end

function r = make_kappa_record(kappa, epsK, info)
    r = empty_record();
    r.kappa = kappa; r.epsilon = epsK; r.M = info.M;
    r.Mzero = info.counts.zero; r.Mone = info.counts.one_total; r.Mtwo_ud = info.counts.two_ud;
    r.Mtwo_uu = info.counts.two_uu; r.Mthree = info.counts.three;
    r.E = info.E0; r.err = info.err; r.inner = info.inner; r.kept = info.nkeep;
    r.solverResidual = info.solverInfo.residual;
    r.Ezero = info.blockEnergy.row.zero; r.Eone = info.blockEnergy.row.one;
    r.Etwo_ud = info.blockEnergy.row.two_ud; r.Etwo_uu = info.blockEnergy.row.two_uu; r.Ethree = info.blockEnergy.row.three;
    r.assemblyTime = info.tAsm; r.solveTime = info.tSol; r.importanceTime = info.tImp; r.blockEnergyTime = info.tBE;
end

function [recordsByM, tSave, basisFile] = append_li_M_record_and_checkpoint( ...
    recordsByM, sampleID, kappa, inner, epsK, M, counts, E0, err, blockEnergy, ...
    nkeep, solverInfo, tAsm, tSol, tImp, tBE, tGrow, tInnerNoSave, tCum, ...
    nImp, scoreMax, scoreMinNonzero, nGrowRaw, nNewM, nAdded, nnzS, nnzH, sparseGB, ...
    oneFuncs, basis, records, recordOpts, runTag, params, importanceMode, updateMode, mass_tol, assemblyOpts, solverOpts)

    t0 = tic; basisFile = "";
    r = empty_m_record();
    r.sampleID = sampleID; r.kappa = kappa; r.inner = inner; r.epsilon = epsK;
    r.M = M; r.Mzero = counts.zero; r.Mone = counts.one_total; r.Mtwo_ud = counts.two_ud;
    r.Mtwo_uu = counts.two_uu; r.Mthree = counts.three;
    r.E = E0; r.err = err;
    r.Ezero = blockEnergy.row.zero; r.Eone = blockEnergy.row.one; r.Etwo_ud = blockEnergy.row.two_ud;
    r.Etwo_uu = blockEnergy.row.two_uu; r.Ethree = blockEnergy.row.three;
    r.kept = nkeep; r.solverResidual = solverInfo.residual;
    r.assemblyTime = tAsm; r.solveTime = tSol; r.importanceTime = tImp; r.blockEnergyTime = tBE;
    r.growTime = tGrow; r.totalInnerTimeNoSave = tInnerNoSave; r.cumulativeTime = tCum;
    r.nImportant = nImp; r.scoreMax = scoreMax; r.scoreMinNonzero = scoreMinNonzero;
    r.nGrowRaw = nGrowRaw; r.nNewM = nNewM; r.nAdded = nAdded; r.nnzS = nnzS; r.nnzH = nnzH; r.sparseGB_est = sparseGB;
    recordsByM(end+1,1) = r;

    if recordOpts.saveBasisEveryMRecord
        basisFile = string(sprintf('%s_basis_byM_%s_sample%04d_kappa%d_inner%d_M%d.mat', ...
            recordOpts.savePrefix, runTag, sampleID, kappa, inner, M));
        basisRecord = struct('sampleID',sampleID,'kappa',kappa,'inner',inner,'M',M, ...
            'oneFuncs',oneFuncs,'basis',basis,'record',recordsByM(end),'records',records, ...
            'params',params,'importanceMode',importanceMode,'updateMode',updateMode,'mass_tol',mass_tol, ...
            'assemblyOpts',assemblyOpts,'solverOpts',solverOpts);
        save(char(basisFile), 'basisRecord', '-v7.3');
    end
    recordsByM(end).basisFile = basisFile;

    if recordOpts.saveRecordsEveryInner
        liveFile = sprintf('%s_records_live_%s.mat', recordOpts.savePrefix, runTag);
        save(liveFile, 'recordsByM', 'records', '-v7.3');
    end
    tSave = toc(t0);
    recordsByM(end).saveTime = tSave;
    recordsByM(end).totalInnerTimeWithSave = tInnerNoSave + tSave;
end

function recordsUnique = unique_records_by_M_keep_best(recordsByM)
    if isempty(recordsByM), recordsUnique = recordsByM; return; end
    Mvals = [recordsByM.M].'; [Mu,~,ic] = unique(Mvals,'stable');
    recordsUnique = repmat(empty_m_record(), numel(Mu), 1);
    for i = 1:numel(Mu)
        idx = find(ic==i); errs = [recordsByM(idx).err]; [~,j] = min(errs);
        recordsUnique(i) = recordsByM(idx(j));
    end
end

%% ========================================================================
% Basis identities and adaptive growth
% ========================================================================
function keys = libasis_keys(B)
    keys = strings(numel(B),1);
    for i=1:numel(B), keys(i)=string(B(i).key); end
end

function [Buniq, map] = unique_libasis(B)
    if isempty(B), Buniq = B; map = containers.Map('KeyType','char','ValueType','double'); return; end
    B = B(:).';
    keys = libasis_keys(B); [~,ia] = unique(keys,'stable'); Buniq = B(ia); Buniq = Buniq(:).';
    map = containers.Map('KeyType','char','ValueType','double');
    for i=1:numel(Buniq), map(char(Buniq(i).key)) = i; end
end

function [B, map] = append_basis(B, map, b)
    if isempty(b), return; end
    if b.a == b.b, return; end
    key = char(b.key);
    if ~isKey(map,key)
        B(end+1) = b; 
        map(key) = numel(B);
    end
end

function b = make_li_basis(block, ia, ib, ic, tag)
    b = empty_libasis();
    ab = sort([ia ib]);
    b.block = char(block); b.a = ab(1); b.b = ab(2); b.c = ic; b.tag = char(tag);
    b.key = string(sprintf('%s_A%d_%d_B%d', char(block), b.a, b.b, b.c));
end

function b = empty_libasis()
    b = struct('block','','a',0,'b',0,'c',0,'tag','','key',"");
end

function [growBasis, oneFuncs, oneKeyMap] = grow_important_li_basis( ...
    importantBasis, oneFuncs, oneKeyMap, sigma0, cscale, D, ZsetMode, ...
    centerConvention, centerForNeighbors, neighborMode)

    out = {}; k = 0;
    refA1 = 1; refA2 = 2; refB = 3;

    for ib = 1:numel(importantBasis)
        B = importantBasis(ib);
        block = string(B.block);
        tag = string(B.tag);

        switch block
            case "zero"
                k=k+1; out{k}=B; %#ok<AGROW>

            case "one"
                if tag == "one_a1"
                    active = active_alpha_excluding(B, refA2);
                    if isempty(active), continue; end
                    neigh = onefunc_neighbors_both(oneFuncs(active), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    for n=1:numel(neigh)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                        k=k+1; out{k}=make_li_basis('one', nid, refA2, refB, 'one_a1'); %#ok<AGROW>
                    end
                elseif tag == "one_a2"
                    active = active_alpha_excluding(B, refA1);
                    if isempty(active), continue; end
                    neigh = onefunc_neighbors_both(oneFuncs(active), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    for n=1:numel(neigh)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                        k=k+1; out{k}=make_li_basis('one', refA1, nid, refB, 'one_a2'); %#ok<AGROW>
                    end
                elseif tag == "one_b"
                    active = B.c;
                    neigh = onefunc_neighbors_both(oneFuncs(active), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    for n=1:numel(neigh)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                        k=k+1; out{k}=make_li_basis('one', refA1, refA2, nid, 'one_b'); %#ok<AGROW>
                    end
                else
                    % Fallback inference.
                    ids = [B.a B.b B.c]; active = ids(ids>3);
                    if isempty(active), continue; end
                    neigh = onefunc_neighbors_both(oneFuncs(active(1)), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    for n=1:numel(neigh)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                        k=k+1; out{k}=make_li_basis('one', nid, refA2, refB, 'one_a1'); %#ok<AGROW>
                    end
                end

            case "two_ud"
                if tag == "two_a1b"
                    activeA = active_alpha_excluding(B, refA2); fixedA = refA2;
                    if isempty(activeA), continue; end
                    neighA = onefunc_neighbors_both(oneFuncs(activeA), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    neighB = onefunc_neighbors_both(oneFuncs(B.c), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    for n=1:numel(neighA)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neighA(n));
                        k=k+1; out{k}=make_li_basis('two_ud', nid, fixedA, B.c, 'two_a1b'); %#ok<AGROW>
                    end
                    for n=1:numel(neighB)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neighB(n));
                        k=k+1; out{k}=make_li_basis('two_ud', activeA, fixedA, nid, 'two_a1b'); %#ok<AGROW>
                    end
                elseif tag == "two_a2b"
                    activeA = active_alpha_excluding(B, refA1); fixedA = refA1;
                    if isempty(activeA), continue; end
                    neighA = onefunc_neighbors_both(oneFuncs(activeA), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    neighB = onefunc_neighbors_both(oneFuncs(B.c), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    for n=1:numel(neighA)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neighA(n));
                        k=k+1; out{k}=make_li_basis('two_ud', fixedA, nid, B.c, 'two_a2b'); %#ok<AGROW>
                    end
                    for n=1:numel(neighB)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neighB(n));
                        k=k+1; out{k}=make_li_basis('two_ud', fixedA, activeA, nid, 'two_a2b'); %#ok<AGROW>
                    end
                else
                    % General two_ud fallback: grow both nonreference alpha/beta entries.
                    growIDs = unique([B.a B.b B.c]);
                    for id0 = growIDs
                        if id0 <= 3, continue; end
                        neigh = onefunc_neighbors_both(oneFuncs(id0), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                        for n=1:numel(neigh)
                            [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                            bb = B; if bb.a==id0, bb.a=nid; elseif bb.b==id0, bb.b=nid; elseif bb.c==id0, bb.c=nid; end
                            k=k+1; out{k}=make_li_basis('two_ud', bb.a, bb.b, bb.c, char(tag)); %#ok<AGROW>
                        end
                    end
                end

            case "two_uu"
                idsA = [B.a B.b];
                for pos=1:2
                    active = idsA(pos); other = idsA(3-pos);
                    neigh = onefunc_neighbors_both(oneFuncs(active), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    for n=1:numel(neigh)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                        if nid ~= other
                            k=k+1; out{k}=make_li_basis('two_uu', nid, other, refB, 'two_aa'); %#ok<AGROW>
                        end
                    end
                end

            case "three"
                ids = [B.a B.b B.c];
                for pos=1:3
                    active = ids(pos);
                    neigh = onefunc_neighbors_both(oneFuncs(active), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    for n=1:numel(neigh)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                        newIDs = ids; newIDs(pos)=nid;
                        if newIDs(1) ~= newIDs(2)
                            k=k+1; out{k}=make_li_basis('three', newIDs(1), newIDs(2), newIDs(3), 'three'); %#ok<AGROW>
                        end
                    end
                end
            otherwise
                error('Unknown Li basis block: %s', block);
        end
    end

    if isempty(out), growBasis = importantBasis([]); else, growBasis = [out{:}]; end
end

function active = active_alpha_excluding(B, fixedRef)
    ids = [B.a B.b];
    ids(ids==fixedRef) = [];
    if isempty(ids), active = []; else, active = ids(1); end
end

function [oneFuncs, oneKeyMap, id] = register_onefunc(oneFuncs, oneKeyMap, f)
    key = char(f.key);
    if isKey(oneKeyMap,key)
        id = oneKeyMap(key);
    else
        oneFuncs = oneFuncs(:).';
        id = numel(oneFuncs)+1;
        oneFuncs(id) = f;
        oneKeyMap(key) = id;
    end
end

%% ========================================================================
% Frame / neighbor functions, canonical Ng + Ngplus
% ========================================================================
function f = empty_onefunc()
    f = struct('kind','','type','','level',0,'j',[0 0 0],'z',[0 0 0], ...
        'terms',struct('coef',{},'alpha',{},'center',{}),'key',"");
end

function neigh = onefunc_neighbors_both(f, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode)
    if ~isfield(f,'kind') || string(f.kind) ~= "frame"
        neigh = f([]); return;
    end
    neigh = unique_onefuncs([neighbors_Ng_onefunc(f,sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode), ...
                             neighbors_Ngplus_onefunc(f,sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode)]);
end

function neigh = neighbors_Ng_onefunc(f, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode)
    target = center_of_basis(f,cscale,centerForNeighbors);
    Z0 = generate_neighbor_set(neighborMode);
    neigh = repmat(empty_onefunc(),0,1);
    switch char(string(f.type))
        case 'phi'
            for k=1:size(Z0,1)
                center = target + (1/cscale)*Z0(k,:);
                neigh(end+1)=make_phi_at_center(sigma0,cscale,0,center,D); %#ok<AGROW>
            end
        case 'psi'
            l=f.level;
            for k=1:size(Z0,1)
                center = target + (1/(cscale*2^(l+1)))*Z0(k,:);
                cand = make_psi_at_center(sigma0,cscale,l,center,D,ZsetMode,centerConvention);
                if ~isempty(cand), neigh(end+1)=cand; end %#ok<AGROW>
            end
    end
    neigh = unique_onefuncs(neigh);
end

function neigh = neighbors_Ngplus_onefunc(f, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode)
    target = center_of_basis(f,cscale,centerForNeighbors);
    Z0 = generate_neighbor_set(neighborMode);
    neigh = repmat(empty_onefunc(),0,1);
    switch char(string(f.type))
        case 'phi'
            l = 0;
            for k=1:size(Z0,1)
                center = target + (1/(cscale*2))*Z0(k,:);
                cand = make_psi_at_center(sigma0,cscale,l,center,D,ZsetMode,centerConvention);
                if ~isempty(cand), neigh(end+1)=cand; end %#ok<AGROW>
            end
        case 'psi'
            l = f.level + 1;
            for k=1:size(Z0,1)
                center = target + (1/(cscale*2^(f.level+2)))*Z0(k,:);
                cand = make_psi_at_center(sigma0,cscale,l,center,D,ZsetMode,centerConvention);
                if ~isempty(cand), neigh(end+1)=cand; end %#ok<AGROW>
            end
    end
    neigh = unique_onefuncs(neigh);
end

function Z0 = generate_neighbor_set(mode)
    switch lower(mode)
        case 'cross'
            Z0 = [0 0 0; 1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1];
        case 'full'
            vals=-1:1; [A,B,C]=ndgrid(vals,vals,vals); Z0=[A(:),B(:),C(:)];
        otherwise
            error('Unknown neighborMode.');
    end
end

function Zset = generate_z_set(D, mode)
    switch lower(mode)
        case 'signed'
            vals=-1:1; [A,B,C]=ndgrid(vals,vals,vals); Zset=[A(:),B(:),C(:)]; Zset=Zset(any(Zset~=0,2),:);
        case 'binary'
            Zset = dec2bin(0:(2^D-1))-'0'; Zset=Zset(any(Zset~=0,2),:);
        otherwise
            error('Unknown ZsetMode.');
    end
end

function f = make_phi(sigma0,cscale,l,j,D)
    scale = cscale*2^l; width = sigma0/scale; alpha = 1/width^2; center = j/scale;
    term = struct('coef',1.0,'alpha',alpha,'center',center);
    f = empty_onefunc(); f.kind='frame'; f.type='phi'; f.level=l; f.j=round(j); f.z=zeros(1,D); f.terms=term;
    f.key = canonical_key_from_center('phi',l,center,cscale);
end

function f = make_phi_at_center(sigma0,cscale,l,center,D)
    scale = cscale*2^l; j = round(center*scale); center2 = j/scale;
    f = make_phi(sigma0,cscale,l,j,D);
    f.terms(1).center = center2;
    f.key = canonical_key_from_center('phi',l,center2,cscale);
end

function f = make_psi(sigma0,cscale,l,j,z,D,centerConvention)
    gamma = 2^(-D/2); Cpsi = (1 - (16/25)*gamma*sqrt(5) + gamma^2)^(-1/2);
    scale = cscale*2^l;
    switch lower(centerConvention)
        case 'eq428', center = (j + 0.5*z)/scale;
        case 'cmap', center = (j - 0.5*z)/scale;
        otherwise, error('Unknown centerConvention.');
    end
    width1 = (sigma0/2)/scale; width2 = sigma0/scale;
    term1 = struct('coef',Cpsi,'alpha',1/width1^2,'center',center);
    term2 = struct('coef',-Cpsi*gamma,'alpha',1/width2^2,'center',center);
    f = empty_onefunc(); f.kind='frame'; f.type='psi'; f.level=l; f.j=round(j); f.z=round(z); f.terms=[term1,term2];
    f.key = canonical_key_from_center('psi',l,center,cscale);
end

function f = make_psi_at_center(sigma0,cscale,l,center,D,ZsetMode,centerConvention)
    scale = cscale*2^l;
    m = round(2*scale*center);
    z = zeros(1,D); j = zeros(1,D);
    switch lower(ZsetMode)
        case 'signed'
            for d=1:D
                if mod(abs(m(d)),2)==0
                    z(d)=0; j(d)=m(d)/2;
                else
                    z(d)=sign(m(d)); if z(d)==0, z(d)=1; end
                    j(d)=(m(d)-z(d))/2;
                end
            end
        case 'binary'
            for d=1:D
                if mod(abs(m(d)),2)==0
                    z(d)=0; j(d)=m(d)/2;
                else
                    z(d)=1; j(d)=(m(d)-1)/2;
                end
            end
        otherwise
            error('Unknown ZsetMode.');
    end
    if all(z==0), f=[]; return; end
    f = make_psi(sigma0,cscale,l,round(j),round(z),D,centerConvention);
    f.key = canonical_key_from_center('psi',l,f.terms(1).center,cscale);
end

function key = canonical_key_from_center(type, level, center, cscale)
    sc = max(1, round(cscale*2^max(level,0)*2));
    ci = round(center * sc * 1e10)/1e10;
    key = string(sprintf('%s_L%d_C%.10g_%.10g_%.10g', type, level, ci(1), ci(2), ci(3)));
end

function C = center_of_basis(f,cscale,mode)
    switch lower(mode)
        case 'actual'
            if isfield(f,'terms') && ~isempty(f.terms), C=f.terms(1).center; else, C=[0 0 0]; end
        case 'cmap'
            if strcmp(f.type,'phi'), C=f.j/cscale; else, C=(f.j-0.5*f.z)/(cscale*2^f.level); end
        otherwise
            error('Unknown center mode.');
    end
end

function B = unique_onefuncs(B)
    if isempty(B), return; end
    keys = strings(numel(B),1);
    for i=1:numel(B), keys(i)=string(B(i).key); end
    [~,ia]=unique(keys,'stable'); B=B(ia);
end

%% ========================================================================
% One-particle cache and Li matrix assembly
% ========================================================================
function oneCache = reset_one_cache(Zcharge,Rnuc)
    oneCache = struct('Z',Zcharge,'Rnuc',Rnuc,'S1',[],'T1',[],'Ven1',[],'H1',[],'n',0);
end

function oneCache = update_one_cache(oneCache, oneFuncs, Zcharge, Rnuc)
    nOld = oneCache.n; nNew = numel(oneFuncs);
    if nNew <= nOld, return; end
    S1=zeros(nNew); T1=zeros(nNew); Ven1=zeros(nNew); H1=zeros(nNew);
    if nOld>0
        S1(1:nOld,1:nOld)=oneCache.S1; T1(1:nOld,1:nOld)=oneCache.T1;
        Ven1(1:nOld,1:nOld)=oneCache.Ven1; H1(1:nOld,1:nOld)=oneCache.H1;
    end
    for i=1:nNew
        jStart = 1;
        if i<=nOld, jStart=nOld+1; end
        for j=jStart:nNew
            [s,t,v]=contracted_one_particle(oneFuncs(i),oneFuncs(j),Zcharge,Rnuc);
            S1(i,j)=s; S1(j,i)=s; T1(i,j)=t; T1(j,i)=t; Ven1(i,j)=v; Ven1(j,i)=v; H1(i,j)=t+v; H1(j,i)=t+v;
        end
    end
    oneCache.S1=S1; oneCache.T1=T1; oneCache.Ven1=Ven1; oneCache.H1=H1; oneCache.n=nNew;
end

function [S,H,info] = assemble_li_matrices(B, oneCache, oneFuncs, opts)
    M = numel(B); S1=oneCache.S1; H1=oneCache.H1; td=build_term_data(oneFuncs);
    normInv = li_norm_inv(B,S1,opts);
    rowBlock = opts.rowBlockSize; blocks = 1:rowBlock:M;
    S = sparse(M,M); H = sparse(M,M); t0=tic;
    if strcmpi(opts.mode,'cpu_parfor_sparse') && opts.startPool
        try
            pool = gcp('nocreate'); if isempty(pool), parpool; end 
        catch ME
            warning('Could not start parallel pool, continuing serial/parfor fallback: %s', ME.message);
        end
    end
    for ib=1:numel(blocks)
        rows = blocks(ib):min(M,blocks(ib)+rowBlock-1);
        [Sr,Hr] = assemble_li_sparse_rows(B,rows,S1,H1,td,normInv,opts);
        S(rows,:) = S(rows,:) + Sr;
        H(rows,:) = H(rows,:) + Hr;
        if opts.showRowProgress
            fprintf('    Li sparse rows %d-%d / %d inserted, nnz(S/H)=%.3e/%.3e, elapsed %.1fs\n', ...
                rows(1), rows(end), M, nnz(S), nnz(H), toc(t0));
        end
    end
    if isfield(opts,'upperTriangle') && opts.upperTriangle
        S = S + triu(S,1).'; H = H + triu(H,1).';
    end
    S = 0.5*(S+S'); H = 0.5*(H+H');
    info = struct('nnzS',nnz(S),'nnzH',nnz(H),'M',M,'upperTriangle',opts.upperTriangle);
end

function normInv = li_norm_inv(B,S1,opts)
    M=numel(B); mode='l2_normalized'; if isfield(opts,'detScaleMode'), mode=lower(char(string(opts.detScaleMode))); end
    switch mode
        case {'l2_normalized','normalized'}
            normInv=zeros(1,M);
            for i=1:M
                a=B(i).a; b=B(i).b; c=B(i).c;
                da = S1(a,a)*S1(b,b)-S1(a,b)*S1(b,a);
                nb = S1(c,c);
                n2 = da*nb;
                if n2 < 1e-14, n2=1e-14; end
                normInv(i)=1/sqrt(n2);
            end
        case {'antisymmetrizer','raw'}
            normInv=ones(1,M);
        otherwise
            error('Unknown detScaleMode: %s', mode);
    end
end

function [Srows,Hrows] = assemble_li_sparse_rows(B,rowIdx,S1,H1,td,normInv,opts)
    nr=numel(rowIdx); M=numel(B);
    avec=[B.a]; bvec=[B.b]; cvec=[B.c];
    switch lower(opts.mode)
        case 'cpu_parfor_sparse'
            IScell=cell(nr,1); JScell=cell(nr,1); VScell=cell(nr,1);
            IHcell=cell(nr,1); JHcell=cell(nr,1); VHcell=cell(nr,1);
            parfor ir=1:nr
                [js,vs,jh,vh]=assemble_one_li_row(B,rowIdx(ir),avec,bvec,cvec,S1,H1,td,normInv,opts);
                IScell{ir}=ir*ones(numel(js),1); JScell{ir}=js(:); VScell{ir}=vs(:);
                IHcell{ir}=ir*ones(numel(jh),1); JHcell{ir}=jh(:); VHcell{ir}=vh(:);
            end
            IS=vertcat(IScell{:}); JS=vertcat(JScell{:}); VS=vertcat(VScell{:});
            IH=vertcat(IHcell{:}); JH=vertcat(JHcell{:}); VH=vertcat(VHcell{:});
        otherwise
            IS=[];JS=[];VS=[];IH=[];JH=[];VH=[];
            for ir=1:nr
                [js,vs,jh,vh]=assemble_one_li_row(B,rowIdx(ir),avec,bvec,cvec,S1,H1,td,normInv,opts);
                IS=[IS; ir*ones(numel(js),1)]; JS=[JS; js(:)]; VS=[VS; vs(:)]; %#ok<AGROW>
                IH=[IH; ir*ones(numel(jh),1)]; JH=[JH; jh(:)]; VH=[VH; vh(:)]; %#ok<AGROW>
            end
    end
    Srows=sparse(IS,JS,VS,nr,M); Hrows=sparse(IH,JH,VH,nr,M);
end

function [js,vs,jh,vh] = assemble_one_li_row(B,row,avec,bvec,cvec,S1,H1,td,normInv,opts)
    a=B(row).a; b=B(row).b; c=B(row).c;
    M = numel(avec);
    d=avec; e=bvec; f=cvec;
    if isfield(opts,'upperTriangle') && opts.upperTriangle
        upperMask = (1:M) >= row;
    else
        upperMask = true(1,M);
    end

    Sad=S1(a,d); Sbe=S1(b,e); Sae=S1(a,e); Sbd=S1(b,d); Scf=S1(c,f);
    Had=H1(a,d); Hbe=H1(b,e); Hae=H1(a,e); Hbd=H1(b,d); Hcf=H1(c,f);

    Dalpha = Sad.*Sbe - Sae.*Sbd;
    Sraw = Dalpha .* Scf;
    HoneAlpha = (Had.*Sbe + Sad.*Hbe - Hae.*Sbd - Sae.*Hbd) .* Scf;
    HoneBeta  = Dalpha .* Hcf;
    Hcheap = HoneAlpha + HoneBeta;

    doScreen = isfield(opts,'screenERIByOverlap') && opts.screenERIByOverlap;
    if doScreen
        tolSpre = get_opt(opts,'screenTolSForERI',1e-12);
        tolHpre = get_opt(opts,'screenTolHCheapForERI',1e-10);
        preMask = upperMask & (abs(Sraw) > tolSpre | abs(Hcheap) > tolHpre);
        if isfield(opts,'forceDiagonal') && opts.forceDiagonal && row <= M, preMask(row) = true; end
        idx = find(preMask);
    else
        idx = find(upperMask);
    end

    Hraw = Hcheap;
    if ~isempty(idx)
        ds=d(idx); es=e(idx); fs=f(idx);
        Sad_i=Sad(idx); Sbe_i=Sbe(idx); Sae_i=Sae(idx); Sbd_i=Sbd(idx); Scf_i=Scf(idx);
        Vaa_i = (eri_row_vectorized_cpu(td,a,ds,b,es) - eri_row_vectorized_cpu(td,a,es,b,ds)) .* Scf_i;
        Vab_i = Sbe_i .* eri_row_vectorized_cpu(td,a,ds,c,fs) ...
            - Sbd_i .* eri_row_vectorized_cpu(td,a,es,c,fs) ...
            - Sae_i .* eri_row_vectorized_cpu(td,b,ds,c,fs) ...
            + Sad_i .* eri_row_vectorized_cpu(td,b,es,c,fs);
        Hraw(idx) = Hraw(idx) + Vaa_i + Vab_i;
    end

    fac = normInv(row) .* normInv;
    srow = Sraw .* fac; hrow = Hraw .* fac;
    maskS = upperMask & (abs(srow) >= opts.dropTolS);
    maskH = upperMask & (abs(hrow) >= opts.dropTolH);
    if isfield(opts,'forceDiagonal') && opts.forceDiagonal && row <= M
        maskS(row) = true; maskH(row) = true;
    end
    js=find(maskS); vs=srow(maskS);
    jh=find(maskH); vh=hrow(maskH);
end

function val = get_opt(opts, name, defaultVal)
    if isfield(opts,name), val = opts.(name); else, val = defaultVal; end
end

function td = build_term_data(oneFuncs)
    M=numel(oneFuncs); maxT=0;
    for i=1:M, maxT=max(maxT,numel(oneFuncs(i).terms)); end
    coef=zeros(M,maxT); alpha=zeros(M,maxT); cx=zeros(M,maxT); cy=zeros(M,maxT); cz=zeros(M,maxT);
    for i=1:M
        for t=1:numel(oneFuncs(i).terms)
            term=oneFuncs(i).terms(t);
            coef(i,t)=term.coef; alpha(i,t)=term.alpha; cx(i,t)=term.center(1); cy(i,t)=term.center(2); cz(i,t)=term.center(3);
        end
    end
    td=struct('coef',coef,'alpha',alpha,'cx',cx,'cy',cy,'cz',cz,'maxTerms',maxT);
end

function vrow = eri_row_vectorized_cpu(td, ia, ibVec, ic, idVec)
    nc=numel(ibVec); vrow=zeros(1,nc); T=td.maxTerms;
    for ta=1:T
        ca=td.coef(ia,ta); if ca==0, continue; end
        aA=td.alpha(ia,ta); Ax=td.cx(ia,ta); Ay=td.cy(ia,ta); Az=td.cz(ia,ta);
        for tc=1:T
            cc=td.coef(ic,tc); if cc==0, continue; end
            aC=td.alpha(ic,tc); Cx=td.cx(ic,tc); Cy=td.cy(ic,tc); Cz=td.cz(ic,tc);
            coefAC=ca*cc;
            for tb=1:T
                cb=td.coef(ibVec,tb).'; activeB=cb~=0; if ~any(activeB), continue; end
                aB_all=td.alpha(ibVec,tb).'; Bx_all=td.cx(ibVec,tb).'; By_all=td.cy(ibVec,tb).'; Bz_all=td.cz(ibVec,tb).';
                for td2=1:T
                    cd=td.coef(idVec,td2).'; mask=activeB & (cd~=0); if ~any(mask), continue; end
                    aB=aB_all(mask); Bx=Bx_all(mask); By=By_all(mask); Bz=Bz_all(mask);
                    idm=idVec(mask);
                    aD=td.alpha(idm,td2).'; Dx=td.cx(idm,td2).'; Dy=td.cy(idm,td2).'; Dz=td.cz(idm,td2).';
                    val=eri_primitive_vec(aA,Ax,Ay,Az,aB,Bx,By,Bz,aC,Cx,Cy,Cz,aD,Dx,Dy,Dz);
                    vrow(mask)=vrow(mask)+coefAC.*cb(mask).*cd(mask).*val;
                end
            end
        end
    end
end

function [S,T,Ven] = contracted_one_particle(f,g,Z,Rnuc)
    S=0; T=0; Ven=0;
    for a=1:numel(f.terms)
        A=f.terms(a);
        for b=1:numel(g.terms)
            B=g.terms(b);
            [s,t,v]=one_particle_primitive(A.alpha,A.center,B.alpha,B.center,Z,Rnuc);
            coef=A.coef*B.coef; S=S+coef*s; T=T+coef*t; Ven=Ven+coef*v;
        end
    end
end

function [S,T,Ven] = one_particle_primitive(alphaA,A,alphaB,B,Z,Rnuc)
    p=alphaA+alphaB; P=(alphaA*A+alphaB*B)./p; RAB2=sum((A-B).^2);
    NA=(alphaA/pi)^(3/4); NB=(alphaB/pi)^(3/4);
    K=NA*NB*exp(-(alphaA*alphaB/(2*p))*RAB2);
    S=K*(2*pi/p)^(3/2);
    T=0.5*alphaA*alphaB*(3/p-(alphaA*alphaB/p^2)*RAB2)*S;
    RP2=sum((P-Rnuc).^2);
    Ven=-Z*K*(4*pi/p)*boys0(0.5*p*RP2);
end

function val = eri_primitive_vec(alphaA,Ax,Ay,Az,alphaB,Bx,By,Bz,alphaC,Cx,Cy,Cz,alphaD,Dx,Dy,Dz)
    p=alphaA+alphaB; q=alphaC+alphaD;
    Px=(alphaA*Ax+alphaB.*Bx)./p; Py=(alphaA*Ay+alphaB.*By)./p; Pz=(alphaA*Az+alphaB.*Bz)./p;
    Qx=(alphaC*Cx+alphaD.*Dx)./q; Qy=(alphaC*Cy+alphaD.*Dy)./q; Qz=(alphaC*Cz+alphaD.*Dz)./q;
    RAB2=(Ax-Bx).^2+(Ay-By).^2+(Az-Bz).^2;
    RCD2=(Cx-Dx).^2+(Cy-Dy).^2+(Cz-Dz).^2;
    RPQ2=(Px-Qx).^2+(Py-Qy).^2+(Pz-Qz).^2;
    NA=(alphaA/pi)^(3/4); NB=(alphaB/pi).^(3/4); NC=(alphaC/pi)^(3/4); ND=(alphaD/pi).^(3/4);
    Kab=NA.*NB.*exp(-(alphaA.*alphaB./(2*p)).*RAB2);
    Kcd=NC.*ND.*exp(-(alphaC.*alphaD./(2*q)).*RCD2);
    arg=(p.*q./(2*(p+q))).*RPQ2;
    val=Kab.*Kcd.*(8*sqrt(2)*pi^(5/2))./(p.*q.*sqrt(p+q)).*boys0_vec(arg);
end

function F = boys0_vec(t)
    F=zeros(size(t)); small=t<1e-10;
    F(small)=1-t(small)/3+t(small).^2/10;
    ts=t(~small); F(~small)=0.5*sqrt(pi).*erf(sqrt(ts))./sqrt(ts);
end

function F = boys0(t)
    if t<1e-10, F=1-t/3+t^2/10; else, F=0.5*sqrt(pi)*erf(sqrt(t))/sqrt(t); end
end

%% ========================================================================
% Solver, importance, diagnostics
% ========================================================================
function [E0,coeff,nkeep,info] = solve_ground_projected(H,S,mass_tol,verbose,targetEnergy,solverOpts)
    M = size(S,1);
    if M > solverOpts.projectedDenseLimit
        error('M=%d exceeds projectedDenseLimit=%d. Increase limit or implement matrix-free solver.', M, solverOpts.projectedDenseLimit);
    end
    H=full(0.5*(H+H')); S=full(0.5*(S+S'));
    [U,D]=eig(S); d=real(diag(D)); [d,ord]=sort(d,'descend'); U=U(:,ord);
    keep=d>mass_tol*max(d);
    if ~any(keep), error('Mass matrix projection removed all dimensions.'); end
    Uk=U(:,keep); dk=d(keep); X=Uk*diag(1./sqrt(dk));
    Hp=X'*H*X; Hp=0.5*(Hp+Hp');
    [Y,Ediag]=eig(Hp); evals=real(diag(Ediag)); [E0,pos]=min(evals);
    coeff=X*Y(:,pos); coeff=coeff/sqrt(real(coeff'*S*coeff)); nkeep=sum(keep);
    res=norm(H*coeff - E0*S*coeff)/max(1,norm(H*coeff));
    info=struct('residual',res,'nkeep',nkeep,'targetEnergy',targetEnergy,'method','dense_mass_projection');
    if verbose
        fprintf('    projected solve: nkeep=%d/%d, E=%.15f, residual=%.3e\n', nkeep, M, E0, res);
    end
end

function scores = partial_orthogonal_importance(c,S)
    M=numel(c);
    if M==1, scores=abs(c); return; end
    B11=S(1,1); B12=S(1,2:end);
    z=c; z(1)=c(1)+(B12/B11)*c(2:end);
    diagPO=zeros(M,1); diagPO(1)=B11;
    sdiag=diag(S);
    diagPO(2:end)=sdiag(2:end)-(B12(:).^2)/B11;
    diagPO=max(real(diagPO),realmin);
    scores=abs(sqrt(diagPO).*z);
end

function be = compute_block_energy_contributions(c,H,B)
    be = empty_block_energy();
    blocks = strings(numel(B),1); for i=1:numel(B), blocks(i)=string(B(i).block); end
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
    be.diag.total = be.diag.zero+be.diag.one+be.diag.two_ud+be.diag.two_uu+be.diag.three;
end

function be = empty_block_energy()
    fields = {'zero','one','two_ud','two_uu','three','total'};
    for k=1:numel(fields), row.(fields{k})=0; diagv.(fields{k})=0; end 
    be=struct('row',row,'diag',diagv);
end

function e = row_energy(c,H,idx)
    if ~any(idx), e=0; else, e=real(c(idx)'*(H(idx,:)*c)); end
end

function e = diag_energy(c,H,idx)
    if ~any(idx), e=0; else, e=real(c(idx)'*(H(idx,idx)*c(idx))); end
end

function counts = count_blocks(B)
    M=numel(B); blocks=strings(M,1);
    for i=1:M, blocks(i)=string(B(i).block); end
    counts=struct();
    counts.zero=nnz(blocks=="zero");
    counts.one_total=nnz(blocks=="one");
    counts.two_ud=nnz(blocks=="two_ud");
    counts.two_uu=nnz(blocks=="two_uu");
    counts.three=nnz(blocks=="three");
    counts.total=M;
end

function gb = estimate_sparse_pair_gb(S,H)
    ns=nnz(S); nh=nnz(H); m=size(S,1);
    gb=(16*(ns+nh)+8*2*(m+1))/1024^3;
end
