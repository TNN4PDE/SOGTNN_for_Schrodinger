%% JH_BeGround_part3_adaptive_v5_reusePairCache_groupedERI_Table510.m
% Be ground-state adaptive scheme for J.H. Table 5.10 reproduction.
% v5: incremental matrix reuse + reusable alpha/beta pair cache + grouped-term ERI kernel.
%     This version targets the remaining Be bottleneck: repeated rebuilding of
%     pair transition matrices and padded primitive loops in V_ab.
%     Removes the main direct-newrows redundancy: repeated same-spin/one-body
%     recomputation for appended Be determinants.
%
% Key speed-up over v1:
%   If B_new = [B_old, newly grown functions], reuse S_old/H_old and assemble
%   only the new-row block against all columns.  If the basis is unchanged
%   across kappa, reuse the full previous matrix with zero assembly cost.
%
% This is Part 3 after:
%   Part 1: JH_BeGround_HF_Q9_orbitals.mat or compatible RHF-SCF-DIIS HF file
%   Part 2: JH_BeGround_BthreeOther0_coarsened_*_nu*.mat candidate files
%
% Main design:
%   - N = 4, M_S = 0, N_up=2, N_down=2.
%   - Basis element represents
%       A_alpha[f_a(x1) f_b(x2)] * A_beta[f_c(x3) f_d(x4)]
%     with L2-normalized alpha and beta determinants.
%   - J.H. Be-specific behavior: keep B_other^[0] and, by default, do not
%     grow same-spin/triple/other blocks.  This follows Table 5.10 where
%     M_two_uu, M_three, and M_other stay fixed while one/two_ud dominate.
%   - Assembly exploits Be structure by caching alpha-pair and beta-pair
%     transition blocks; only the opposite-spin Coulomb cofactor contraction
%     remains determinant-row dependent.
%
% Output format follows Adaptive_Li: records, recordsByM, CSV, basis snapshots,
% live records, checkpoints, and finalState .mat files.

clear; clc; close all;
format long e;

%% ============================================================
% 0. User-facing controls / API
% =============================================================
initMode = 'nuList';
targetNuList = [15];
targetCoarseFiles = { ...
    % 'JH_BeGround_BthreeOther0_coarsened_fromExistingRaw_M1163_nu12_candidate.mat'
    % 'JH_BeGround_BthreeOther0_coarsened_fromExistingRaw_M1633_nu13_candidate.mat'
    % 'JH_BeGround_BthreeOther0_coarsened_fromExistingRaw_M2229_nu14_candidate.mat'
};
runAllInitializations = true;

% Adaptive controls.  First debug can use kappa_max=4, max_inner=2.
kappa_max = 10;
max_inner = 3;
max_basis_allowed = 25000;
updateMode = 'monotone_union';
importanceMode = 'partial_orthogonal';

% Be-specific growth.  Table 5.10 keeps M_two_uu, M_three and M_other fixed;
% grow only one-particle and opposite-spin two-particle replacement blocks.
growthPolicy = struct();
growthPolicy.growOne = true;
growthPolicy.growTwoUD = true;
growthPolicy.growTwoUU = false;
growthPolicy.growThree = false;
growthPolicy.growOther = false;
growthPolicy.keepUngrownImportant = true;

% Exact references / diagnostics from J.H. tables.
E_ref_exact = -14.66736;
E_ref_tilde_table = -14.659783;   % Table 5.10 final reported Be value, kappa=9
E_ref_HF_table = -14.57302;

% Solver and assembly controls.
mass_tol = 1e-10;
verboseEig = false;
useInitialMatricesWhenAvailable = false;

assemblyOpts = struct();
assemblyOpts.mode = 'cpu_parfor_sparse';
assemblyOpts.rowBlockSize = 512;       % stable default; 256 if RAM/parpool workers are enough
assemblyOpts.startPool = true;
assemblyOpts.dropTolS = 1e-13;
assemblyOpts.dropTolH = 1e-12;
assemblyOpts.screenERIByOverlap = false;
assemblyOpts.screenTolSForERI = 1e-13;
assemblyOpts.screenTolHCheapForERI = 1e-12;
assemblyOpts.eriCoeffTol = 1e-13;      % skip numerically zero cofactor ERI terms; set 0 for strict audit
assemblyOpts.forceDiagonal = true;
assemblyOpts.showRowProgress = false;
assemblyOpts.progressEveryBlocks = 4;
assemblyOpts.upperTriangle = true;
assemblyOpts.detScaleMode = 'l2_normalized';
assemblyOpts.usePairCache = true;
assemblyOpts.useIncrementalAssembly = false;
assemblyOpts.keepLastMatrixInMemory = false;
assemblyOpts.maxCachedMatrixM = 15000;      % user stated M<1.5e4; raise if RAM permits
assemblyOpts.saveIncrementalMatrixCache = false; % disk cache is slower; in-memory reuse is default
assemblyOpts.incrementalBackend = 'auto_fast'; % auto: GPU paircached if available, otherwise CPU paircached
assemblyOpts.preferGPU = false;                % default CPU-parfor is often faster than row-wise GPU on RTX 3050; set true for audit
assemblyOpts.gpuDeviceID = 1;
assemblyOpts.gpuMinM = 3000;                    % avoid GPU overhead for tiny matrices
assemblyOpts.gpuMinNewRows = 8;
assemblyOpts.gpuRowProgressEvery = 25;
assemblyOpts.gpuGatherEveryRow = true;          % conservative memory behavior for 4GB GPUs
assemblyOpts.vabBatchMode = 'batched4x_grouped';  % batched 4x + group columns by true primitive term counts
assemblyOpts.vabChunkSizeCPU = 20000;             % CPU chunk length in columns for grouped batched Vab
assemblyOpts.vabChunkSizeGPU = 4096;              % GPU chunk length; safer for RTX 3050 4GB
assemblyOpts.reusePairCache = true;                % reuse/update alpha/beta pair cache across inner/kappa
assemblyOpts.groupERIByTermCount = true;           % avoid padding all columns to maxTerms^2

solverOpts = struct();
solverOpts.projectedDenseLimit = 20000;
solverOpts.rejectBelowExact = true;
solverOpts.energyLowerTol = 5e-4;
solverOpts.exactLowerBound = E_ref_exact;

recordOpts = struct();
recordOpts.computeBlockEnergyEachInner = false;   % turn on only for final table diagnostics
recordOpts.saveRecordsEveryInner = true;
recordOpts.saveBasisEveryMRecord = true;
recordOpts.savePrefix = 'JH_BeGround_Table510';

checkpointOpts = struct();
checkpointOpts.saveBasisOnly = true;
checkpointOpts.dropMatrixAfterEachInner = true;  % local S/H can be cleared; matrixCache keeps one sparse copy

innerStop = struct();
innerStop.enable = false;
innerStop.minInner = 2;
innerStop.minAdded = 5000;
innerStop.minGrowthRel = 0.35;
innerStop.minAbsGain = 5e-7;
innerStop.minRelGain = 0.01;

%% ============================================================
% 1. Discover and run selected initial spaces
% =============================================================
coarseFiles = discover_be_coarse_files(initMode, targetNuList, targetCoarseFiles);
if isempty(coarseFiles)
    error('No Be coarsened initial files found. Save nu=12/13/14 candidates first.');
end
if ~runAllInitializations
    coarseFiles = coarseFiles(1);
end

fprintf('\n============================================================\n');
fprintf('J.H. Be Table 5.10 adaptive scheme, multi-initial API\n');
fprintf('initMode=%s | runAllInitializations=%d | kappa_max=%d | max_inner=%d\n', ...
    initMode, runAllInitializations, kappa_max, max_inner);
fprintf('Growth policy: one=%d two_ud=%d two_uu=%d three=%d other=%d\n', ...
    growthPolicy.growOne, growthPolicy.growTwoUD, growthPolicy.growTwoUU, growthPolicy.growThree, growthPolicy.growOther);
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
    fprintf('Starting Be adaptive run %d/%d from %s\n', iRun, numel(coarseFiles), coarseFile);
    fprintf('runTag = %s\n', runTag);
    fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n');

    [records, recordsByM, finalFile] = run_one_be_adaptive( ...
        coarseFile, runTag, kappa_max, max_inner, max_basis_allowed, updateMode, importanceMode, growthPolicy, ...
        E_ref_exact, E_ref_tilde_table, E_ref_HF_table, mass_tol, verboseEig, useInitialMatricesWhenAvailable, ...
        assemblyOpts, solverOpts, recordOpts, checkpointOpts, innerStop);

    allRuns(end+1).coarseFile = string(coarseFile); %#ok<SAGROW>
    allRuns(end).finalFile = string(finalFile);
    allRuns(end).records = records;
    allRuns(end).recordsByM = recordsByM;
end

save('JH_BeGround_Table510_adaptive_multiInit_summary.mat','allRuns','-v7.3');
fprintf('\nAll Be adaptive runs finished. Summary saved: JH_BeGround_Table510_adaptive_multiInit_summary.mat\n');

%% ========================================================================
% Main run function
% ========================================================================
function [records, recordsByM, finalFile] = run_one_be_adaptive( ...
    coarseFile, runTag, kappa_max, max_inner, max_basis_allowed, updateMode, importanceMode, growthPolicy, ...
    E_ref_exact, E_ref_tilde_table, E_ref_HF_table, mass_tol, verboseEig, useInitialMatricesWhenAvailable, ...
    assemblyOpts, solverOpts, recordOpts, checkpointOpts, innerStop)

    Sload = load(coarseFile);
    if isfield(Sload,'part3')
        part3 = Sload.part3;
    elseif isfield(Sload,'part3cand')
        part3 = Sload.part3cand;
    else
        error('File %s does not contain part3 or part3cand.', coarseFile);
    end

    hf = part3.hf;
    params = part3.params;
    oneFuncs = part3.oneFuncs(:).';
    basis = part3.basis0(:).';

    % Parameters from coarsening, with robust fallbacks.
    Zcharge = get_struct_default(params,'Zcharge',4);
    Rnuc = get_struct_default(params,'Rnuc',[0 0 0]);
    sigma0 = get_struct_default(params,'sigma0',1);
    cscale = get_struct_default(params,'cscale',0.25);
    D = get_struct_default(params,'D',3);
    ZsetMode = get_struct_default(params,'ZsetMode','signed');
    centerConvention = get_struct_default(params,'centerConvention','eq428');
    centerForNeighbors = get_struct_default(params,'centerForNeighbors','actual');
    neighborMode = get_struct_default(params,'neighborMode','cross');
    assemblyOpts.detScaleMode = get_struct_default(params,'detScaleMode',assemblyOpts.detScaleMode);

    oneKeyMap = containers.Map('KeyType','char','ValueType','double');
    for ii = 1:numel(oneFuncs), oneKeyMap(char(oneFuncs(ii).key)) = ii; end

    oneCache = reset_one_cache(Zcharge, Rnuc);
    records = repmat(empty_record(), kappa_max, 1);
    recordsByM = repmat(empty_m_record(), 0, 1);
    sampleID = 0;
    tGlobal = tic;

    % Incremental matrix cache.  This is the main Be acceleration path:
    % kappa=1 inner=1 loads S0/H0 from the coarse candidate; subsequent
    % inner/kappa steps reuse the top-left block and assemble only rows for
    % newly added basis functions.
    matrixCache = empty_matrix_cache();

    fprintf('Loaded initial Be space: %s\n', coarseFile);
    c0 = count_blocks(basis);
    if isfield(part3,'E0'), Einit = part3.E0; else, Einit = NaN; end
    fprintf('Initial B^[0]: M=%d | Mone=%d | Mtwo_ud=%d | Mtwo_uu=%d | Mthree=%d | Mother=%d | E0(coarse)=%.15f\n', ...
        c0.total, c0.one_total, c0.two_ud, c0.two_uu, c0.three, c0.other, Einit);
    fprintf('J.H. Table 5.10 target rows: kappa=1 M=1417 E=-14.644213; kappa=9 M=24775 E=-14.659783.\n');
    fprintf('Be-specific note: Table 5.10 keeps Mtwo_uu=264, Mthree=64, Mother=2 fixed; default growth only one/two_ud.\n');
    fprintf('Parameters: sigma=%g, c=%g, neighbor=%s, update=%s, importance=%s, pairCache=%d\n\n', ...
        sigma0, cscale, neighborMode, updateMode, importanceMode, assemblyOpts.usePairCache);

    initialMatrixAvailable = useInitialMatricesWhenAvailable && isfield(part3,'S0') && isfield(part3,'H0') && ...
        size(part3.S0,1)==numel(basis) && size(part3.H0,1)==numel(basis);

    for kappa = 1:kappa_max
        epsK = 2^(-kappa);
        fprintf('\n############################################################\n');
        fprintf('Be adaptive kappa = %d, epsilon = %.6e, runTag=%s\n', kappa, epsK, runTag);
        fprintf('############################################################\n');

        finalInfo = struct();
        changed = false;
        prevAccepted = struct('valid',false);

        for inner = 1:max_inner
            tInner = tic;
            M = numel(basis);
            counts = count_blocks(basis);
            fprintf('\n  inner %d | M=%d | Mone/two_ud/two_uu/three/other = %d / %d / %d / %d / %d\n', ...
                inner, M, counts.one_total, counts.two_ud, counts.two_uu, counts.three, counts.other);

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

            cacheStatus = 'none';
            if kappa==1 && inner==1 && initialMatrixAvailable
                S = sparse(part3.S0); H = sparse(part3.H0);
                asmInfo = struct('backend','loaded_initial_S0_H0','M',M,'nnzS',nnz(S),'nnzH',nnz(H)); %#ok<NASGU>
                cacheStatus = 'loaded initial S0/H0 from coarse candidate';
            elseif get_opt(assemblyOpts,'useIncrementalAssembly',false) && matrixCache.valid
                [canReuse, nOld, whyReuse] = matrix_cache_prefix_match(matrixCache, basis);
                if canReuse && nOld == M
                    S = matrixCache.S; H = matrixCache.H;
                    asmInfo = struct('backend','reused_cached_full_matrix','M',M,'nnzS',nnz(S),'nnzH',nnz(H)); %#ok<NASGU>
                    cacheStatus = sprintf('reused full cached S/H (%s)', whyReuse);
                elseif canReuse && nOld < M
                    [S,H,asmInfo] = assemble_be_matrices_incremental( ...
                        basis, oneCache, oneFuncs, assemblyOpts, matrixCache.S, matrixCache.H, nOld, matrixCache.pairData); %#ok<NASGU>
                    cacheStatus = sprintf('incremental reuse old M=%d -> new M=%d (%s)', nOld, M, whyReuse);
                else
                    [S,H,asmInfo] = assemble_be_matrices(basis, oneCache, oneFuncs, assemblyOpts); %#ok<NASGU>
                    cacheStatus = sprintf('full assembly; cache miss: %s', whyReuse);
                end
            else
                [S,H,asmInfo] = assemble_be_matrices(basis, oneCache, oneFuncs, assemblyOpts); %#ok<NASGU>
                cacheStatus = 'full assembly; no valid cache';
            end
            tAsm = toc(ticAsm);
            fprintf('  assembly: %.2f s | %s | nnz(S/H)=%.3e/%.3e | sparse est %.3f GB\n', ...
                tAsm, cacheStatus, nnz(S), nnz(H), estimate_sparse_pair_gb(S,H));

            nnzS = nnz(S); nnzH = nnz(H); sparseGB = estimate_sparse_pair_gb(S,H);

            if get_opt(assemblyOpts,'keepLastMatrixInMemory',true) && M <= get_opt(assemblyOpts,'maxCachedMatrixM',15000)
                matrixCache = update_matrix_cache(matrixCache, basis, S, H, kappa, inner, cacheStatus, asmInfo);
            else
                matrixCache = empty_matrix_cache();
            end

            ticSol = tic;
            [E0, coeff, nkeep, solverInfo] = solve_ground_projected(H, S, mass_tol, verboseEig, E_ref_exact);
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
            fprintf('  row energy: zero=%.6f one=%.6f two_ud=%.6f two_uu=%.3e three=%.3e other=%.3e sum=%.15f\n', ...
                blockEnergy.row.zero, blockEnergy.row.one, blockEnergy.row.two_ud, ...
                blockEnergy.row.two_uu, blockEnergy.row.three, blockEnergy.row.other, blockEnergy.row.total);

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
            [growBasis, oneFuncs, oneKeyMap] = grow_important_be_basis( ...
                basis(important), oneFuncs, oneKeyMap, sigma0, cscale, D, ZsetMode, ...
                centerConvention, centerForNeighbors, neighborMode, growthPolicy);
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
            countsNew = count_blocks(basisNew);
            fprintf('  grown raw=%d | unique new M=%d | added=%d | new blocks one/two_ud/two_uu/three/other=%d/%d/%d/%d/%d | grow %.2f s\n', ...
                nGrowRaw, nNewM, nAdded, countsNew.one_total, countsNew.two_ud, countsNew.two_uu, countsNew.three, countsNew.other, tGrow);

            sampleID = sampleID + 1;
            tInnerNoSave = toc(tInner);
            [recordsByM, tSave, ~] = append_be_M_record_and_checkpoint( ...
                recordsByM, sampleID, kappa, inner, epsK, M, counts, E0, err, blockEnergy, ...
                nkeep, solverInfo, tAsm, tSol, tImp, tBE, tGrow, tInnerNoSave, toc(tGlobal), ...
                nImp, scoreMax, scoreMinNonzero, nGrowRaw, nNewM, nAdded, nnzS, nnzH, sparseGB, ...
                oneFuncs, basis, records, recordOpts, runTag, params, importanceMode, updateMode, growthPolicy, mass_tol, assemblyOpts, solverOpts);
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
                initialMatrixAvailable = false;
            end
        end

        if isempty(fieldnames(finalInfo))
            warning('No solved state for kappa=%d; stopping.', kappa);
            records = records(1:kappa-1);
            break;
        end

        records(kappa) = make_kappa_record(kappa, epsK, finalInfo);
        fprintf('\n>>> record kappa=%d | M=%d | E=%.15f | err=%.3e | Mone=%d | Mud=%d | Muu=%d | M3=%d | Mother=%d | inner=%d\n', ...
            kappa, finalInfo.M, finalInfo.E0, finalInfo.err, finalInfo.counts.one_total, ...
            finalInfo.counts.two_ud, finalInfo.counts.two_uu, finalInfo.counts.three, finalInfo.counts.other, finalInfo.inner);

        adaptiveState = struct('coarseFile',coarseFile,'runTag',runTag,'records',records,'recordsByM',recordsByM, ...
            'kappa',kappa,'oneFuncs',oneFuncs,'basis',basis,'params',params, ...
            'importanceMode',importanceMode,'updateMode',updateMode,'growthPolicy',growthPolicy,'mass_tol',mass_tol, ...
            'assemblyOpts',assemblyOpts,'solverOpts',solverOpts,'matrixCacheMeta',matrix_cache_meta(matrixCache));
        save(sprintf('%s_adaptive_checkpoint_%s_kappa%d_M%d.mat', recordOpts.savePrefix, runTag, kappa, numel(basis)), ...
            'adaptiveState','-v7.3');

        if ~changed
            % Continue to next kappa with the same closed basis.
        end
    end

    fprintf('\n==================== Be TABLE 5.10 STYLE SUMMARY: %s ====================\n', runTag);
    fprintf(' kappa      M        E                    err       Mone   Mtwo_ud Mtwo_uu Mthree Mother    Ezero      Eone       Etwo_ud    Etwo_uu    Ethree    Eother\n');
    for k = 1:numel(records)
        r = records(k);
        if isnan(r.kappa), continue; end
        fprintf('%5d  %7d  %.15f  %.3e  %6d %8d %7d %6d %6d  %9.3f %9.3f %10.3f %10.3e %10.3e %10.3e\n', ...
            r.kappa, r.M, r.E, r.err, r.Mone, r.Mtwo_ud, r.Mtwo_uu, r.Mthree, r.Mother, ...
            r.Ezero, r.Eone, r.Etwo_ud, r.Etwo_uu, r.Ethree, r.Eother);
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
        'importanceMode',importanceMode,'updateMode',updateMode,'growthPolicy',growthPolicy,'mass_tol',mass_tol, ...
        'assemblyOpts',assemblyOpts,'solverOpts',solverOpts,'matrixCacheMeta',matrix_cache_meta(matrixCache), 'E_ref_exact',E_ref_exact, ...
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
% Incremental matrix-cache helpers
% ========================================================================
function C = empty_matrix_cache()
    C = struct('valid',false,'M',0,'basisKeys',strings(0,1),'S',[],'H',[], ...
        'pairData',[],'kappa',NaN,'inner',NaN,'status','');
end

function C = update_matrix_cache(C, B, S, H, kappa, inner, status, asmInfo)
    oldPairData = [];
    if isfield(C,'pairData'), oldPairData = C.pairData; end
    C.valid = true;
    C.M = numel(B);
    C.basisKeys = libasis_keys(B);
    C.S = S;
    C.H = H;
    if nargin >= 8 && isstruct(asmInfo) && isfield(asmInfo,'pairDataForCache')
        C.pairData = asmInfo.pairDataForCache;
    else
        C.pairData = oldPairData;
    end
    C.kappa = kappa;
    C.inner = inner;
    C.status = char(string(status));
end

function meta = matrix_cache_meta(C)
    if isempty(C) || ~isfield(C,'valid') || ~C.valid
        meta = struct('valid',false,'M',0,'kappa',NaN,'inner',NaN,'status','');
    else
        meta = struct('valid',true,'M',C.M,'kappa',C.kappa,'inner',C.inner,'status',C.status, ...
            'nnzS',nnz(C.S),'nnzH',nnz(C.H),'hasPairCache',isfield(C,'pairData') && ~isempty(C.pairData));
    end
end

function [ok,nOld,reason] = matrix_cache_prefix_match(C, B)
    ok = false; nOld = 0; reason = 'invalid cache';
    if isempty(C) || ~isfield(C,'valid') || ~C.valid
        return;
    end
    M = numel(B); nOld = C.M;
    if nOld > M
        reason = sprintf('cached M=%d > current M=%d', nOld, M);
        return;
    end
    keys = libasis_keys(B);
    if numel(C.basisKeys) ~= nOld
        reason = 'cached key length mismatch';
        return;
    end
    if all(keys(1:nOld) == C.basisKeys(:))
        ok = true;
        if nOld == M
            reason = 'same basis keys';
        else
            reason = 'old basis is prefix of current basis';
        end
    else
        reason = 'old basis is not a prefix; stable-order assumption failed';
    end
end

%% ========================================================================
% File discovery and records
% ========================================================================
function files = discover_be_coarse_files(initMode, targetNuList, targetCoarseFiles)
    files = struct([]);
    mode = lower(char(string(initMode)));
    switch mode
        case 'explicit'
            for i = 1:numel(targetCoarseFiles)
                f = targetCoarseFiles{i}; if isstring(f), f = char(f); end
                if exist(f,'file')
                    files = append_dir_unique(files, dir(f));
                else
                    warning('Explicit coarse file not found: %s', f);
                end
            end
        case 'nulist'
            all = [dir('JH_BeGround_BthreeOther0_coarsened_fromExistingRaw_M*_nu*_candidate.mat'); ...
                   dir('JH_BeGround_BthreeOther0_coarsened_v4pairFast_M*_nu*.mat'); ...
                   dir('JH_BeGround_BthreeOther0_coarsened_v3termFast_M*_nu*.mat'); ...
                   dir('JH_BeGround_BthreeOther0_coarsened_v2fast_M*_nu*.mat'); ...
                   dir('JH_BeGround_BthreeOther0_coarsened_v1canon_M*_nu*.mat')];
            if isempty(all), return; end
            all = unique_dir_by_name_keep_latest(all);
            for nu = targetNuList(:).'
                hit = struct([]); pat = sprintf('_nu%d', nu);
                for i = 1:numel(all)
                    if contains(all(i).name, pat)
                        hit = append_dir_unique(hit, all(i));
                    end
                end
                if isempty(hit)
                    warning('No Be coarsened file found for nu=%d.', nu);
                    continue;
                end
                isCand = contains({hit.name}, '_candidate.mat');
                if any(isCand)
                    cand = hit(isCand); [~,ord] = max([cand.datenum]); d = cand(ord);
                else
                    [~,ord] = max([hit.datenum]); d = hit(ord);
                end
                files = append_dir_unique(files, d);
            end
        case 'latest'
            all = [dir('JH_BeGround_BthreeOther0_coarsened_fromExistingRaw_M*_nu*_candidate.mat'); ...
                   dir('JH_BeGround_BthreeOther0_coarsened_v4pairFast_M*_nu*.mat'); ...
                   dir('JH_BeGround_BthreeOther0_coarsened_v3termFast_M*_nu*.mat'); ...
                   dir('JH_BeGround_BthreeOther0_coarsened_v2fast_M*_nu*.mat'); ...
                   dir('JH_BeGround_BthreeOther0_coarsened_v1canon_M*_nu*.mat')];
            if isempty(all), return; end
            all = unique_dir_by_name_keep_latest(all);
            [~,ord] = max([all.datenum]); files = all(ord);
        otherwise
            error('Unknown initMode: %s', initMode);
    end
    files = files(:).';
end

function files = append_dir_unique(files, d)
    if isempty(d), return; end
    if isempty(files)
        files = d(:).';
        return;
    end
    names = {files.name};
    for k = 1:numel(d)
        if ~any(strcmp(names, d(k).name))
            files(end+1) = d(k); 
            names{end+1} = d(k).name; 
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

function tag = make_run_tag_from_file(filename, iRun)
    [~,base,~] = fileparts(filename);
    m = regexp(base,'nu(\d+)','tokens','once');
    if ~isempty(m)
        tag = sprintf('nu%s', m{1});
    else
        tag = sprintf('run%d', iRun);
    end
end

function r = empty_record()
    r = struct('kappa',NaN,'epsilon',NaN,'inner',NaN, ...
        'M',NaN,'Mzero',NaN,'Mone',NaN,'Mtwo_ud',NaN,'Mtwo_uu',NaN,'Mthree',NaN,'Mother',NaN, ...
        'E',NaN,'err',NaN,'Ezero',NaN,'Eone',NaN,'Etwo_ud',NaN,'Etwo_uu',NaN,'Ethree',NaN,'Eother',NaN, ...
        'kept',NaN,'solverResidual',NaN,'assemblyTime',NaN,'solveTime',NaN, ...
        'importanceTime',NaN,'blockEnergyTime',NaN);
end

function r = empty_m_record()
    r = struct('sampleID',NaN,'kappa',NaN,'inner',NaN,'epsilon',NaN, ...
        'M',NaN,'Mzero',NaN,'Mone',NaN,'Mtwo_ud',NaN,'Mtwo_uu',NaN,'Mthree',NaN,'Mother',NaN, ...
        'E',NaN,'err',NaN,'Ezero',NaN,'Eone',NaN,'Etwo_ud',NaN,'Etwo_uu',NaN,'Ethree',NaN,'Eother',NaN, ...
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
    r.Mtwo_uu = info.counts.two_uu; r.Mthree = info.counts.three; r.Mother = info.counts.other;
    r.E = info.E0; r.err = info.err; r.inner = info.inner; r.kept = info.nkeep;
    r.solverResidual = info.solverInfo.residual;
    r.Ezero = info.blockEnergy.row.zero; r.Eone = info.blockEnergy.row.one;
    r.Etwo_ud = info.blockEnergy.row.two_ud; r.Etwo_uu = info.blockEnergy.row.two_uu;
    r.Ethree = info.blockEnergy.row.three; r.Eother = info.blockEnergy.row.other;
    r.assemblyTime = info.tAsm; r.solveTime = info.tSol; r.importanceTime = info.tImp; r.blockEnergyTime = info.tBE;
end

function [recordsByM, tSave, basisFile] = append_be_M_record_and_checkpoint( ...
    recordsByM, sampleID, kappa, inner, epsK, M, counts, E0, err, blockEnergy, ...
    nkeep, solverInfo, tAsm, tSol, tImp, tBE, tGrow, tInnerNoSave, tCum, ...
    nImp, scoreMax, scoreMinNonzero, nGrowRaw, nNewM, nAdded, nnzS, nnzH, sparseGB, ...
    oneFuncs, basis, records, recordOpts, runTag, params, importanceMode, updateMode, growthPolicy, mass_tol, assemblyOpts, solverOpts)

    t0 = tic; basisFile = "";
    r = empty_m_record();
    r.sampleID = sampleID; r.kappa = kappa; r.inner = inner; r.epsilon = epsK;
    r.M = M; r.Mzero = counts.zero; r.Mone = counts.one_total; r.Mtwo_ud = counts.two_ud;
    r.Mtwo_uu = counts.two_uu; r.Mthree = counts.three; r.Mother = counts.other;
    r.E = E0; r.err = err;
    r.Ezero = blockEnergy.row.zero; r.Eone = blockEnergy.row.one; r.Etwo_ud = blockEnergy.row.two_ud;
    r.Etwo_uu = blockEnergy.row.two_uu; r.Ethree = blockEnergy.row.three; r.Eother = blockEnergy.row.other;
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
            'params',params,'importanceMode',importanceMode,'updateMode',updateMode,'growthPolicy',growthPolicy,'mass_tol',mass_tol, ...
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
% Basis identities and Be adaptive growth
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

function [growBasis, oneFuncs, oneKeyMap] = grow_important_be_basis( ...
    importantBasis, oneFuncs, oneKeyMap, sigma0, cscale, D, ZsetMode, ...
    centerConvention, centerForNeighbors, neighborMode, growthPolicy)

    out = {}; k = 0;
    refA1 = 1; refA2 = 2; refB1 = 3; refB2 = 4;

    for ib = 1:numel(importantBasis)
        B = importantBasis(ib);
        block = string(B.block);
        tag = string(B.tag);

        switch block
            case "zero"
                if growthPolicy.keepUngrownImportant, k=k+1; out{k}=B; end 

            case "one"
                if ~growthPolicy.growOne
                    if growthPolicy.keepUngrownImportant, k=k+1; out{k}=B; end 
                    continue;
                end
                if tag == "one_a1"
                    active = active_alpha_excluding_be(B, refA2);
                    neigh = onefunc_neighbors_both(oneFuncs(active), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    for n=1:numel(neigh)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                        k=k+1; out{k}=make_be_basis('one', nid, refA2, refB1, refB2, 'one_a1'); 
                    end
                elseif tag == "one_a2"
                    active = active_alpha_excluding_be(B, refA1);
                    neigh = onefunc_neighbors_both(oneFuncs(active), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    for n=1:numel(neigh)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                        k=k+1; out{k}=make_be_basis('one', refA1, nid, refB1, refB2, 'one_a2'); 
                    end
                elseif tag == "one_b1"
                    active = active_beta_excluding_be(B, refB2);
                    neigh = onefunc_neighbors_both(oneFuncs(active), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    for n=1:numel(neigh)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                        k=k+1; out{k}=make_be_basis('one', refA1, refA2, nid, refB2, 'one_b1'); 
                    end
                elseif tag == "one_b2"
                    active = active_beta_excluding_be(B, refB1);
                    neigh = onefunc_neighbors_both(oneFuncs(active), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    for n=1:numel(neigh)
                        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                        k=k+1; out{k}=make_be_basis('one', refA1, refA2, refB1, nid, 'one_b2'); 
                    end
                else
                    ids = [B.a B.b B.c B.d]; activeIDs = ids(ids>4);
                    for id0 = unique(activeIDs)
                        neigh = onefunc_neighbors_both(oneFuncs(id0), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                        for n=1:numel(neigh)
                            [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                            bb=B; bb=replace_id_in_be_basis(bb,id0,nid);
                            k=k+1; out{k}=make_be_basis('one', bb.a, bb.b, bb.c, bb.d, char(tag)); 
                        end
                    end
                end

            case "two_ud"
                if ~growthPolicy.growTwoUD
                    if growthPolicy.keepUngrownImportant, k=k+1; out{k}=B; end 
                    continue;
                end
                switch tag
                    case "two_a1b1"
                        activeA = active_alpha_excluding_be(B, refA2); fixedA = refA2;
                        activeB = active_beta_excluding_be(B, refB2); fixedB = refB2;
                        [out,k,oneFuncs,oneKeyMap] = grow_two_ud_one_active(out,k,oneFuncs,oneKeyMap,B, ...
                            activeA,fixedA,activeB,fixedB,'two_a1b1',sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    case "two_a1b2"
                        activeA = active_alpha_excluding_be(B, refA2); fixedA = refA2;
                        activeB = active_beta_excluding_be(B, refB1); fixedB = refB1;
                        [out,k,oneFuncs,oneKeyMap] = grow_two_ud_one_active(out,k,oneFuncs,oneKeyMap,B, ...
                            activeA,fixedA,activeB,fixedB,'two_a1b2',sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    case "two_a2b1"
                        activeA = active_alpha_excluding_be(B, refA1); fixedA = refA1;
                        activeB = active_beta_excluding_be(B, refB2); fixedB = refB2;
                        [out,k,oneFuncs,oneKeyMap] = grow_two_ud_one_active(out,k,oneFuncs,oneKeyMap,B, ...
                            activeA,fixedA,activeB,fixedB,'two_a2b1',sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    case "two_a2b2"
                        activeA = active_alpha_excluding_be(B, refA1); fixedA = refA1;
                        activeB = active_beta_excluding_be(B, refB1); fixedB = refB1;
                        [out,k,oneFuncs,oneKeyMap] = grow_two_ud_one_active(out,k,oneFuncs,oneKeyMap,B, ...
                            activeA,fixedA,activeB,fixedB,'two_a2b2',sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                    otherwise
                        growIDs = unique([B.a B.b B.c B.d]);
                        for id0 = growIDs
                            if id0 <= 4, continue; end
                            neigh = onefunc_neighbors_both(oneFuncs(id0), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                            for n=1:numel(neigh)
                                [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                                bb=B; bb=replace_id_in_be_basis(bb,id0,nid);
                                k=k+1; out{k}=make_be_basis('two_ud', bb.a, bb.b, bb.c, bb.d, char(tag)); 
                            end
                        end
                end

            case "two_uu"
                if growthPolicy.growTwoUU
                    ids = [B.a B.b B.c B.d];
                    for pos=1:4
                        active = ids(pos);
                        if active <= 4, continue; end
                        neigh = onefunc_neighbors_both(oneFuncs(active), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                        for n=1:numel(neigh)
                            [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                            newIDs=ids; newIDs(pos)=nid;
                            if newIDs(1)~=newIDs(2) && newIDs(3)~=newIDs(4)
                                k=k+1; out{k}=make_be_basis('two_uu', newIDs(1),newIDs(2),newIDs(3),newIDs(4),char(tag)); 
                            end
                        end
                    end
                elseif growthPolicy.keepUngrownImportant
                    k=k+1; out{k}=B; 
                end

            case "three"
                if growthPolicy.growThree
                    ids = [B.a B.b B.c B.d];
                    for pos=1:4
                        active = ids(pos);
                        if active <= 4, continue; end
                        neigh = onefunc_neighbors_both(oneFuncs(active), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                        for n=1:numel(neigh)
                            [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                            newIDs=ids; newIDs(pos)=nid;
                            if newIDs(1)~=newIDs(2) && newIDs(3)~=newIDs(4)
                                k=k+1; out{k}=make_be_basis('three', newIDs(1),newIDs(2),newIDs(3),newIDs(4),char(tag)); 
                            end
                        end
                    end
                elseif growthPolicy.keepUngrownImportant
                    k=k+1; out{k}=B; 
                end

            case "other"
                if growthPolicy.growOther
                    ids = [B.a B.b B.c B.d];
                    for pos=1:4
                        active = ids(pos);
                        neigh = onefunc_neighbors_both(oneFuncs(active), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
                        for n=1:numel(neigh)
                            [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neigh(n));
                            newIDs=ids; newIDs(pos)=nid;
                            if newIDs(1)~=newIDs(2) && newIDs(3)~=newIDs(4)
                                k=k+1; out{k}=make_be_basis('other', newIDs(1),newIDs(2),newIDs(3),newIDs(4),char(tag)); 
                            end
                        end
                    end
                elseif growthPolicy.keepUngrownImportant
                    k=k+1; out{k}=B; 
                end
            otherwise
                error('Unknown Be basis block: %s', block);
        end
    end

    if isempty(out), growBasis = importantBasis([]); else, growBasis = [out{:}]; end
end

function [out,k,oneFuncs,oneKeyMap] = grow_two_ud_one_active(out,k,oneFuncs,oneKeyMap,B, ...
    activeA,fixedA,activeB,fixedB,tag,sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode)
    neighA = onefunc_neighbors_both(oneFuncs(activeA), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
    neighB = onefunc_neighbors_both(oneFuncs(activeB), sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode);
    for n=1:numel(neighA)
        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neighA(n));
        k=k+1; out{k}=make_be_basis('two_ud', nid, fixedA, activeB, fixedB, tag); 
    end
    for n=1:numel(neighB)
        [oneFuncs,oneKeyMap,nid]=register_onefunc(oneFuncs,oneKeyMap,neighB(n));
        k=k+1; out{k}=make_be_basis('two_ud', activeA, fixedA, nid, fixedB, tag); 
    end
end

function active = active_alpha_excluding_be(B, fixedRef)
    ids = [B.a B.b]; ids(ids==fixedRef) = [];
    if isempty(ids), active = []; else, active = ids(1); end
end

function active = active_beta_excluding_be(B, fixedRef)
    ids = [B.c B.d]; ids(ids==fixedRef) = [];
    if isempty(ids), active = []; else, active = ids(1); end
end

function bb = replace_id_in_be_basis(bb, oldID, newID)
    if bb.a==oldID, bb.a=newID; end
    if bb.b==oldID, bb.b=newID; end
    if bb.c==oldID, bb.c=newID; end
    if bb.d==oldID, bb.d=newID; end
end

function neigh = onefunc_neighbors_both(f, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode)
    if isempty(f) || ~isfield(f,'kind') || string(f.kind) ~= "frame"
        neigh = f([]); return;
    end
    neigh = unique_onefuncs([neighbors_Ng_onefunc(f,sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode), ...
                             neighbors_Ngplus_onefunc(f,sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode)]);
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
                if ~isempty(cand), neigh(end+1)=cand; end 
            end
        case 'psi'
            l = f.level + 1;
            for k=1:size(Z0,1)
                center = target + (1/(cscale*2^(f.level+2)))*Z0(k,:);
                cand = make_psi_at_center(sigma0,cscale,l,center,D,ZsetMode,centerConvention);
                if ~isempty(cand), neigh(end+1)=cand; end 
            end
    end
    neigh = unique_onefuncs(neigh);
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

function gb = estimate_sparse_pair_gb(S,H)
    ns=nnz(S); nh=nnz(H); m=size(S,1);
    gb=(16*(ns+nh)+8*2*(m+1))/1024^3;
end

function v = get_struct_default(s, field, defaultVal)
    if isstruct(s) && isfield(s, field)
        v = s.(field);
    else
        v = defaultVal;
    end
end

% ========================================================================
function f = empty_onefunc()
    f = struct('kind','','type','','level',0,'j',[0 0 0],'z',[0 0 0], ...
        'terms',struct('coef',{},'alpha',{},'center',{}),'key',string(''));
end

function b = empty_bebasis()
    b = struct('block','','a',0,'b',0,'c',0,'d',0,'tag','','key',string(''));
end

function [oneFuncs, oneKeyMap, id] = register_onefunc(oneFuncs, oneKeyMap, f)
    key = char(f.key);
    if isKey(oneKeyMap,key)
        id = oneKeyMap(key);
    else
        id = numel(oneFuncs)+1;
        oneFuncs(id) = f;
        oneKeyMap(key) = id;
    end
end

function [B, map] = append_basis(B, map, b)
    if isempty(b), return; end
    if b.a == b.b, return; end
    if b.c == b.d, return; end
    key = char(b.key);
    if ~isKey(map,key)
        B(end+1) = b; 
        map(key) = numel(B);
    end
end

function b = make_be_basis(block, ia, ib, ic, id, tag)
    b = empty_bebasis();
    ab = sort([ia ib]); cd = sort([ic id]);
    b.block = char(block); b.a = ab(1); b.b = ab(2); b.c = cd(1); b.d = cd(2); b.tag = char(tag);
    b.key = string(sprintf('%s_A%d_%d_B%d_%d', char(block), b.a, b.b, b.c, b.d));
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
    counts.other = nnz(blocks=="other");
    counts.total = M;
end

function frame = make_initial_frame(sigma0, cscale, L, D, ZsetMode, centerConvention)
    Zset = generate_z_set(D, ZsetMode);
    frame = repmat(empty_onefunc(), 0, 1);
    vals = -1:1; [A,B,C] = ndgrid(vals,vals,vals); js=[A(:),B(:),C(:)];
    for n=1:size(js,1)
        frame(end+1)=make_phi(sigma0,cscale,0,js(n,:),D); 
    end
    for l=0:(L-2)
        for iz=1:size(Zset,1)
            z=Zset(iz,:); j=-z;
            frame(end+1)=make_psi(sigma0,cscale,l,j,z,D,centerConvention); 
        end
    end
end

function neigh = neighbors_Ng_onefunc(f, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode)
    target = center_of_basis(f,cscale,centerForNeighbors);
    Z0 = generate_neighbor_set(neighborMode);
    neigh = repmat(empty_onefunc(),0,1);
    switch char(string(f.type))
        case 'phi'
            for k=1:size(Z0,1)
                center = target + (1/cscale)*Z0(k,:);
                neigh(end+1)=make_phi_at_center(sigma0,cscale,0,center,D); 
            end
        case 'psi'
            l=f.level;
            for k=1:size(Z0,1)
                center = target + (1/(cscale*2^(l+1)))*Z0(k,:);
                cand = make_psi_at_center(sigma0,cscale,l,center,D,ZsetMode,centerConvention);
                if ~isempty(cand), neigh(end+1)=cand; end 
            end
        otherwise
            neigh = repmat(empty_onefunc(),0,1);
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
            Zset = dec2bin(0:(2^D-1))- '0'; Zset=Zset(any(Zset~=0,2),:);
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
    scale = cscale*2^l;
    j = round(center * scale);
    center2 = j / scale;
    f = make_phi(sigma0,cscale,l,j,D);
    f.terms(1).center = center2;
    f.key = canonical_key_from_center('phi',l,center2,cscale);
end

function f = make_psi(sigma0,cscale,l,j,z,D,centerConvention)
    gamma = 2^(-D/2); Cpsi = (1 - (16/25)*gamma*sqrt(5) + gamma^2)^(-1/2);
    scale = cscale*2^l;
    switch lower(centerConvention)
        case 'eq428', center = (j + 0.5*z)/scale;
        case 'cmap',  center = (j - 0.5*z)/scale;
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
    m = round(2 * scale * center);
    z = zeros(1,D); j = zeros(1,D);
    switch lower(ZsetMode)
        case 'signed'
            for d=1:D
                if mod(abs(m(d)),2)==0
                    z(d)=0; j(d)=m(d)/2;
                else
                    z(d)=sign(m(d)); if z(d)==0, z(d)=1; end
                    switch lower(centerConvention)
                        case 'eq428', j(d)=(m(d)-z(d))/2;
                        case 'cmap',  j(d)=(m(d)+z(d))/2;
                    end
                end
            end
        case 'binary'
            for d=1:D
                if mod(abs(m(d)),2)==0
                    z(d)=0; j(d)=m(d)/2;
                else
                    z(d)=1;
                    switch lower(centerConvention)
                        case 'eq428', j(d)=(m(d)-1)/2;
                        case 'cmap',  j(d)=(m(d)+1)/2;
                    end
                end
            end
        otherwise
            error('Unknown ZsetMode.');
    end
    if all(z==0)
        f = [];
        return;
    end
    f = make_psi(sigma0,cscale,l,round(j),round(z),D,centerConvention);
    f.key = canonical_key_from_center('psi',l,f.terms(1).center,cscale);
end

function key = canonical_key_from_center(type, level, center, cscale)
    sc = max(1, round(cscale*2^max(level,0)*2)); %#ok<NASGU>
    ci = round(center * 1e10)/1e10;
    key = string(sprintf('%s_L%d_C%.10g_%.10g_%.10g', type, level, ci(1), ci(2), ci(3)));
end

function C = center_of_basis(f,cscale,mode)
    switch lower(mode)
        case 'actual'
            if isfield(f,'terms') && ~isempty(f.terms)
                C = f.terms(1).center;
            else
                C = [0 0 0];
            end
        case 'cmap'
            if strcmp(f.type,'phi')
                C = f.j / cscale;
            else
                C = (f.j - 0.5*f.z)/(cscale*2^f.level);
            end
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

function [S,H,info] = assemble_be_matrices(B, oneCache, oneFuncs, opts)
    M = numel(B); S1=oneCache.S1; H1=oneCache.H1; td=build_term_data(oneFuncs);
    normInv = be_norm_inv(B,S1,opts);
    if isfield(opts,'usePairCache') && opts.usePairCache
        pairData = build_be_pair_cache(B,S1,H1,td,opts);
        [S,H,info] = assemble_be_matrices_paircached(B,S1,H1,td,normInv,opts,pairData);
        info.pairCache = pairData.info;
        info.pairDataForCache = pairData;
    else
        error('This v4 file is intended to use pair-cached Be assembly. Set opts.usePairCache=true.');
    end
end


function [S,H,info] = assemble_be_matrices_incremental(B, oneCache, oneFuncs, opts, Sold, Hold, nOld, oldPairData)
    % Incremental assembly for monotone_union adaptive growth.
    % If B = [Bold, Bnew], old-old matrix is copied from Sold/Hold, and only
    % rows nOld+1:M against all columns are assembled exactly.
    M = numel(B);
    if nargin < 8, oldPairData = []; end
    if nOld <= 0 || nOld >= M
        [S,H,info] = assemble_be_matrices(B, oneCache, oneFuncs, opts);
        info.backend = 'incremental_fallback_full';
        return;
    end
    if size(Sold,1) ~= nOld || size(Hold,1) ~= nOld
        warning('Incremental cache matrix size mismatch; falling back to full assembly.');
        [S,H,info] = assemble_be_matrices(B, oneCache, oneFuncs, opts);
        info.backend = 'incremental_size_mismatch_full';
        return;
    end

    S1 = oneCache.S1; H1 = oneCache.H1; td = build_term_data(oneFuncs);
    normInv = be_norm_inv(B,S1,opts);

    optsFullRows = opts;
    optsFullRows.upperTriangle = false;   % new rows must include old columns too
    optsFullRows.showRowProgress = false; % progress handled outside
    newRows = (nOld+1):M;
    tNew = tic;
    incBackend = lower(char(string(get_opt(opts,'incrementalBackend','auto_fast'))));
    if strcmp(incBackend,'auto_fast') || strcmp(incBackend,'auto')
        if should_use_gpu_newrows(opts, M, numel(newRows))
            incBackend = 'paircached_gpu_newrows';
        else
            incBackend = 'paircached_newrows';
        end
    end

    switch incBackend
        case {'direct_newrows','direct'}
            % Kept only as an audit/fallback backend.  For Be this repeats
            % same-spin and one-body determinant work and is usually slower.
            [SnewRows,HnewRows] = assemble_be_sparse_rows_direct(B,newRows,S1,H1,td,normInv,optsFullRows);
            pairInfo = struct('backend','direct_newrows','nAlphaPairs',NaN,'nBetaPairs',NaN,'buildTime',0);
        case {'paircached_newrows','paircached'}
            pairData = build_be_pair_cache_reuse(B,S1,H1,td,opts,oldPairData);
            [SnewRows,HnewRows] = assemble_be_sparse_rows_paircached(B,newRows,S1,H1,td,normInv,optsFullRows,pairData);
            pairInfo = pairData.info; pairInfo.backend = 'paircached_newrows';
        case {'paircached_gpu_newrows','gpu_newrows','gpu'}
            pairData = build_be_pair_cache_reuse(B,S1,H1,td,opts,oldPairData);
            [SnewRows,HnewRows,gpuInfo] = assemble_be_sparse_rows_paircached_gpuVab(B,newRows,S1,H1,td,normInv,optsFullRows,pairData);
            pairInfo = pairData.info; pairInfo.backend = 'paircached_gpu_newrows'; pairInfo.gpuInfo = gpuInfo;
        otherwise
            error('Unknown incrementalBackend: %s', incBackend);
    end
    tRows = toc(tNew);

    oldIdx = 1:nOld; newIdx = (nOld+1):M;
    Sold = sparse(Sold); Hold = sparse(Hold);
    SnewRows = sparse(SnewRows); HnewRows = sparse(HnewRows);

    S = sparse(M,M); H = sparse(M,M);
    S(oldIdx,oldIdx) = Sold; H(oldIdx,oldIdx) = Hold;

    % Cross blocks from newly assembled rows.
    S(newIdx,oldIdx) = SnewRows(:,oldIdx);
    S(oldIdx,newIdx) = SnewRows(:,oldIdx).';
    H(newIdx,oldIdx) = HnewRows(:,oldIdx);
    H(oldIdx,newIdx) = HnewRows(:,oldIdx).';

    % New-new block.  Symmetrize defensively because it was assembled rowwise.
    Snn = SnewRows(:,newIdx); Hnn = HnewRows(:,newIdx);
    S(newIdx,newIdx) = 0.5*(Snn + Snn.');
    H(newIdx,newIdx) = 0.5*(Hnn + Hnn.');

    S = 0.5*(S+S.'); H = 0.5*(H+H.');
    info = struct('nnzS',nnz(S),'nnzH',nnz(H),'M',M,'oldM',nOld,'newRows',numel(newRows), ...
        'upperTriangle',false,'backend',['incremental_' incBackend],'newRowsTime',tRows, ...
        'pairCache',pairInfo,'incrementalBackend',incBackend);
    if exist('pairData','var'), info.pairDataForCache = pairData; end
    fprintf('    Incremental assembly: reused old block M=%d; assembled %d new row(s) against M=%d in %.2fs.\n', ...
        nOld, numel(newRows), M, tRows);
end

function normInv = be_norm_inv(B,S1,opts)
    M=numel(B); mode='l2_normalized'; if isfield(opts,'detScaleMode'), mode=lower(char(string(opts.detScaleMode))); end
    switch mode
        case {'l2_normalized','normalized'}
            normInv=zeros(1,M);
            for i=1:M
                a=B(i).a; b=B(i).b; c=B(i).c; d=B(i).d;
                da = S1(a,a)*S1(b,b)-S1(a,b)*S1(b,a);
                db = S1(c,c)*S1(d,d)-S1(c,d)*S1(d,c);
                n2 = da*db;
                if n2 < 1e-14, n2=1e-14; end
                normInv(i)=1/sqrt(n2);
            end
        case {'antisymmetrizer','raw'}
            normInv=ones(1,M);
        otherwise
            error('Unknown detScaleMode: %s', mode);
    end
end

function pairData = build_be_pair_cache(B,S1,H1,td,opts)
    % Be determinant = alpha pair x beta pair.  Cache all pair-transition
    % overlap and same-spin Hamiltonian blocks once, instead of recomputing
    % them for every beta/alpha partner determinant.
    M=numel(B);
    Araw=[[B.a].',[B.b].'];
    Braw=[[B.c].',[B.d].'];
    [Apairs,Aidx] = register_pair_list(Araw);
    [Bpairs,Bidx] = register_pair_list(Braw);
    t=tic;
    [SA,HA] = build_pair_transition_mats(Apairs,S1,H1,td);
    [SB,HB] = build_pair_transition_mats(Bpairs,S1,H1,td);
    pairData = struct();
    pairData.Apairs=Apairs; pairData.Bpairs=Bpairs;
    pairData.Aidx=Aidx(:).'; pairData.Bidx=Bidx(:).';
    pairData.SA=SA; pairData.HA=HA; pairData.SB=SB; pairData.HB=HB;
    pairData.info=struct('nAlphaPairs',size(Apairs,1),'nBetaPairs',size(Bpairs,1),'buildTime',toc(t));
    fprintf('    Pair cache built: nAlphaPairs=%d, nBetaPairs=%d, time=%.2fs\n', ...
        size(Apairs,1), size(Bpairs,1), pairData.info.buildTime);
end

function [pairs,idx] = register_pair_list(P)
    n=size(P,1); pairs=zeros(0,2); idx=zeros(n,1);
    mp=containers.Map('KeyType','char','ValueType','double');
    for i=1:n
        p=sort(P(i,:)); key=sprintf('%d_%d',p(1),p(2));
        if isKey(mp,key)
            idx(i)=mp(key);
        else
            pairs(end+1,:)=p; 
            idx(i)=size(pairs,1); mp(key)=idx(i);
        end
    end
end

function [SP,HP] = build_pair_transition_mats(pairs,S1,H1,td)
    N=size(pairs,1);
    [SP,HP] = build_pair_transition_rows(pairs,1:N,S1,H1,td);
    SP=0.5*(SP+SP.'); HP=0.5*(HP+HP.');
end

function [SProws,HProws] = build_pair_transition_rows(pairs,rowIds,S1,H1,td)
    N=size(pairs,1); nr=numel(rowIds);
    SProws=zeros(nr,N); HProws=zeros(nr,N);
    e=pairs(:,1).'; f=pairs(:,2).';
    for rr=1:nr
        i=rowIds(rr);
        a=pairs(i,1); b=pairs(i,2);
        Sae=S1(a,e); Saf=S1(a,f); Sbe=S1(b,e); Sbf=S1(b,f);
        Hae=H1(a,e); Haf=H1(a,f); Hbe=H1(b,e); Hbf=H1(b,f);
        detP = Sae.*Sbf - Saf.*Sbe;
        Hone = Hae.*Sbf + Sae.*Hbf - Haf.*Sbe - Saf.*Hbe;
        Vsame = eri_row_vectorized_cpu(td,a,e,b,f) - eri_row_vectorized_cpu(td,a,f,b,e);
        SProws(rr,:)=detP;
        HProws(rr,:)=Hone + Vsame;
    end
end

function pairData = build_be_pair_cache_reuse(B,S1,H1,td,opts,oldPairData)
    % Reuse/update alpha and beta pair transition matrices across monotone
    % adaptive steps.  The old pair list is normally a stable prefix of the
    % new pair list, so only transition rows/cols involving newly introduced
    % unique pairs are evaluated.
    if nargin < 6 || isempty(oldPairData) || ~get_opt(opts,'reusePairCache',true)
        pairData = build_be_pair_cache(B,S1,H1,td,opts);
        pairData.info.reused = false;
        return;
    end
    t=tic;
    Araw=[[B.a].',[B.b].']; Braw=[[B.c].',[B.d].'];
    [Apairs,Aidx] = register_pair_list(Araw);
    [Bpairs,Bidx] = register_pair_list(Braw);
    [SA,HA,reuseA] = update_one_pair_cache(Apairs,S1,H1,td,oldPairData.Apairs,oldPairData.SA,oldPairData.HA);
    [SB,HB,reuseB] = update_one_pair_cache(Bpairs,S1,H1,td,oldPairData.Bpairs,oldPairData.SB,oldPairData.HB);
    pairData = struct();
    pairData.Apairs=Apairs; pairData.Bpairs=Bpairs;
    pairData.Aidx=Aidx(:).'; pairData.Bidx=Bidx(:).';
    pairData.SA=SA; pairData.HA=HA; pairData.SB=SB; pairData.HB=HB;
    pairData.info=struct('nAlphaPairs',size(Apairs,1),'nBetaPairs',size(Bpairs,1), ...
        'buildTime',toc(t),'reused',true,'reuseA',reuseA,'reuseB',reuseB);
    fprintf('    Pair cache reused/updated: A %d->%d (+%d), B %d->%d (+%d), time=%.2fs\n', ...
        reuseA.nOld, reuseA.nNew, reuseA.nAdded, reuseB.nOld, reuseB.nNew, reuseB.nAdded, pairData.info.buildTime);
end

function [Snew,Hnew,info] = update_one_pair_cache(pairs,S1,H1,td,oldPairs,Sold,Hold)
    N=size(pairs,1); Nold=size(oldPairs,1);
    info=struct('nOld',Nold,'nNew',N,'nAdded',max(N-Nold,0),'reusedPrefix',false,'fallback',false);
    if Nold>0 && Nold<=N && size(Sold,1)==Nold && size(Hold,1)==Nold && isequal(pairs(1:Nold,:),oldPairs)
        info.reusedPrefix=true;
        if Nold==N
            Snew=Sold; Hnew=Hold; info.nAdded=0; return;
        end
        Snew=zeros(N,N); Hnew=zeros(N,N);
        Snew(1:Nold,1:Nold)=Sold; Hnew(1:Nold,1:Nold)=Hold;
        newIds=(Nold+1):N;
        [Srows,Hrows]=build_pair_transition_rows(pairs,newIds,S1,H1,td);
        Snew(newIds,:)=Srows; Hnew(newIds,:)=Hrows;
        Snew(:,newIds)=Srows.'; Hnew(:,newIds)=Hrows.';
        Snew=0.5*(Snew+Snew.'); Hnew=0.5*(Hnew+Hnew.');
    else
        [Snew,Hnew]=build_pair_transition_mats(pairs,S1,H1,td);
        info.fallback=true; info.nAdded=N;
    end
end

function [S,H,info] = assemble_be_matrices_paircached(B,S1,H1,td,normInv,opts,pairData)
    M=numel(B); rowBlock=opts.rowBlockSize; blocks=1:rowBlock:M;
    S=sparse(M,M); H=sparse(M,M); t0=tic;
    if strcmpi(opts.mode,'cpu_parfor_sparse') && opts.startPool
        try
            pool = gcp('nocreate'); if isempty(pool), parpool; end %#ok<NASGU>
        catch ME
            warning('Could not start parallel pool, continuing serial/parfor fallback: %s', ME.message);
        end
    end
    for ib=1:numel(blocks)
        rows=blocks(ib):min(M,blocks(ib)+rowBlock-1);
        [Sr,Hr]=assemble_be_sparse_rows_paircached(B,rows,S1,H1,td,normInv,opts,pairData);
        S(rows,:)=S(rows,:)+Sr; H(rows,:)=H(rows,:)+Hr;
        if opts.showRowProgress
            pe=get_opt(opts,'progressEveryBlocks',1);
            if mod(ib,pe)==0 || ib==numel(blocks)
                fprintf('    Be pair-cached rows %d-%d / %d inserted, nnz(S/H)=%.3e/%.3e, elapsed %.1fs\n', ...
                    rows(1), rows(end), M, nnz(S), nnz(H), toc(t0));
            end
        end
    end
    if isfield(opts,'upperTriangle') && opts.upperTriangle
        S=S+triu(S,1).'; H=H+triu(H,1).';
    end
    S=0.5*(S+S'); H=0.5*(H+H');
    info=struct('nnzS',nnz(S),'nnzH',nnz(H),'M',M,'upperTriangle',opts.upperTriangle,'backend','paircached');
end

function [Srows,Hrows] = assemble_be_sparse_rows_paircached(B,rowIdx,S1,H1,td,normInv,opts,pairData)
    nr=numel(rowIdx); M=numel(B);
    avec=[B.a]; bvec=[B.b]; cvec=[B.c]; dvec=[B.d];
    switch lower(opts.mode)
        case 'cpu_parfor_sparse'
            IScell=cell(nr,1); JScell=cell(nr,1); VScell=cell(nr,1);
            IHcell=cell(nr,1); JHcell=cell(nr,1); VHcell=cell(nr,1);
            parfor ir=1:nr
                [js,vs,jh,vh]=assemble_one_be_row_paircached(B,rowIdx(ir),avec,bvec,cvec,dvec,S1,H1,td,normInv,opts,pairData);
                IScell{ir}=ir*ones(numel(js),1); JScell{ir}=js(:); VScell{ir}=vs(:);
                IHcell{ir}=ir*ones(numel(jh),1); JHcell{ir}=jh(:); VHcell{ir}=vh(:);
            end
            IS=vertcat(IScell{:}); JS=vertcat(JScell{:}); VS=vertcat(VScell{:});
            IH=vertcat(IHcell{:}); JH=vertcat(JHcell{:}); VH=vertcat(VHcell{:});
        otherwise
            IS=[];JS=[];VS=[];IH=[];JH=[];VH=[];
            for ir=1:nr
                [js,vs,jh,vh]=assemble_one_be_row_paircached(B,rowIdx(ir),avec,bvec,cvec,dvec,S1,H1,td,normInv,opts,pairData);
                IS=[IS; ir*ones(numel(js),1)]; JS=[JS; js(:)]; VS=[VS; vs(:)]; 
                IH=[IH; ir*ones(numel(jh),1)]; JH=[JH; jh(:)]; VH=[VH; vh(:)]; 
            end
    end
    Srows=sparse(IS,JS,VS,nr,M); Hrows=sparse(IH,JH,VH,nr,M);
end

function [js,vs,jh,vh] = assemble_one_be_row_paircached(B,row,avec,bvec,cvec,dvec,S1,H1,td,normInv,opts,pairData)
    a=B(row).a; b=B(row).b; c=B(row).c; d0=B(row).d;
    M=numel(avec); e=avec; f=bvec; g=cvec; h=dvec;
    if isfield(opts,'upperTriangle') && opts.upperTriangle
        upperMask=(1:M)>=row;
    else
        upperMask=true(1,M);
    end

    ia=pairData.Aidx(row); ib=pairData.Bidx(row);
    JA=pairData.Aidx; JB=pairData.Bidx;
    SArow=pairData.SA(ia,JA); HArow=pairData.HA(ia,JA);
    SBrow=pairData.SB(ib,JB); HBrow=pairData.HB(ib,JB);
    Sraw = SArow .* SBrow;
    Hraw = HArow .* SBrow + SArow .* HBrow;

    if isfield(opts,'screenERIByOverlap') && opts.screenERIByOverlap
        preMask = upperMask & (abs(Sraw) >= opts.screenTolSForERI | abs(Hraw) >= opts.screenTolHCheapForERI);
        if isfield(opts,'forceDiagonal') && opts.forceDiagonal && row <= M
            preMask(row)=true;
        end
        idx=find(preMask);
    else
        idx=find(upperMask);
    end

    if ~isempty(idx)
        ei=e(idx); fi=f(idx); gi=g(idx); hi=h(idx);
        % Cofactors of alpha and beta transition overlap determinants.
        Sae=S1(a,ei); Saf=S1(a,fi); Sbe=S1(b,ei); Sbf=S1(b,fi);
        Scg=S1(c,gi); Sch=S1(c,hi); Sdg=S1(d0,gi); Sdh=S1(d0,hi);
        CA11=Sbf; CA12=-Sbe; CA21=-Saf; CA22=Sae;
        CB11=Sdh; CB12=-Sdg; CB21=-Sch; CB22=Scg;

        % Opposite-spin alpha-beta Coulomb.  Old versions evaluated the
        % 16 cofactor ERI terms one-by-one.  This batched kernel groups them
        % into 4 vectorized calls by the left pair (a,c), (a,d), (b,c), (b,d),
        % reducing MATLAB function-call overhead and improving BLAS/GPU vector length.
        Vab_i = be_vab_batched_cpu(td,a,b,c,d0,ei,fi,gi,hi, ...
            CA11,CA12,CA21,CA22,CB11,CB12,CB21,CB22,opts);
        Hraw(idx)=Hraw(idx)+Vab_i;
    end

    fac=normInv(row).*normInv;
    srow=Sraw.*fac; hrow=Hraw.*fac;
    maskS=upperMask & (abs(srow)>=opts.dropTolS);
    maskH=upperMask & (abs(hrow)>=opts.dropTolH);
    if isfield(opts,'forceDiagonal') && opts.forceDiagonal && row <= M
        maskS(row)=true; maskH(row)=true;
    end
    js=find(maskS); vs=srow(maskS);
    jh=find(maskH); vh=hrow(maskH);
end



function tf = should_use_gpu_newrows(opts, M, nNewRows)
    % Conservative GPU dispatch.  GPU is useful only when each row has a long
    % V_ab vector.  For tiny blocks, parfor CPU is usually faster.
    tf = false;
    if ~get_opt(opts,'preferGPU',false), return; end
    if M < get_opt(opts,'gpuMinM',1800), return; end
    if nNewRows < get_opt(opts,'gpuMinNewRows',8), return; end
    try
        g = gpuDevice(get_opt(opts,'gpuDeviceID',1)); %#ok<NASGU>
        tf = true;
    catch
        tf = false;
    end
end

function [Srows,Hrows,gpuInfo] = assemble_be_sparse_rows_paircached_gpuVab(B,rowIdx,S1,H1,td,normInv,opts,pairData)
    % GPU backend for newly appended rows.  We keep all cheap determinant and
    % pair-cache operations on CPU, and move only the expensive V_ab batched
    % Coulomb contractions to the GPU.  Rows are serial on one GPU to avoid
    % multiple workers fighting for a small 3050 device.
    nr=numel(rowIdx); M=numel(B);
    avec=[B.a]; bvec=[B.b]; cvec=[B.c]; dvec=[B.d];
    tdg = build_term_data_gpu(td);
    IS=cell(nr,1); JS=cell(nr,1); VS=cell(nr,1);
    IH=cell(nr,1); JH=cell(nr,1); VH=cell(nr,1);
    t0=tic;
    for ir=1:nr
        [js,vs,jh,vh]=assemble_one_be_row_paircached_gpuVab(B,rowIdx(ir),avec,bvec,cvec,dvec,S1,H1,td,tdg,normInv,opts,pairData);
        IS{ir}=ir*ones(numel(js),1); JS{ir}=js(:); VS{ir}=vs(:);
        IH{ir}=ir*ones(numel(jh),1); JH{ir}=jh(:); VH{ir}=vh(:);
        pe=get_opt(opts,'gpuRowProgressEvery',0);
        if pe>0 && (mod(ir,pe)==0 || ir==nr)
            fprintf('      GPU Vab newrow %d/%d, elapsed %.1fs\n', ir, nr, toc(t0));
        end
    end
    Srows=sparse(vertcat(IS{:}),vertcat(JS{:}),vertcat(VS{:}),nr,M);
    Hrows=sparse(vertcat(IH{:}),vertcat(JH{:}),vertcat(VH{:}),nr,M);
    gpuInfo=struct('used',true,'nRows',nr,'time',toc(t0),'deviceID',get_opt(opts,'gpuDeviceID',1));
end

function [js,vs,jh,vh] = assemble_one_be_row_paircached_gpuVab(B,row,avec,bvec,cvec,dvec,S1,H1,td,tdg,normInv,opts,pairData)
    a=B(row).a; b=B(row).b; c=B(row).c; d0=B(row).d;
    M=numel(avec); e=avec; f=bvec; g=cvec; h=dvec;
    if isfield(opts,'upperTriangle') && opts.upperTriangle
        upperMask=(1:M)>=row;
    else
        upperMask=true(1,M);
    end

    ia=pairData.Aidx(row); ib=pairData.Bidx(row);
    JA=pairData.Aidx; JB=pairData.Bidx;
    SArow=pairData.SA(ia,JA); HArow=pairData.HA(ia,JA);
    SBrow=pairData.SB(ib,JB); HBrow=pairData.HB(ib,JB);
    Sraw = SArow .* SBrow;
    Hraw = HArow .* SBrow + SArow .* HBrow;

    if isfield(opts,'screenERIByOverlap') && opts.screenERIByOverlap
        preMask = upperMask & (abs(Sraw) >= opts.screenTolSForERI | abs(Hraw) >= opts.screenTolHCheapForERI);
        if isfield(opts,'forceDiagonal') && opts.forceDiagonal && row <= M
            preMask(row)=true;
        end
        idx=find(preMask);
    else
        idx=find(upperMask);
    end

    if ~isempty(idx)
        ei=e(idx); fi=f(idx); gi=g(idx); hi=h(idx);
        Sae=S1(a,ei); Saf=S1(a,fi); Sbe=S1(b,ei); Sbf=S1(b,fi);
        Scg=S1(c,gi); Sch=S1(c,hi); Sdg=S1(d0,gi); Sdh=S1(d0,hi);
        CA11=Sbf; CA12=-Sbe; CA21=-Saf; CA22=Sae;
        CB11=Sdh; CB12=-Sdg; CB21=-Sch; CB22=Scg;
        Vab_i = be_vab_batched_gpu(td,tdg,a,b,c,d0,ei,fi,gi,hi, ...
            CA11,CA12,CA21,CA22,CB11,CB12,CB21,CB22,opts);
        Hraw(idx)=Hraw(idx)+Vab_i;
    end

    fac=normInv(row).*normInv;
    srow=Sraw.*fac; hrow=Hraw.*fac;
    maskS=upperMask & (abs(srow)>=opts.dropTolS);
    maskH=upperMask & (abs(hrow)>=opts.dropTolH);
    if isfield(opts,'forceDiagonal') && opts.forceDiagonal && row <= M
        maskS(row)=true; maskH(row)=true;
    end
    js=find(maskS); vs=srow(maskS);
    jh=find(maskH); vh=hrow(maskH);
end

function tdg = build_term_data_gpu(td)
    tdg = td;
    tdg.coef = gpuArray(td.coef);
    tdg.ncoef = gpuArray(td.ncoef);
    tdg.alpha = gpuArray(td.alpha);
    tdg.cx = gpuArray(td.cx);
    tdg.cy = gpuArray(td.cy);
    tdg.cz = gpuArray(td.cz);
    tdg.nTermsCPU = td.nTerms;
end

function Vab = be_vab_batched_cpu(td,a,b,c,d0,ei,fi,gi,hi,CA11,CA12,CA21,CA22,CB11,CB12,CB21,CB22,opts)
    n=numel(ei); Vab=zeros(1,n); if n==0, return; end
    chunk=get_opt(opts,'vabChunkSizeCPU',12000);
    for s0=1:chunk:n
        s1=min(n,s0+chunk-1); ii=s0:s1;
        Vab(ii)=be_vab_batched_cpu_core(td,a,b,c,d0,ei(ii),fi(ii),gi(ii),hi(ii), ...
            CA11(ii),CA12(ii),CA21(ii),CA22(ii),CB11(ii),CB12(ii),CB21(ii),CB22(ii),opts);
    end
end

function V = be_vab_batched_cpu_core(td,a,b,c,d0,ei,fi,gi,hi,CA11,CA12,CA21,CA22,CB11,CB12,CB21,CB22,opts)
    n=numel(ei); V=zeros(1,n); coeffTol=get_opt(opts,'eriCoeffTol',0);
    % left (a,c): right combinations (e,g),(e,h),(f,g),(f,h)
    V = V + be_vab_group_cpu(td,a,c, [ei ei fi fi], [gi hi gi hi], ...
        [CA11.*CB11, CA11.*CB12, CA12.*CB11, CA12.*CB12], n, coeffTol);
    % left (a,d)
    V = V + be_vab_group_cpu(td,a,d0, [ei ei fi fi], [gi hi gi hi], ...
        [CA11.*CB21, CA11.*CB22, CA12.*CB21, CA12.*CB22], n, coeffTol);
    % left (b,c)
    V = V + be_vab_group_cpu(td,b,c, [ei ei fi fi], [gi hi gi hi], ...
        [CA21.*CB11, CA21.*CB12, CA22.*CB11, CA22.*CB12], n, coeffTol);
    % left (b,d)
    V = V + be_vab_group_cpu(td,b,d0, [ei ei fi fi], [gi hi gi hi], ...
        [CA21.*CB21, CA21.*CB22, CA22.*CB21, CA22.*CB22], n, coeffTol);
end

function out = be_vab_group_cpu(td, leftA, leftB, ibStack, idStack, prefStack, n, coeffTol)
    out=zeros(1,n);
    if coeffTol>0
        m=abs(prefStack)>coeffTol;
        vals=zeros(1,4*n);
        if any(m)
            vals(m)=eri_row_vectorized_cpu(td,leftA,ibStack(m),leftB,idStack(m));
        end
    else
        vals=eri_row_vectorized_cpu(td,leftA,ibStack,leftB,idStack);
    end
    out = sum(reshape(prefStack.*vals,n,4),2).';
end

function Vab = be_vab_batched_gpu(td,tdg,a,b,c,d0,ei,fi,gi,hi,CA11,CA12,CA21,CA22,CB11,CB12,CB21,CB22,opts)
    n=numel(ei); Vab=zeros(1,n); if n==0, return; end
    chunk=get_opt(opts,'vabChunkSizeGPU',4096);
    for s0=1:chunk:n
        s1=min(n,s0+chunk-1); ii=s0:s1;
        Vab(ii)=be_vab_batched_gpu_core(td,tdg,a,b,c,d0,ei(ii),fi(ii),gi(ii),hi(ii), ...
            CA11(ii),CA12(ii),CA21(ii),CA22(ii),CB11(ii),CB12(ii),CB21(ii),CB22(ii));
    end
end

function V = be_vab_batched_gpu_core(td,tdg,a,b,c,d0,ei,fi,gi,hi,CA11,CA12,CA21,CA22,CB11,CB12,CB21,CB22)
    n=numel(ei); Vg=gpuArray.zeros(1,n);
    Vg = Vg + be_vab_group_gpu(tdg,a,c,[ei ei fi fi],[gi hi gi hi], ...
        [CA11.*CB11, CA11.*CB12, CA12.*CB11, CA12.*CB12], n);
    Vg = Vg + be_vab_group_gpu(tdg,a,d0,[ei ei fi fi],[gi hi gi hi], ...
        [CA11.*CB21, CA11.*CB22, CA12.*CB21, CA12.*CB22], n);
    Vg = Vg + be_vab_group_gpu(tdg,b,c,[ei ei fi fi],[gi hi gi hi], ...
        [CA21.*CB11, CA21.*CB12, CA22.*CB11, CA22.*CB12], n);
    Vg = Vg + be_vab_group_gpu(tdg,b,d0,[ei ei fi fi],[gi hi gi hi], ...
        [CA21.*CB21, CA21.*CB22, CA22.*CB21, CA22.*CB22], n);
    V = gather(Vg);
end

function outg = be_vab_group_gpu(tdg, leftA, leftB, ibStack, idStack, prefStack, n)
    vals = eri_row_vectorized_gpu(tdg,leftA,ibStack,leftB,idStack);
    outg = sum(reshape(gpuArray(prefStack).*vals,n,4),2).';
end


function [Srows,Hrows] = assemble_be_sparse_rows_direct(B,rowIdx,S1,H1,td,normInv,opts)
    % Direct exact rows used by incremental assembly.  This is usually faster
    % than rebuilding the full pair cache when only a moderate number of new
    % rows has been appended to a large old matrix.
    nr=numel(rowIdx); M=numel(B);
    avec=[B.a]; bvec=[B.b]; cvec=[B.c]; dvec=[B.d];
    switch lower(opts.mode)
        case 'cpu_parfor_sparse'
            IScell=cell(nr,1); JScell=cell(nr,1); VScell=cell(nr,1);
            IHcell=cell(nr,1); JHcell=cell(nr,1); VHcell=cell(nr,1);
            parfor ir=1:nr
                [js,vs,jh,vh]=assemble_one_be_row_direct(B,rowIdx(ir),avec,bvec,cvec,dvec,S1,H1,td,normInv,opts);
                IScell{ir}=ir*ones(numel(js),1); JScell{ir}=js(:); VScell{ir}=vs(:);
                IHcell{ir}=ir*ones(numel(jh),1); JHcell{ir}=jh(:); VHcell{ir}=vh(:);
            end
            IS=vertcat(IScell{:}); JS=vertcat(JScell{:}); VS=vertcat(VScell{:});
            IH=vertcat(IHcell{:}); JH=vertcat(JHcell{:}); VH=vertcat(VHcell{:});
        otherwise
            IS=[];JS=[];VS=[];IH=[];JH=[];VH=[];
            for ir=1:nr
                [js,vs,jh,vh]=assemble_one_be_row_direct(B,rowIdx(ir),avec,bvec,cvec,dvec,S1,H1,td,normInv,opts);
                IS=[IS; ir*ones(numel(js),1)]; JS=[JS; js(:)]; VS=[VS; vs(:)]; 
                IH=[IH; ir*ones(numel(jh),1)]; JH=[JH; jh(:)]; VH=[VH; vh(:)]; 
            end
    end
    Srows=sparse(IS,JS,VS,nr,M); Hrows=sparse(IH,JH,VH,nr,M);
end

function [js,vs,jh,vh] = assemble_one_be_row_direct(B,row,avec,bvec,cvec,dvec,S1,H1,td,normInv,opts)
    % Direct single-row kernel used only for the fast preselector.  It avoids
    % building the full pair cache for the full M=4459 raw list.
    a=B(row).a; b=B(row).b; c=B(row).c; d0=B(row).d;
    M=numel(avec); e=avec; f=bvec; g=cvec; h=dvec;
    if isfield(opts,'upperTriangle') && opts.upperTriangle
        upperMask=(1:M)>=row;
    else
        upperMask=true(1,M);
    end
    Sae=S1(a,e); Saf=S1(a,f); Sbe=S1(b,e); Sbf=S1(b,f);
    Hae=H1(a,e); Haf=H1(a,f); Hbe=H1(b,e); Hbf=H1(b,f);
    Scg=S1(c,g); Sch=S1(c,h); Sdg=S1(d0,g); Sdh=S1(d0,h);
    Hcg=H1(c,g); Hch=H1(c,h); Hdg=H1(d0,g); Hdh=H1(d0,h);
    detA=Sae.*Sbf-Saf.*Sbe; detB=Scg.*Sdh-Sch.*Sdg;
    Sraw=detA.*detB;
    HoneA=(Hae.*Sbf+Sae.*Hbf-Haf.*Sbe-Saf.*Hbe).*detB;
    HoneB=detA.*(Hcg.*Sdh+Scg.*Hdh-Hch.*Sdg-Sch.*Hdg);
    Hraw=HoneA+HoneB;
    idx=find(upperMask);
    if ~isempty(idx)
        ei=e(idx); fi=f(idx); gi=g(idx); hi=h(idx);
        detA_i=detA(idx); detB_i=detB(idx);
        Vaa=(eri_row_vectorized_cpu(td,a,ei,b,fi)-eri_row_vectorized_cpu(td,a,fi,b,ei)).*detB_i;
        Vbb=detA_i.*(eri_row_vectorized_cpu(td,c,gi,d0,hi)-eri_row_vectorized_cpu(td,c,hi,d0,gi));
        CA11=Sbf(idx); CA12=-Sbe(idx); CA21=-Saf(idx); CA22=Sae(idx);
        CB11=Sdh(idx); CB12=-Sdg(idx); CB21=-Sch(idx); CB22=Scg(idx);
        Vab = be_vab_batched_cpu(td,a,b,c,d0,ei,fi,gi,hi, ...
            CA11,CA12,CA21,CA22,CB11,CB12,CB21,CB22,opts);
        Hraw(idx)=Hraw(idx)+Vaa+Vbb+Vab;
    end
    fac=normInv(row).*normInv; srow=Sraw.*fac; hrow=Hraw.*fac;
    maskS=upperMask & (abs(srow)>=opts.dropTolS); maskH=upperMask & (abs(hrow)>=opts.dropTolH);
    if isfield(opts,'forceDiagonal') && opts.forceDiagonal && row<=M, maskS(row)=true; maskH(row)=true; end
    js=find(maskS); vs=srow(maskS); jh=find(maskH); vh=hrow(maskH);
end

function [preIdx,info] = preselect_be_working_basis_zero_coupling(B, oneCache, oneFuncs, opts, expansion, minExtra, keepOther)
    % Conservative fast preselector.  It does NOT change the neighbor list;
    % it only avoids assembling coefficients for raw functions with extremely
    % small direct coupling to the HF determinant.  The final selected system
    % is still exact-reassembled before being passed to adaptive.
    S1=oneCache.S1; H1=oneCache.H1; td=build_term_data(oneFuncs);
    opts0=opts; opts0.upperTriangle=false; opts0.dropTolS=0; opts0.dropTolH=0;
    opts0.screenERIByOverlap=false; opts0.eriCoeffTol=0; opts0.forceDiagonal=true;
    normInv=be_norm_inv(B,S1,opts0);
    avec=[B.a]; bvec=[B.b]; cvec=[B.c]; dvec=[B.d];
    [js,vs,jh,vh]=assemble_one_be_row_direct(B,1,avec,bvec,cvec,dvec,S1,H1,td,normInv,opts0);
    M=numel(B); srow=zeros(1,M); hrow=zeros(1,M); srow(js)=vs; hrow(jh)=vh;
    Ezero=hrow(1)/srow(1);
    score=abs(hrow - Ezero*srow);
    score(~isfinite(score))=0;

    target=struct('one',524,'two_ud',562,'two_uu',264,'three',64,'other',2);
    quota.one=min(nnz(strcmp({B.block},'one')), max(target.one+minExtra, ceil(expansion*target.one)));
    quota.two_ud=min(nnz(strcmp({B.block},'two_ud')), max(target.two_ud+minExtra, ceil(expansion*target.two_ud)));
    quota.two_uu=min(nnz(strcmp({B.block},'two_uu')), max(target.two_uu+minExtra, ceil(expansion*target.two_uu)));
    quota.three=min(nnz(strcmp({B.block},'three')), max(target.three+minExtra, ceil(expansion*target.three)));
    quota.other=nnz(strcmp({B.block},'other'));

    preIdx=1;
    [preIdx,sel.one]    = add_top_block(preIdx,B,score,'one',quota.one);
    [preIdx,sel.two_ud] = add_top_block(preIdx,B,score,'two_ud',quota.two_ud);
    [preIdx,sel.two_uu] = add_top_block(preIdx,B,score,'two_uu',quota.two_uu);
    [preIdx,sel.three]  = add_top_block(preIdx,B,score,'three',quota.three);
    if keepOther
        ids=find(strcmp({B.block},'other')); preIdx=[preIdx, ids(:).']; 
    end
    preIdx=unique(preIdx,'stable');
    info=struct('used',true,'Ezero',Ezero,'score',score,'quota',quota,'selected',sel,'preIdx',preIdx);
end

function [preIdx,selIds] = add_top_block(preIdx,B,score,blockName,nkeep)
    ids=find(strcmp({B.block},blockName));
    if isempty(ids) || nkeep<=0, selIds=[]; return; end
    [~,ord]=sort(score(ids),'descend');
    selIds=ids(ord(1:min(nkeep,numel(ord))));
    preIdx=[preIdx, selIds(:).'];
end

function val = get_opt(opts, name, defaultVal)
    if isfield(opts,name), val = opts.(name); else, val = defaultVal; end
end

function td = build_term_data(oneFuncs)
    % Compact term data with per-function nTerms.  This is crucial for Be:
    % HF orbitals have Q=12 terms, but every frame function has only 1 term.
    % The old v2 ERI routine looped over maxTerms^4 even for frame-frame ERIs.
    M=numel(oneFuncs); maxT=0; nTerms=zeros(M,1);
    for i=1:M
        nTerms(i)=numel(oneFuncs(i).terms);
        maxT=max(maxT,nTerms(i));
    end
    coef=zeros(M,maxT); ncoef=zeros(M,maxT); alpha=zeros(M,maxT); cx=zeros(M,maxT); cy=zeros(M,maxT); cz=zeros(M,maxT);
    for i=1:M
        for t=1:nTerms(i)
            term=oneFuncs(i).terms(t);
            coef(i,t)=term.coef; ncoef(i,t)=term.coef*(term.alpha/pi)^(3/4); alpha(i,t)=term.alpha;
            cx(i,t)=term.center(1); cy(i,t)=term.center(2); cz(i,t)=term.center(3);
        end
    end
    td=struct('coef',coef,'ncoef',ncoef,'alpha',alpha,'cx',cx,'cy',cy,'cz',cz, ...
              'maxTerms',maxT,'nTerms',nTerms);
end

function vrow = eri_row_vectorized_cpu(td, ia, ibVec, ic, idVec)
    % Grouped contracted ERI row.  Columns are grouped by their true
    % primitive term counts (nTerms(ib), nTerms(id)).  This avoids the v4
    % cost of repeatedly scanning length-n masks for tb/td levels and avoids
    % the proposed dense maxTerms padding, which is very expensive when only
    % a few HF orbitals have Q=12 terms.
    nc=numel(ibVec); vrow=zeros(1,nc);
    if nc==0, return; end

    nA=td.nTerms(ia); nC=td.nTerms(ic);
    nB=reshape(td.nTerms(ibVec),1,[]);
    nD=reshape(td.nTerms(idVec),1,[]);
    code = int32(nB(:))*1000 + int32(nD(:));
    ucode = unique(code,'stable');

    for ug=1:numel(ucode)
        mask = (code.' == ucode(ug));
        if ~any(mask), continue; end
        pos=find(mask);
        ibG=ibVec(pos); idG=idVec(pos);
        nb=nB(pos(1)); nd=nD(pos(1));
        vrow(pos)=eri_row_vectorized_cpu_fixed_counts(td,ia,ibG,ic,idG,nA,nb,nC,nd);
    end
end

function out = eri_row_vectorized_cpu_fixed_counts(td, ia, ibVec, ic, idVec, nA, nB, nC, nD)
    nc=numel(ibVec); out=zeros(1,nc);
    for ta=1:nA
        ca=td.ncoef(ia,ta); if ca==0, continue; end
        aA=td.alpha(ia,ta); Ax=td.cx(ia,ta); Ay=td.cy(ia,ta); Az=td.cz(ia,ta);
        for tc=1:nC
            cc=td.ncoef(ic,tc); if cc==0, continue; end
            aC=td.alpha(ic,tc); Cx=td.cx(ic,tc); Cy=td.cy(ic,tc); Cz=td.cz(ic,tc);
            coefAC=ca*cc;
            for tb=1:nB
                cb=td.ncoef(ibVec,tb).';
                activeB=(cb~=0); if ~any(activeB), continue; end
                aB=td.alpha(ibVec,tb).'; Bx=td.cx(ibVec,tb).'; By=td.cy(ibVec,tb).'; Bz=td.cz(ibVec,tb).';
                for td2=1:nD
                    cd=td.ncoef(idVec,td2).';
                    mask=activeB & (cd~=0); if ~any(mask), continue; end
                    idm=idVec(mask);
                    val=eri_primitive_vec_nonorm(aA,Ax,Ay,Az,aB(mask),Bx(mask),By(mask),Bz(mask), ...
                        aC,Cx,Cy,Cz,td.alpha(idm,td2).',td.cx(idm,td2).',td.cy(idm,td2).',td.cz(idm,td2).');
                    out(mask)=out(mask)+coefAC.*cb(mask).*cd(mask).*val;
                end
            end
        end
    end
end

function vrow = eri_row_vectorized_gpu(td, ia, ibVec, ic, idVec)
    % Optional GPU version.  It keeps the same grouped-term logic and uses
    % pre-normalized coefficients.  For RTX 3050, CPU parfor can still be
    % faster; therefore preferGPU=false is the default in v5.
    nc=numel(ibVec); vrow=gpuArray.zeros(1,nc);
    if nc==0, return; end
    nA=td.nTermsCPU(ia); nC=td.nTermsCPU(ic);
    nB=reshape(td.nTermsCPU(ibVec),1,[]);
    nD=reshape(td.nTermsCPU(idVec),1,[]);
    code=int32(nB(:))*1000+int32(nD(:)); ucode=unique(code,'stable');
    for ug=1:numel(ucode)
        mask=(code.'==ucode(ug)); if ~any(mask), continue; end
        pos=find(mask); ibG=ibVec(pos); idG=idVec(pos);
        nb=nB(pos(1)); nd=nD(pos(1));
        vrow(pos)=eri_row_vectorized_gpu_fixed_counts(td,ia,ibG,ic,idG,nA,nb,nC,nd);
    end
end

function out = eri_row_vectorized_gpu_fixed_counts(td, ia, ibVec, ic, idVec, nA, nB, nC, nD)
    nc=numel(ibVec); out=gpuArray.zeros(1,nc);
    for ta=1:nA
        ca=td.ncoef(ia,ta); aA=td.alpha(ia,ta); Ax=td.cx(ia,ta); Ay=td.cy(ia,ta); Az=td.cz(ia,ta);
        for tc=1:nC
            cc=td.ncoef(ic,tc); aC=td.alpha(ic,tc); Cx=td.cx(ic,tc); Cy=td.cy(ic,tc); Cz=td.cz(ic,tc);
            coefAC=ca*cc;
            for tb=1:nB
                cb=td.ncoef(ibVec,tb).';
                aB=td.alpha(ibVec,tb).'; Bx=td.cx(ibVec,tb).'; By=td.cy(ibVec,tb).'; Bz=td.cz(ibVec,tb).';
                for td2=1:nD
                    cd=td.ncoef(idVec,td2).';
                    idm=idVec;
                    val=eri_primitive_vec_gpu_nonorm(aA,Ax,Ay,Az,aB,Bx,By,Bz, ...
                        aC,Cx,Cy,Cz,td.alpha(idm,td2).',td.cx(idm,td2).',td.cy(idm,td2).',td.cz(idm,td2).');
                    out=out+coefAC.*cb.*cd.*val;
                end
            end
        end
    end
end

function val = eri_primitive_vec_gpu_nonorm(alphaA,Ax,Ay,Az,alphaB,Bx,By,Bz,alphaC,Cx,Cy,Cz,alphaD,Dx,Dy,Dz)
    p=alphaA+alphaB; q=alphaC+alphaD;
    Px=(alphaA*Ax+alphaB.*Bx)./p; Py=(alphaA*Ay+alphaB.*By)./p; Pz=(alphaA*Az+alphaB.*Bz)./p;
    Qx=(alphaC*Cx+alphaD.*Dx)./q; Qy=(alphaC*Cy+alphaD.*Dy)./q; Qz=(alphaC*Cz+alphaD.*Dz)./q;
    RAB2=(Ax-Bx).^2+(Ay-By).^2+(Az-Bz).^2;
    RCD2=(Cx-Dx).^2+(Cy-Dy).^2+(Cz-Dz).^2;
    RPQ2=(Px-Qx).^2+(Py-Qy).^2+(Pz-Qz).^2;
    Kab=exp(-(alphaA.*alphaB./(2*p)).*RAB2);
    Kcd=exp(-(alphaC.*alphaD./(2*q)).*RCD2);
    arg=(p.*q./(2*(p+q))).*RPQ2;
    val=Kab.*Kcd.*(8*sqrt(2)*pi^(5/2))./(p.*q.*sqrt(p+q)).*boys0_vec_gpu(arg);
end

function val = eri_primitive_vec_gpu(alphaA,Ax,Ay,Az,alphaB,Bx,By,Bz,alphaC,Cx,Cy,Cz,alphaD,Dx,Dy,Dz)
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
    val=Kab.*Kcd.*(8*sqrt(2)*pi^(5/2))./(p.*q.*sqrt(p+q)).*boys0_vec_gpu(arg);
end

function F = boys0_vec_gpu(t)
    F=zeros(size(t),'like',t);
    small=t<1e-10;
    F(small)=1-t(small)/3+t(small).^2/10;
    ts=t(~small);
    F(~small)=0.5*sqrt(pi).*erf(sqrt(ts))./sqrt(ts);
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

function val = eri_primitive_vec_nonorm(alphaA,Ax,Ay,Az,alphaB,Bx,By,Bz,alphaC,Cx,Cy,Cz,alphaD,Dx,Dy,Dz)
    % Primitive ERI kernel for pre-normalized coefficients.  The Gaussian
    % normalization factors are absorbed into td.ncoef, avoiding repeated
    % power evaluations inside the innermost ERI loop.
    p=alphaA+alphaB; q=alphaC+alphaD;
    Px=(alphaA*Ax+alphaB.*Bx)./p; Py=(alphaA*Ay+alphaB.*By)./p; Pz=(alphaA*Az+alphaB.*Bz)./p;
    Qx=(alphaC*Cx+alphaD.*Dx)./q; Qy=(alphaC*Cy+alphaD.*Dy)./q; Qz=(alphaC*Cz+alphaD.*Dz)./q;
    RAB2=(Ax-Bx).^2+(Ay-By).^2+(Az-Bz).^2;
    RCD2=(Cx-Dx).^2+(Cy-Dy).^2+(Cz-Dz).^2;
    RPQ2=(Px-Qx).^2+(Py-Qy).^2+(Pz-Qz).^2;
    Kab=exp(-(alphaA.*alphaB./(2*p)).*RAB2);
    Kcd=exp(-(alphaC.*alphaD./(2*q)).*RCD2);
    arg=(p.*q./(2*(p+q))).*RPQ2;
    val=Kab.*Kcd.*(8*sqrt(2)*pi^(5/2))./(p.*q.*sqrt(p+q)).*boys0_vec(arg);
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

function [E0,coeff,nkeep,info] = solve_ground_projected(H,S,mass_tol,verbose,targetEnergy)
    if nargin<5, targetEnergy=NaN; end
    H=full(0.5*(H+H')); S=full(0.5*(S+S'));
    [U,D]=eig(S); d=real(diag(D)); keep=d>mass_tol*max(d);
    if ~any(keep), error('Mass matrix projection removed all dimensions.'); end
    Uk=U(:,keep); dk=d(keep); X=Uk*diag(1./sqrt(dk)); Hp=0.5*(X'*H*X + (X'*H*X)');
    [Y,Ediag]=eig(Hp); evals=real(diag(Ediag)); [E0,pos]=min(evals);
    coeff=X*Y(:,pos); coeff=coeff/sqrt(real(coeff'*S*coeff)); nkeep=sum(keep);
    res=norm(H*coeff - E0*S*coeff)/max(1,norm(H*coeff));
    info=struct('residual',res,'nkeep',nkeep,'targetEnergy',targetEnergy);
    if verbose
        fprintf('    projected solve: nkeep=%d/%d, E=%.15f, residual=%.3e\n', nkeep, size(S,1), E0, res);
    end
end

function be = compute_block_energy_contributions(c,H,B)
    be = empty_block_energy();
    blocks = strings(numel(B),1); for i=1:numel(B), blocks(i)=string(B(i).block); end
    be.row.zero = row_energy(c,H,blocks=="zero");
    be.row.one = row_energy(c,H,blocks=="one");
    be.row.two_ud = row_energy(c,H,blocks=="two_ud");
    be.row.two_uu = row_energy(c,H,blocks=="two_uu");
    be.row.three = row_energy(c,H,blocks=="three");
    be.row.other = row_energy(c,H,blocks=="other");
    be.row.total = real(c'*H*c);
    be.diag.zero = diag_energy(c,H,blocks=="zero");
    be.diag.one = diag_energy(c,H,blocks=="one");
    be.diag.two_ud = diag_energy(c,H,blocks=="two_ud");
    be.diag.two_uu = diag_energy(c,H,blocks=="two_uu");
    be.diag.three = diag_energy(c,H,blocks=="three");
    be.diag.other = diag_energy(c,H,blocks=="other");
    be.diag.total = be.diag.zero + be.diag.one + be.diag.two_ud + be.diag.two_uu + be.diag.three + be.diag.other;
end

function be = empty_block_energy()
    fields = {'zero','one','two_ud','two_uu','three','other','total'};
    for k=1:numel(fields), row.(fields{k})=0; diagv.(fields{k})=0; end 
    be=struct('row',row,'diag',diagv);
end

function e = row_energy(c,H,idx)
    e = real(c(idx)' * (H(idx,:) * c));
end

function e = diag_energy(c,H,idx)
    e = real(c(idx)' * (H(idx,idx) * c(idx)));
end

function E = get_hf_energy(hf)
    if isfield(hf,'E'), E = hf.E; else, E = NaN; end
end
