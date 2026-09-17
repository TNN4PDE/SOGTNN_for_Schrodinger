%% JH_He_Table51_part4_adaptive_v11_projected_solver_nosparseGE.m
% Part 4 of the J.H.-style He Table 5.1 reproduction.
%
% Goal:
%   Starting from the coarsened B^[0] system from Part 3, perform
%   Algorithm 4.1 adaptive expansion/refinement for the He atom.
%
% Input:
%   JH_He_Bthree0_coarsened_M310_nu13.mat
%   JH_He_Bthree0_raw_assembled_M550.mat
%   JH_He_HF_Q9_orbital_v2.mat
%
% Main J.H. settings:
%   He atom, N=2, M_S=0, N_up=1, N_down=1.
%   sigma=1, c=1/2, L=7 for the initial frame.
%   epsilon_kappa = 2^(-kappa).
%   Initial B^[0] is coarse_delta(B_three^[0]).
%
% Basis blocks:
%   zero:
%       psi(r1) psi(r2)
%   one_up:
%       phi(r1) psi(r2)
%   one_down:
%       psi(r1) phi(r2)
%   two_ud:
%       phi_i(r1) phi_j(r2)
%
% Important:
%   In the initial B_three^[0], two_ud is diagonal phi_i x phi_i.
%   During adaptive refinement, Algorithm 4.1 replaces one particle function
%   at a time; therefore two_ud becomes a general pair phi_i x phi_j.
%
% Importance score:
%   Default uses J.H. partial-orthogonalized coefficients with m1=1:
%       score_mu = |(S_B v)_mu|.
%   Set importanceMode='raw' for debugging.

clear; clc; close all;
format long e;

%% ============================================================
% 0. Parameters
% =============================================================
coarseFiles = dir('JH_He_Bthree0_coarsened_M*_nu*.mat');
if isempty(coarseFiles)
    error('Cannot find JH_He_Bthree0_coarsened_M*_nu*.mat. Run Part 3 first.');
end
[~,iLatest] = max([coarseFiles.datenum]);
coarseFile = coarseFiles(iLatest).name;

fprintf('\n============================================================\n');
fprintf('J.H. He Table 5.1 Part 4: adaptive kappa iteration FINAL projected solver, no generalized sparse eigs + M-record timing\n');
fprintf('Loading coarsened system: %s\n', coarseFile);
fprintf('============================================================\n\n');

load(coarseFile, 'part3');
load(part3.inputFile, 'part2');

% Physical/numerical parameters.
Zcharge = part2.params.Zcharge;
Rnuc = part2.params.Rnuc;
sigma0 = part2.params.sigma0;
cscale = part2.params.cscale;
D = part2.params.D;
ZsetMode = part2.params.ZsetMode;
centerConvention = part2.params.centerConvention;
centerForNeighbors = 'actual';
neighborMode = 'cross';

E_ref_exact = -2.9037243770341196;

% Adaptive controls.
kappa_max = 9;              % Table 5.1 goes to kappa=11; set 9 for quick tests.
max_inner = 30;
max_basis_allowed = 40000;   % dense matrices at large M require substantial memory.
updateMode = 'monotone_union';  % Best matched practical mode from tests: cross + monotone_union.

% Importance mode:
%   'partial_orthogonal' is J.H.-style for particle-wise splitting.
%   'raw' is a useful debugging alternative.
importanceMode = 'partial_orthogonal';

% Solver controls.
mass_tol = 1e-10;
solverOpts.directEigLimit = 3500;
solverOpts.fullFallbackLimit = 7000;
solverOpts.shiftRelGap = 1e-4;
solverOpts.shiftAbsGap = 1e-6;
solverOpts.eigsTol = 1e-11;
solverOpts.eigsMaxit = 1000;
solverOpts.eigsP = 100;
solverOpts.resTol = 1e-7;
verboseEig = true;

% Assembly / memory controls.
% This version is sparse-stream by default. It stores only sparse S,H and never
% stores Hone/Vee separately. Dense row blocks are thresholded, inserted into
% sparse matrices, and immediately cleared.
assemblyOpts = struct();
assemblyOpts.mode = 'cpu_parfor_sparse';    % 'cpu_serial_sparse', 'cpu_parfor_sparse', or 'gpu_serial_sparse'
assemblyOpts.rowBlockSize = 24;            % rows per streaming chunk; reduce to 8/16 if RAM is tight
assemblyOpts.dropTolS = 1e-12;             % sparse threshold for overlap matrix entries; lower to 1e-13 for stricter check
assemblyOpts.dropTolH = 1e-11;             % sparse threshold for Hamiltonian entries; lower to 1e-12 for stricter check
assemblyOpts.startPool = true;
assemblyOpts.showRowProgress = true;
assemblyOpts.useSingleGPUWorking = false;  % GPU work arrays in single if true; faster but less strict
assemblyOpts.symmetrizeAfterInsert = true;
assemblyOpts.rebuildEveryAssembly = true;   % memory-min: never append/copy old matrices; rebuild current S,H only
assemblyOpts.keepMatrixCache = false;       % do not keep S,H inside twoCache after returning them

% Final solver controls.
% v11 disables generalized sparse eigs(H,S,...).  All solved states use the
% old stable mass-projection route: first remove the null/near-null mass
% directions, then solve a standard eigenproblem in the projected space.
solverOpts.denseLimit = inf;              % force trusted projected solver for all M by default
solverOpts.projectedDenseLimit = 40000;   % safety limit for full(S),full(H) projection; lower if RAM is tight
solverOpts.useGeneralizedSparseEigs = false;
solverOpts.largeProjectedMode = 'full_mass_projection'; % exact/stable, but memory-heavy
solverOpts.allowSparseMassProjectionFallback = false;   % optional experimental fallback; default OFF
solverOpts.massEigsInitialK = 2048;
solverOpts.massEigsStepFactor = 1.5;
solverOpts.massEigsMaxK = 25000;
solverOpts.massEigsTol = 1e-10;
solverOpts.massEigsMaxit = 2000;
solverOpts.massEigsP = 200;
solverOpts.sparseSigmaOffset = 1e-3;       % kept only for legacy metadata; not used by generalized eigs
solverOpts.spdJitter = 0;                 % no S+tau*I copy
solverOpts.exactLowerBound = E_ref_exact; % variational guard
solverOpts.energyLowerTol = 1e-5;
solverOpts.rayleighEigTol = 1e-5;
solverOpts.minPositiveSNorm = 1e-14;
solverOpts.rejectBelowExact = true;
solverOpts.suppressEigsWarnings = true;

% Checkpoint controls: save only basis/config/records, not matrices.
checkpointOpts = struct();
checkpointOpts.saveBasisOnly = true;
checkpointOpts.dropMatrixCacheAfterKappa = true;  % lowest RAM: save basis only and clear matrices after every kappa

% M-record / extrapolation controls.
% Every solved inner state is recorded as one M-sample, so we can fit
%   err(M),  time(M),  and cumulative wall time(M)
% instead of wasting intermediate adaptive results.
recordOpts = struct();
recordOpts.computeBlockEnergyEachInner = true;   % set false to save a little time at huge M
recordOpts.saveRecordsEveryInner = true;         % records-only live checkpoint, very small
recordOpts.saveBasisEveryMRecord = true;         % basis/config checkpoint for each solved M, no matrices
recordOpts.recordsLiveFile = 'JH_He_Table51_M_records_live.mat';
recordOpts.csvFile = 'JH_He_Table51_M_records.csv';

% Optional practical early stop. For strict Algorithm 4.1 set enable=false.
innerStop.enable = false;
innerStop.minInner = 2;
innerStop.minAdded = 1000;
innerStop.minGrowthRel = 0.25;
innerStop.minRelGain = 0.01;
innerStop.minAbsGain = 1e-7;

fprintf('Parameters:\n');
fprintf('  Z=%g, sigma=%g, c=%g, L0=%d\n', Zcharge, sigma0, cscale, part2.params.L0);
fprintf('  kappa_max=%d, updateMode=%s, importanceMode=%s\n', ...
    kappa_max, updateMode, importanceMode);
fprintf('  max_basis_allowed=%d, mass_tol=%.1e\n\n', max_basis_allowed, mass_tol);

%% ============================================================
% 1. Initialize one-particle and two-particle caches from Part 2/3
% =============================================================
oneFuncs = part2.oneFuncs;         % includes ref + initial frame functions
twoBasis = part3.twoBasis0;        % selected coarsened two-electron basis, IDs refer to oneFuncs

% Rebuild one-particle integral cache using Part 2 data for initial pool.
oneCache = struct();
oneCache.S1 = part2.S1;
oneCache.H1 = part2.H1;
oneCache.keys = onefunc_keys(oneFuncs);
oneCache.M = numel(oneFuncs);
oneCache.Zcharge = Zcharge;
oneCache.Rnuc = Rnuc;

% Two-electron matrix cache is intentionally empty in memory-min mode.
% We save/reuse only basis configuration; S,H are rebuilt analytically for each solve.
twoCache = reset_two_cache();

% ERI cache disabled in memory-min mode. Row assembly is vectorized and streamed.
eriCache = [];

% Dictionaries for one-particle and two-particle basis identity.
oneKeyMap = containers.Map('KeyType','char','ValueType','double');
for i = 1:numel(oneFuncs)
    oneKeyMap(char(oneFuncs(i).key)) = i;
end

twoKeyMap = containers.Map('KeyType','char','ValueType','double');
for i = 1:numel(twoBasis)
    twoKeyMap(char(twoBasis(i).key)) = i;
end

counts0 = count_blocks(twoBasis);
fprintf('Initial B^[0] from coarsening:\n');
fprintf('  M=%d | Mzero=%d | Mone=%d | Mtwo_ud=%d\n', ...
    counts0.total, counts0.zero, counts0.one_total, counts0.two_ud);
fprintf('  E0(coarse) = %.15f\n', part3.E0);
fprintf('  Table 5.1 target kappa=1: M=310, E=-2.897506, Mone=198, Mtwo=111\n\n');

%% ============================================================
% 2. Adaptive kappa loop
% =============================================================
records = repmat(empty_record(), kappa_max, 1);

% recordsByM stores EVERY solved inner state, indexed by the actual basis size M.
% This is the main output for extrapolation:
%   error vs M, and wall time vs M.
recordsByM = repmat(empty_m_record(), 0, 1);
sampleID = 0;
tGlobal = tic;

for kappa = 1:kappa_max
    epsK = 2^(-kappa);

    fprintf('\n############################################################\n');
    fprintf('kappa = %d, epsilon = %.6e\n', kappa, epsK);
    fprintf('############################################################\n');

    changed = true;
    inner = 0;

    prevAccepted = struct('valid',false);
    finalInfo = struct();

    while changed
        inner = inner + 1;
        tInnerTotal = tic;

        if inner > max_inner
            warning('Reached max_inner=%d at kappa=%d.', max_inner, kappa);
            break;
        end

        M = numel(twoBasis);
        if M > max_basis_allowed
            warning('M=%d exceeds max_basis_allowed=%d. Stop adaptive loop.', M, max_basis_allowed);
            changed = false;
            break;
        end

        fprintf('\n  inner %d | M=%d | Mone/two = ', inner, M);
        tmpCounts = count_blocks(twoBasis);
        fprintf('%d / %d\n', tmpCounts.one_total, tmpCounts.two_ud);

        % Update matrices if needed.
        ticAsm = tic;
        [oneCache, twoCache, S2, H2, asmInfo, eriCache] = update_two_matrix_cache_sparse( ...
            oneCache, twoCache, oneFuncs, twoBasis, Zcharge, Rnuc, eriCache, assemblyOpts);
        tAsm = toc(ticAsm);

        nnzS = nnz(S2);
        nnzH = nnz(H2);
        sparseGB = estimate_sparse_pair_gb(S2, H2);
        fprintf('  assembly/cache: %.2f s | reused=%d | oldM=%d | addedM=%d | nnz(S/H)=%.3e/%.3e | est sparse GB=%.3f\n', ...
            tAsm, asmInfo.reused, asmInfo.oldM, asmInfo.addedM, nnzS, nnzH, sparseGB);

        % Solve.
        ticSol = tic;
        [E0, coeff, nkeep, solverInfo] = solve_ground_generalized_sparseaware( ...
            H2, S2, mass_tol, verboseEig, E_ref_exact, solverOpts);
        tSol = toc(ticSol);

        err = abs(E0 - E_ref_exact);
        fprintf('  solve: %.2f s | E=%.15f | err=%.3e | kept=%d | solver=%s\n', ...
            tSol, E0, err, nkeep, solverInfo.method);

        % Importance scores.
        ticImp = tic;
        switch lower(importanceMode)
            case 'partial_orthogonal'
                scores = partial_orthogonal_importance(coeff, S2);
            case 'raw'
                scores = abs(coeff);
            otherwise
                error('Unknown importanceMode.');
        end
        tImp = toc(ticImp);

        important = scores >= epsK;
        nImp = nnz(important);
        if any(scores > 0)
            scoreMinNonzero = min(scores(scores>0));
        else
            scoreMinNonzero = NaN;
        end
        scoreMax = max(scores);

        fprintf('  important=%d / %d | score max=%.3e, min(nonzero)=%.3e | importance %.2f s\n', ...
            nImp, M, scoreMax, scoreMinNonzero, tImp);

        % Optional block-energy record for this solved M.
        ticBE = tic;
        if recordOpts.computeBlockEnergyEachInner
            innerBlockEnergy = compute_block_energy_contributions(coeff, H2, twoBasis);
        else
            innerBlockEnergy = empty_block_energy();
        end
        tBE = toc(ticBE);

        % Store current solved state in case we need final record.
        finalInfo.E0 = E0;
        finalInfo.err = err;
        finalInfo.coeff = coeff;
        finalInfo.nkeep = nkeep;
        finalInfo.solverInfo = solverInfo;
        finalInfo.inner = inner;
        finalInfo.M = numel(twoBasis);
        finalInfo.blockEnergy = innerBlockEnergy;
        finalInfo.tAsm = tAsm;
        finalInfo.tSol = tSol;
        finalInfo.tImp = tImp;
        finalInfo.tBE = tBE;
        finalInfo.nnzS = nnzS;
        finalInfo.nnzH = nnzH;
        finalInfo.sparseGB = sparseGB;

        % Optional practical early stop by current-vs-previous gain.
        if innerStop.enable && inner >= innerStop.minInner && prevAccepted.valid
            addedFromPrev = M - prevAccepted.M;
            growthRel = addedFromPrev / max(prevAccepted.M,1);
            absGain = prevAccepted.err - err;
            relGain = absGain / max(prevAccepted.err, realmin);

            if addedFromPrev >= innerStop.minAdded && growthRel >= innerStop.minGrowthRel && ...
               (absGain <= innerStop.minAbsGain || relGain <= innerStop.minRelGain)
                fprintf('  inner-stop: added=%d growth=%.2f%%, gain=%.3e rel=%.2f%%. Roll back.\n', ...
                    addedFromPrev, 100*growthRel, absGain, 100*relGain);

                twoBasis = prevAccepted.twoBasis;
                oneFuncs = prevAccepted.oneFuncs;
                oneCache = prevAccepted.oneCache;
                twoCache = prevAccepted.twoCache;
                eriCache = prevAccepted.eriCache;
                finalInfo = prevAccepted.finalInfo;
                changed = false;
                break;
            end
        end

        % Grow from important basis functions.
        ticGrow = tic;
        oldTwoKeys = twobasis_keys(twoBasis);
        oldM = numel(twoBasis);

        [growBasis, oneFuncs, oneKeyMap] = grow_important_twobody_basis( ...
            twoBasis(important), oneFuncs, oneKeyMap, sigma0, cscale, D, ...
            ZsetMode, centerConvention, centerForNeighbors, neighborMode);

        switch lower(updateMode)
            case 'monotone_union'
                candidate = [twoBasis, growBasis];
            case 'paper_reset'
                % Algorithm 4.1 step 4b proposal: B_prop = Ng(Bprime) union Ng+(Bprime).
                % We only ACCEPT this proposal if it contains at least one basis
                % not present in B_old. If not, the current B_old is the closed
                % space to record for this kappa.
                candidate = growBasis;
            otherwise
                error('Unknown updateMode: %s', updateMode);
        end

        [twoBasisNew, twoKeyMap] = unique_twobasis(candidate);
        newTwoKeys = twobasis_keys(twoBasisNew);

        added = setdiff(newTwoKeys, oldTwoKeys);
        nGrowRaw = numel(candidate);
        nNewM = numel(twoBasisNew);
        nAdded = numel(added);
        tGrow = toc(ticGrow);
        fprintf('  grown raw=%d | unique new M=%d | added=%d | grow %.2f s\n', ...
            nGrowRaw, nNewM, nAdded, tGrow);

        % Save this solved inner state as an M-indexed sample for extrapolation.
        sampleID = sampleID + 1;
        tInnerNoSave = toc(tInnerTotal);
        [recordsByM, tSaveRecord, basisFile] = append_M_record_and_checkpoint( ...
            recordsByM, sampleID, kappa, inner, epsK, M, tmpCounts, E0, err, ...
            innerBlockEnergy, nkeep, solverInfo, tAsm, tSol, tImp, tBE, tGrow, ...
            tInnerNoSave, toc(tGlobal), nImp, scoreMax, scoreMinNonzero, ...
            nGrowRaw, nNewM, nAdded, nnzS, nnzH, sparseGB, ...
            oneFuncs, twoBasis, records, recordOpts, checkpointOpts, ...
            sigma0, cscale, Zcharge, ZsetMode, centerConvention, centerForNeighbors, ...
            neighborMode, importanceMode, mass_tol, updateMode, assemblyOpts, solverOpts);

        fprintf('  M-record #%d | kappa.inner=%d.%d | M=%d | err=%.3e | tAsm=%.1fs tSol=%.1fs tGrow=%.1fs tSave=%.1fs | cum=%.1fs\n', ...
            sampleID, kappa, inner, M, err, tAsm, tSol, tGrow, tSaveRecord, toc(tGlobal));

        if innerStop.enable
            % Only keep rollback snapshots when early stopping is enabled.
            % Otherwise this duplicates the sparse matrix cache in memory.
            prevAccepted.valid = true;
            prevAccepted.M = M;
            prevAccepted.err = err;
            prevAccepted.twoBasis = twoBasis;
            prevAccepted.oneFuncs = oneFuncs;
            prevAccepted.oneCache = oneCache;
            prevAccepted.twoCache = twoCache;
            prevAccepted.eriCache = eriCache;
            prevAccepted.finalInfo = finalInfo;
        end

        changed = ~isempty(added);
        if changed
            twoBasis = twoBasisNew;
            % Memory-min: current S2/H2 correspond to old B and are no longer needed.
            clear S2 H2 scores important;
            twoCache = reset_two_cache();
            if exist('OCTAVE_VERSION','builtin') == 0
                drawnow;
            end
        else
            fprintf('  inner closure reached for kappa=%d; keep current B_old with M=%d.\n', ...
                kappa, M);
        end
    end

    % Final matrices/info should already correspond to current twoBasis when closure is reached.
    % Only rebuild if loop ended before a valid final solve was retained.
    if isfield(finalInfo,'E0') && isfield(finalInfo,'M') && numel(twoBasis)==finalInfo.M && exist('H2','var') && exist('S2','var')
        E0 = finalInfo.E0;
        coeff = finalInfo.coeff;
        nkeep = finalInfo.nkeep;
        solverInfo = finalInfo.solverInfo;
        innerUsed = finalInfo.inner;
    else
        twoCache = reset_two_cache();
        [oneCache, twoCache, S2, H2, asmInfo, eriCache] = update_two_matrix_cache_sparse( ...
            oneCache, twoCache, oneFuncs, twoBasis, Zcharge, Rnuc, eriCache, assemblyOpts);
        [E0, coeff, nkeep, solverInfo] = solve_ground_generalized_sparseaware( ...
            H2, S2, mass_tol, verboseEig, E_ref_exact, solverOpts);
        innerUsed = inner;
    end

    counts = count_blocks(twoBasis);
    blockEnergy = compute_block_energy_contributions(coeff, H2, twoBasis);
    err = abs(E0 - E_ref_exact);

    records(kappa).kappa = kappa;
    records(kappa).epsilon = epsK;
    records(kappa).M = counts.total;
    records(kappa).Mzero = counts.zero;
    records(kappa).Mone = counts.one_total;
    records(kappa).Mtwo_ud = counts.two_ud;
    records(kappa).E = E0;
    records(kappa).err = err;
    records(kappa).inner = innerUsed;
    records(kappa).kept = nkeep;
    records(kappa).solverMethod = string(solverInfo.method);
    records(kappa).solverResidual = solverInfo.residual;
    records(kappa).Ezero = blockEnergy.row.zero;
    records(kappa).Eone = blockEnergy.row.one;
    records(kappa).Etwo_ud = blockEnergy.row.two_ud;
    if isfield(finalInfo,'tAsm'), records(kappa).assemblyTime = finalInfo.tAsm; end
    if isfield(finalInfo,'tSol'), records(kappa).solveTime = finalInfo.tSol; end
    if isfield(finalInfo,'tImp'), records(kappa).importanceTime = finalInfo.tImp; end
    if isfield(finalInfo,'tBE'), records(kappa).blockEnergyTime = finalInfo.tBE; end
    records(kappa).cumulativeTime = toc(tGlobal);

    fprintf('\n>>> record kappa=%d | M=%d | E=%.15f | err=%.3e | Mone=%d | Mtwo=%d | inner=%d\n', ...
        kappa, counts.total, E0, err, counts.one_total, counts.two_ud, innerUsed);
    fprintf('    row energy: Ezero=%.6f | Eone=%.6f | Etwo=%.6f | sum=%.15f\n', ...
        blockEnergy.row.zero, blockEnergy.row.one, blockEnergy.row.two_ud, blockEnergy.row.total);

    saveNameTmp = sprintf('JH_He_Table51_basis_checkpoint_kappa%d_M%d.mat', kappa, counts.total);
    adaptiveState = struct('records',records, 'recordsByM',recordsByM, 'kappa',kappa, 'oneFuncs',oneFuncs, ...
        'twoBasis',twoBasis, ...
        'params',struct('sigma0',sigma0,'cscale',cscale,'Zcharge',Zcharge, ...
        'ZsetMode',ZsetMode,'centerConvention',centerConvention, ...
        'centerForNeighbors',centerForNeighbors,'neighborMode',neighborMode, ...
        'importanceMode',importanceMode,'mass_tol',mass_tol,'updateMode',updateMode, ...
        'assemblyOpts',assemblyOpts,'solverOpts',solverOpts));
    save(saveNameTmp, 'adaptiveState', '-v7.3');

    if checkpointOpts.dropMatrixCacheAfterKappa
        twoCache = reset_two_cache();
        clear S2 H2 coeff scores important blockEnergy;
        if exist('pack','file') == 2
            try pack; catch, end
        end
    end
end

%% ============================================================
% 3. Print Table 5.1 style summary and save
% =============================================================
fprintf('\n==================== TABLE 5.1 STYLE SUMMARY ====================\n');
fprintf(' kappa      M        E                    err          Mone      Mtwo       Ezero      Eone       Etwo\n');
for k = 1:kappa_max
    r = records(k);
    fprintf('%5d  %7d  %.15f  %.3e  %8d  %8d  %9.2f  %9.2f  %9.2f\n', ...
        r.kappa, r.M, r.E, r.err, r.Mone, r.Mtwo_ud, r.Ezero, r.Eone, r.Etwo_ud);
end

recordsByM_unique = unique_records_by_M_keep_best(recordsByM);

fprintf('\n==================== M-INDEXED INNER RECORDS FOR EXTRAPOLATION ====================\n');
fprintf(' sample  k.i       M        E                    err        tAsm     tSol    tGrow   tInner  tCum\n');
for q = 1:numel(recordsByM)
    r = recordsByM(q);
    fprintf('%6d  %2d.%02d  %7d  %.15f  %.3e  %8.1f %8.1f %8.1f %8.1f %8.1f\n', ...
        r.sampleID, r.kappa, r.inner, r.M, r.E, r.err, ...
        r.assemblyTime, r.solveTime, r.growTime, r.totalInnerTimeNoSave, r.cumulativeTime);
end

saveName = sprintf('JH_He_Table51_adaptive_basis_final_kappa%d_M%d.mat', ...
    records(kappa_max).kappa, records(kappa_max).M);
finalState = struct('records',records,'recordsByM',recordsByM, ...
    'recordsByM_unique',recordsByM_unique, ...
    'oneFuncs',oneFuncs,'twoBasis',twoBasis, ...
    'params',struct('sigma0',sigma0,'cscale',cscale,'Zcharge',Zcharge, ...
    'ZsetMode',ZsetMode,'centerConvention',centerConvention, ...
    'centerForNeighbors',centerForNeighbors,'neighborMode',neighborMode, ...
    'importanceMode',importanceMode,'mass_tol',mass_tol,'updateMode',updateMode, ...
    'assemblyOpts',assemblyOpts,'solverOpts',solverOpts,'recordOpts',recordOpts));
save(saveName, 'finalState', '-v7.3');

try
    Trec = struct2table(recordsByM);
    writetable(Trec, recordOpts.csvFile);
    fprintf('\nSaved M-record CSV: %s\n', recordOpts.csvFile);
catch ME
    warning('Could not write M-record CSV: %s', ME);
end

fprintf('\nSaved final adaptive basis/config only: %s\n', saveName);

%% ========================================================================
% Local functions: keys and uniqueness
% ========================================================================

function r = empty_record()
    r = struct('kappa',NaN,'epsilon',NaN,'M',NaN,'Mzero',NaN, ...
        'Mone',NaN,'Mtwo_ud',NaN,'E',NaN,'err',NaN,'inner',NaN, ...
        'kept',NaN,'solverMethod',"",'solverResidual',NaN, ...
        'Ezero',NaN,'Eone',NaN,'Etwo_ud',NaN, ...
        'assemblyTime',NaN,'solveTime',NaN,'importanceTime',NaN, ...
        'blockEnergyTime',NaN,'cumulativeTime',NaN);
end

function r = empty_m_record()
    r = struct('sampleID',NaN,'kappa',NaN,'inner',NaN,'epsilon',NaN, ...
        'M',NaN,'Mzero',NaN,'Mone',NaN,'Mtwo_ud',NaN, ...
        'E',NaN,'err',NaN,'Ezero',NaN,'Eone',NaN,'Etwo_ud',NaN, ...
        'kept',NaN,'solverMethod',"",'solverResidual',NaN, ...
        'assemblyTime',NaN,'solveTime',NaN,'importanceTime',NaN, ...
        'blockEnergyTime',NaN,'growTime',NaN,'saveTime',NaN, ...
        'totalInnerTimeNoSave',NaN,'totalInnerTimeWithSave',NaN, ...
        'cumulativeTime',NaN,'nImportant',NaN,'scoreMax',NaN, ...
        'scoreMinNonzero',NaN,'nGrowRaw',NaN,'nNewM',NaN,'nAdded',NaN, ...
        'nnzS',NaN,'nnzH',NaN,'sparseGB_est',NaN,'basisFile',"");
end

function be = empty_block_energy()
    be = struct();
    be.row = struct('zero',NaN,'one',NaN,'two_ud',NaN,'total',NaN);
end

function [recordsByM, tSave, basisFile] = append_M_record_and_checkpoint( ...
    recordsByM, sampleID, kappa, inner, epsK, M, counts, E0, err, blockEnergy, ...
    nkeep, solverInfo, tAsm, tSol, tImp, tBE, tGrow, tInnerNoSave, tCum, ...
    nImp, scoreMax, scoreMinNonzero, nGrowRaw, nNewM, nAdded, nnzS, nnzH, sparseGB, ...
    oneFuncs, twoBasis, records, recordOpts, ~, ...
    sigma0, cscale, Zcharge, ZsetMode, centerConvention, centerForNeighbors, ...
    neighborMode, importanceMode, mass_tol, updateMode, assemblyOpts, solverOpts)

    tSaveClock = tic;
    basisFile = "";

    r = empty_m_record();
    r.sampleID = sampleID;
    r.kappa = kappa;
    r.inner = inner;
    r.epsilon = epsK;
    r.M = M;
    r.Mzero = counts.zero;
    r.Mone = counts.one_total;
    r.Mtwo_ud = counts.two_ud;
    r.E = E0;
    r.err = err;
    r.Ezero = blockEnergy.row.zero;
    r.Eone = blockEnergy.row.one;
    r.Etwo_ud = blockEnergy.row.two_ud;
    r.kept = nkeep;
    r.solverMethod = string(solverInfo.method);
    r.solverResidual = solverInfo.residual;
    r.assemblyTime = tAsm;
    r.solveTime = tSol;
    r.importanceTime = tImp;
    r.blockEnergyTime = tBE;
    r.growTime = tGrow;
    r.totalInnerTimeNoSave = tInnerNoSave;
    r.cumulativeTime = tCum;
    r.nImportant = nImp;
    r.scoreMax = scoreMax;
    r.scoreMinNonzero = scoreMinNonzero;
    r.nGrowRaw = nGrowRaw;
    r.nNewM = nNewM;
    r.nAdded = nAdded;
    r.nnzS = nnzS;
    r.nnzH = nnzH;
    r.sparseGB_est = sparseGB;

    recordsByM(end+1,1) = r;

    if isfield(recordOpts,'saveBasisEveryMRecord') && recordOpts.saveBasisEveryMRecord
        basisFile = string(sprintf('JH_He_Table51_basis_byM_sample%04d_kappa%d_inner%d_M%d.mat', ...
            sampleID, kappa, inner, M));
        basisRecord = struct();
        basisRecord.sampleID = sampleID;
        basisRecord.kappa = kappa;
        basisRecord.inner = inner;
        basisRecord.M = M;
        basisRecord.oneFuncs = oneFuncs;
        basisRecord.twoBasis = twoBasis;
        basisRecord.record = recordsByM(end);
        basisRecord.params = struct('sigma0',sigma0,'cscale',cscale,'Zcharge',Zcharge, ...
            'ZsetMode',ZsetMode,'centerConvention',centerConvention, ...
            'centerForNeighbors',centerForNeighbors,'neighborMode',neighborMode, ...
            'importanceMode',importanceMode,'mass_tol',mass_tol,'updateMode',updateMode, ...
            'assemblyOpts',assemblyOpts,'solverOpts',solverOpts);
        save(char(basisFile), 'basisRecord', '-v7.3');
    end

    if isfield(recordOpts,'saveRecordsEveryInner') && recordOpts.saveRecordsEveryInner
        save(recordOpts.recordsLiveFile, 'recordsByM', 'records', '-v7.3');
    end

    tSave = toc(tSaveClock);
    recordsByM(end).saveTime = tSave;
    recordsByM(end).basisFile = basisFile;
    recordsByM(end).totalInnerTimeWithSave = tInnerNoSave + tSave;
end

function recordsUnique = unique_records_by_M_keep_best(recordsByM)
    if isempty(recordsByM)
        recordsUnique = recordsByM;
        return;
    end

    Mvals = [recordsByM.M].';
    [Mu,~,ic] = unique(Mvals, 'stable');
    recordsUnique = repmat(empty_m_record(), numel(Mu), 1);

    for i = 1:numel(Mu)
        idx = find(ic == i);
        errs = [recordsByM(idx).err];
        [~,jbest] = min(errs);
        recordsUnique(i) = recordsByM(idx(jbest));
    end
end

function gb = estimate_sparse_pair_gb(S,H)
    % Approximate MATLAB sparse memory: values + row indices + col pointers.
    % This is an estimate only, but is useful for monitoring scaling.
    ns = nnz(S);
    nh = nnz(H);
    m = size(S,1);
    gb = (16*(ns + nh) + 8*2*(m + 1)) / 1024^3;
end

function keys = onefunc_keys(oneFuncs)
    keys = strings(numel(oneFuncs),1);
    for i = 1:numel(oneFuncs)
        keys(i) = string(oneFuncs(i).key);
    end
end

function keys = twobasis_keys(twoBasis)
    keys = strings(numel(twoBasis),1);
    for i = 1:numel(twoBasis)
        keys(i) = string(twoBasis(i).key);
    end
end

function [Buniq, map] = unique_twobasis(B)
    keys = twobasis_keys(B);
    [~,ia] = unique(keys,'stable');
    Buniq = B(ia);
    map = containers.Map('KeyType','char','ValueType','double');
    for i = 1:numel(Buniq)
        map(char(Buniq(i).key)) = i;
    end
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

%% ========================================================================
% Local functions: adaptive growth
% ========================================================================

function [growBasis, oneFuncs, oneKeyMap] = grow_important_twobody_basis( ...
    importantBasis, oneFuncs, oneKeyMap, sigma0, cscale, D, ZsetMode, ...
    centerConvention, centerForNeighbors, neighborMode)

    out = {};
    k = 0;

    idxRef = 1;

    for ib = 1:numel(importantBasis)
        B = importantBasis(ib);
        block = string(B.block);

        switch block
            case "zero"
                % Keep Phi_1 = psi tensor psi. Without this, strict paper_reset
                % would remove the rank-1 reference although J.H.s S_B construction
                % assumes Phi_1 is always the zero block.
                k = k + 1;
                out{k} = B; %#ok<AGROW>

            case "one_up"
                f = oneFuncs(B.left);
                neigh = onefunc_neighbors_both(f, sigma0, cscale, D, ZsetMode, ...
                    centerConvention, centerForNeighbors, neighborMode);

                for n = 1:numel(neigh)
                    [oneFuncs, oneKeyMap, nid] = register_onefunc(oneFuncs, oneKeyMap, neigh(n));
                    k = k + 1;
                    out{k} = make_twobody_basis('one_up', nid, idxRef); %#ok<AGROW>
                end

            case "one_down"
                f = oneFuncs(B.right);
                neigh = onefunc_neighbors_both(f, sigma0, cscale, D, ZsetMode, ...
                    centerConvention, centerForNeighbors, neighborMode);

                for n = 1:numel(neigh)
                    [oneFuncs, oneKeyMap, nid] = register_onefunc(oneFuncs, oneKeyMap, neigh(n));
                    k = k + 1;
                    out{k} = make_twobody_basis('one_down', idxRef, nid); %#ok<AGROW>
                end

            case "two_ud"
                fL = oneFuncs(B.left);
                fR = oneFuncs(B.right);

                neighL = onefunc_neighbors_both(fL, sigma0, cscale, D, ZsetMode, ...
                    centerConvention, centerForNeighbors, neighborMode);
                neighR = onefunc_neighbors_both(fR, sigma0, cscale, D, ZsetMode, ...
                    centerConvention, centerForNeighbors, neighborMode);

                for n = 1:numel(neighL)
                    [oneFuncs, oneKeyMap, nid] = register_onefunc(oneFuncs, oneKeyMap, neighL(n));
                    k = k + 1;
                    out{k} = make_twobody_basis('two_ud', nid, B.right); %#ok<AGROW>
                end

                for n = 1:numel(neighR)
                    [oneFuncs, oneKeyMap, nid] = register_onefunc(oneFuncs, oneKeyMap, neighR(n));
                    k = k + 1;
                    out{k} = make_twobody_basis('two_ud', B.left, nid); %#ok<AGROW>
                end

            otherwise
                error('Unknown block type: %s', block);
        end
    end

    if isempty(out)
        growBasis = importantBasis([]);
    else
        growBasis = [out{:}];
    end
end

function [oneFuncs, oneKeyMap, id] = register_onefunc(oneFuncs, oneKeyMap, f)
    key = char(f.key);
    if isKey(oneKeyMap, key)
        id = oneKeyMap(key);
    else
        id = numel(oneFuncs) + 1;
        oneFuncs(id) = f; 
        oneKeyMap(key) = id;
    end
end

function b = make_twobody_basis(block, leftID, rightID)
    b = struct();
    b.block = char(block);
    b.left = leftID;
    b.right = rightID;
    b.frameID = 0;
    b.key = string(sprintf('%s_L%d_R%d', char(block), leftID, rightID));
end

function neigh = onefunc_neighbors_both(f, sigma0, cscale, D, ZsetMode, ...
    centerConvention, centerForNeighbors, neighborMode)
    if ~isfield(f,'kind') || string(f.kind) ~= "frame"
        neigh = f([]);
        return;
    end
    n1 = neighbors_Ng_onefunc(f, sigma0, cscale, D, ZsetMode, ...
        centerConvention, centerForNeighbors, neighborMode);
    n2 = neighbors_Ngplus_onefunc(f, sigma0, cscale, D, ZsetMode, ...
        centerConvention, centerForNeighbors, neighborMode);
    neigh = unique_onefuncs([n1, n2]);
end

function neigh = neighbors_Ng_onefunc(f, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode)
    Z0 = generate_neighbor_set(neighborMode);
    C0 = center_of_basis(f, cscale, centerForNeighbors);
    cellF = {};
    k = 0;

    switch string(f.type)
        case "phi"
            step = 1 / cscale;
            for n = 1:size(Z0,1)
                target = C0 + step * Z0(n,:);
                fs = functions_at_center(target, sigma0, cscale, D, ZsetMode, ...
                    centerConvention, centerForNeighbors, 'phi', 0);
                for j = 1:numel(fs)
                    k = k + 1; cellF{k} = fs(j); %#ok<AGROW>
                end
            end

        case "psi"
            l = f.level;
            step = 1 / (cscale * 2^(l+1));
            for n = 1:size(Z0,1)
                target = C0 + step * Z0(n,:);
                fs = functions_at_center(target, sigma0, cscale, D, ZsetMode, ...
                    centerConvention, centerForNeighbors, 'psi', l);
                for j = 1:numel(fs)
                    k = k + 1; cellF{k} = fs(j); %#ok<AGROW>
                end
            end

        otherwise
            cellF = {};
    end

    if isempty(cellF)
        neigh = f([]);
    else
        neigh = unique_onefuncs([cellF{:}]);
    end
end

function neigh = neighbors_Ngplus_onefunc(f, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode)
    Z0 = generate_neighbor_set(neighborMode);
    C0 = center_of_basis(f, cscale, centerForNeighbors);
    cellF = {};
    k = 0;

    switch string(f.type)
        case "phi"
            step = 1 / (cscale * 2);
            level = 0;
            for n = 1:size(Z0,1)
                target = C0 + step * Z0(n,:);
                fs = functions_at_center(target, sigma0, cscale, D, ZsetMode, ...
                    centerConvention, centerForNeighbors, 'psi', level);
                for j = 1:numel(fs)
                    k = k + 1; cellF{k} = fs(j); %#ok<AGROW>
                end
            end

        case "psi"
            level = f.level + 1;
            step = 1 / (cscale * 2^(f.level+2));
            for n = 1:size(Z0,1)
                target = C0 + step * Z0(n,:);
                fs = functions_at_center(target, sigma0, cscale, D, ZsetMode, ...
                    centerConvention, centerForNeighbors, 'psi', level);
                for j = 1:numel(fs)
                    k = k + 1; cellF{k} = fs(j); %#ok<AGROW>
                end
            end

        otherwise
            cellF = {};
    end

    if isempty(cellF)
        neigh = f([]);
    else
        neigh = unique_onefuncs([cellF{:}]);
    end
end

function Z0 = generate_neighbor_set(mode)
    switch lower(mode)
        case 'cross'
            Z0 = [
                0 0 0
                1 0 0
               -1 0 0
                0 1 0
                0 -1 0
                0 0 1
                0 0 -1
            ];
        case 'full'
            vals = -1:1;
            [A,B,C] = ndgrid(vals, vals, vals);
            Z0 = [A(:), B(:), C(:)];
        otherwise
            error('Unknown neighborMode.');
    end
end

function fs = functions_at_center(target, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, kind, level)
    tol = 1e-10;
    fsCell = {};
    idx = 0;

    switch kind
        case 'phi'
            j_real = cscale * target;
            j = round(j_real);
            if norm(j - j_real) < tol
                f = make_phi(sigma0, cscale, 0, j, D);
                if norm(center_of_basis(f, cscale, centerForNeighbors) - target) < 1e-8
                    idx = idx + 1; fsCell{idx} = f; 
                end
            end

        case 'psi'
            Zset = generate_z_set(D, ZsetMode);
            scale = cscale * 2^level;
            for iz = 1:size(Zset,1)
                z = Zset(iz,:);
                switch lower(centerForNeighbors)
                    case 'actual'
                        j_real = scale * target - 0.5*z;
                    case 'paper_minus'
                        j_real = scale * target + 0.5*z;
                    otherwise
                        error('Unknown centerForNeighbors.');
                end
                j = round(j_real);
                if norm(j - j_real) < tol
                    f = make_psi(sigma0, cscale, level, j, z, D, centerConvention);
                    if norm(center_of_basis(f, cscale, centerForNeighbors) - target) < 1e-8
                        idx = idx + 1; fsCell{idx} = f; %#ok<AGROW>
                    end
                end
            end
    end

    if isempty(fsCell)
        fs = [];
    else
        fs = unique_onefuncs([fsCell{:}]);
    end
end

function Zset = generate_z_set(D, mode)
    switch lower(mode)
        case 'signed'
            vals = -1:1;
            [Z1,Z2,Z3] = ndgrid(vals, vals, vals);
            allz = [Z1(:), Z2(:), Z3(:)];
            Zset = allz(any(allz ~= 0, 2), :);
        case 'binary'
            allz = dec2bin(0:(2^D-1)) - '0';
            Zset = allz(any(allz ~= 0, 2), :);
        otherwise
            error('Unknown ZsetMode.');
    end
end

function f = make_phi(sigma0, cscale, l, j, D)
    scale = cscale * 2^l;
    width = sigma0 / scale;
    alpha = 1 / width^2;
    center = j / scale;
    term = struct('coef',1.0,'alpha',alpha,'center',center);
    f = struct('kind','frame','type','phi','level',l,'j',round(j), ...
        'z',zeros(1,D),'terms',term,'key',canonical_key_from_center('phi',l,center,cscale));
end

function f = make_psi(sigma0, cscale, l, j, z, D, centerConvention)
    gamma = 2^(-D/2);
    Cpsi = (1 - (16/25)*gamma*sqrt(5) + gamma^2)^(-1/2);
    scale = cscale * 2^l;
    switch lower(centerConvention)
        case 'eq428'
            center = (j + 0.5*z) / scale;
        case 'cmap'
            center = (j - 0.5*z) / scale;
        otherwise
            error('Unknown centerConvention.');
    end
    width1 = (sigma0/2) / scale;
    width2 = sigma0 / scale;
    term1 = struct('coef',Cpsi,'alpha',1/width1^2,'center',center);
    term2 = struct('coef',-Cpsi*gamma,'alpha',1/width2^2,'center',center);
    f = struct('kind','frame','type','psi','level',l,'j',round(j), ...
        'z',round(z),'terms',[term1,term2], ...
        'key',canonical_key_from_center('psi',l,center,cscale));
end

function key = canonical_key_from_center(type, level, center, cscale)
    scale = cscale * 2^level;
    idx2 = round(2 * scale * center);
    key = string(sprintf('%s_l%d_C2_%d_%d_%d', type, level, idx2(1), idx2(2), idx2(3)));
end

function C = center_of_basis(f, cscale, mode)
    switch lower(mode)
        case 'actual'
            C = f.terms(1).center;
        case 'paper_minus'
            if string(f.type) == "phi"
                C = f.j / cscale;
            else
                C = (f.j - 0.5*f.z) / (cscale * 2^f.level);
            end
        otherwise
            error('Unknown center mode.');
    end
end

function B = unique_onefuncs(B)
    if isempty(B), return; end
    keys = onefunc_keys(B);
    [~,ia] = unique(keys,'stable');
    B = B(ia);
end

%% ========================================================================
% Local functions: matrix caches and integrals
% ========================================================================

function [S, T, Ven] = contracted_one_particle(f, g, Z, Rnuc)
    S = 0; T = 0; Ven = 0;
    for a = 1:numel(f.terms)
        ta = f.terms(a);
        for b = 1:numel(g.terms)
            tb = g.terms(b);
            [Sab,Tab,Vab] = one_particle_primitive( ...
                ta.alpha, ta.center, tb.alpha, tb.center, Z, Rnuc);
            coef = ta.coef * tb.coef;
            S = S + coef*Sab;
            T = T + coef*Tab;
            Ven = Ven + coef*Vab;
        end
    end
end

% function val = contracted_eri(fa, fb, ga, gb)
%     % Kept for validation/debug only. The adaptive assembly uses eri_row_vectorized.
%     val = 0.0;
%     for a = 1:numel(fa.terms)
%         ta = fa.terms(a);
%         for b = 1:numel(fb.terms)
%             tb = fb.terms(b);
%             coefAB = ta.coef * tb.coef;
%             for c = 1:numel(ga.terms)
%                 tc = ga.terms(c);
%                 for d = 1:numel(gb.terms)
%                     td = gb.terms(d);
%                     coef = coefAB * tc.coef * td.coef;
%                     val = val + coef * eri_primitive( ...
%                         ta.alpha, ta.center, tb.alpha, tb.center, ...
%                         tc.alpha, tc.center, td.alpha, td.center);
%                 end
%             end
%         end
%     end
% end

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

% function val = eri_primitive(alphaA, A, alphaB, B, alphaC, C, alphaD, D)
%     p = alphaA + alphaB;
%     q = alphaC + alphaD;
%     P = (alphaA*A + alphaB*B) ./ p;
%     Qc = (alphaC*C + alphaD*D) ./ q;
%     RAB2 = sum((A-B).^2);
%     RCD2 = sum((C-D).^2);
%     RPQ2 = sum((P-Qc).^2);
%     NA = (alphaA/pi)^(3/4);
%     NB = (alphaB/pi)^(3/4);
%     NC = (alphaC/pi)^(3/4);
%     ND = (alphaD/pi)^(3/4);
%     Kab = NA * NB * exp(-(alphaA*alphaB/(2*p)) * RAB2);
%     Kcd = NC * ND * exp(-(alphaC*alphaD/(2*q)) * RCD2);
%     arg = (p*q/(2*(p+q))) * RPQ2;
%     val = Kab * Kcd * (8*sqrt(2)*pi^(5/2)) / ...
%         (p*q*sqrt(p+q)) * boys0(arg);
% end


function val = eri_primitive_vec(alphaA, Ax, Ay, Az, alphaB, Bx, By, Bz, alphaC, Cx, Cy, Cz, alphaD, Dx, Dy, Dz)
    % Vectorized version of eri_primitive for L2-normalized 3D Gaussians
    % G_alpha(x-A) = (alpha/pi)^(3/4) exp[-alpha |x-A|^2 / 2].
    % alphaA/A and alphaC/C may be scalars, while alphaB/B and alphaD/D
    % are row vectors. This function also works for gpuArray inputs.

    p = alphaA + alphaB;
    q = alphaC + alphaD;

    Px = (alphaA .* Ax + alphaB .* Bx) ./ p;
    Py = (alphaA .* Ay + alphaB .* By) ./ p;
    Pz = (alphaA .* Az + alphaB .* Bz) ./ p;

    Qx = (alphaC .* Cx + alphaD .* Dx) ./ q;
    Qy = (alphaC .* Cy + alphaD .* Dy) ./ q;
    Qz = (alphaC .* Cz + alphaD .* Dz) ./ q;

    RAB2 = (Ax - Bx).^2 + (Ay - By).^2 + (Az - Bz).^2;
    RCD2 = (Cx - Dx).^2 + (Cy - Dy).^2 + (Cz - Dz).^2;
    RPQ2 = (Px - Qx).^2 + (Py - Qy).^2 + (Pz - Qz).^2;

    NA = (alphaA ./ pi).^(3/4);
    NB = (alphaB ./ pi).^(3/4);
    NC = (alphaC ./ pi).^(3/4);
    ND = (alphaD ./ pi).^(3/4);

    Kab = NA .* NB .* exp(-(alphaA .* alphaB ./ (2 .* p)) .* RAB2);
    Kcd = NC .* ND .* exp(-(alphaC .* alphaD ./ (2 .* q)) .* RCD2);

    arg = (p .* q ./ (2 .* (p + q))) .* RPQ2;

    val = Kab .* Kcd .* (8 .* sqrt(2) .* pi^(5/2)) ./ ...
        (p .* q .* sqrt(p + q)) .* boys0_vec_any(arg);
end

function F = boys0_vec_any(t)
    % Vectorized Boys F0(t), compatible with double arrays and gpuArray.
    F = 0 .* t;
    small = t < 1e-10;

    if isa(t, 'gpuArray')
        hasSmall = gather(any(small(:)));
    else
        hasSmall = any(small(:));
    end

    if hasSmall
        ts = t(small);
        F(small) = 1 - ts./3 + ts.^2./10 - ts.^3./42 + ts.^4./216 - ts.^5./1320;
    end

    large = ~small;
    if isa(t, 'gpuArray')
        hasLarge = gather(any(large(:)));
    else
        hasLarge = any(large(:));
    end

    if hasLarge
        tl = t(large);
        F(large) = 0.5 .* sqrt(pi) ./ sqrt(tl) .* erf(sqrt(tl));
    end
end

function F = boys0(t)
    if t < 1e-10
        F = 1 - t/3 + t^2/10 - t^3/42 + t^4/216 - t^5/1320;
    else
        F = 0.5 * sqrt(pi) / sqrt(t) * erf(sqrt(t));
    end
end

%% ========================================================================
% Local functions: importance, solver, diagnostics
% ========================================================================

function scores = partial_orthogonal_importance(c, S)
    % J.H. Eq. (4.67)-(4.68) with m1=1.
    % B11 is scalar, B12 is 1 x (M-1).
    M = numel(c);
    if M == 1
        scores = abs(c);
        return;
    end

    B11 = S(1,1);
    B12 = S(1,2:end);

    z = c;
    z(1) = c(1) + (B12 / B11) * c(2:end);

    diagPO = zeros(M,1);
    diagPO(1) = B11;
    sdiag = diag(S);
    diagPO(2:end) = sdiag(2:end) - (B12(:).^2) / B11;
    diagPO = max(diagPO, realmin);

    w = sqrt(diagPO) .* z;
    scores = abs(w);
end

function out = compute_block_energy_contributions(c, H, twoBasis)
    M = numel(twoBasis);
    blocks = strings(M,1);
    for i = 1:M
        blocks(i) = string(twoBasis(i).block);
    end
    idxZero = find(blocks=="zero");
    idxOne = find(blocks=="one_up" | blocks=="one_down");
    idxTwo = find(blocks=="two_ud");

    out = struct();
    out.row.zero = row_energy(c,H,idxZero);
    out.row.one = row_energy(c,H,idxOne);
    out.row.two_ud = row_energy(c,H,idxTwo);
    out.row.total = out.row.zero + out.row.one + out.row.two_ud;
end

function e = row_energy(c,H,idx)
    if isempty(idx)
        e = 0;
    else
        e = real(c(idx)' * H(idx,:) * c);
    end
end

%% ========================================================================
% Sparse-stream replacement functions appended for v5.
% ========================================================================

function A = sparsify_symmetric(A, tol)
    if isempty(A)
        A = sparse([]);
        return;
    end
    if ~issparse(A)
        A(abs(A) < tol) = 0;
        A = sparse(A);
    else
        [i,j,v] = find(A);
        keep = abs(v) >= tol | (i == j);
        A = sparse(i(keep), j(keep), v(keep), size(A,1), size(A,2));
    end
    A = 0.5*(A + A.');
end

function [oneCache, twoCache, S2, H2, info, eriCache] = update_two_matrix_cache_sparse( ...
    oneCache, twoCache, oneFuncs, twoBasis, Zcharge, Rnuc, eriCache, assemblyOpts)

    oneCache = update_one_cache_sp(oneCache, oneFuncs, Zcharge, Rnuc, assemblyOpts);

    keys = twobasis_keys(twoBasis);
    M = numel(twoBasis);
    info = struct('reused',false,'oldM',0,'addedM',M);

    if isfield(assemblyOpts,'rebuildEveryAssembly') && assemblyOpts.rebuildEveryAssembly
        twoCache = reset_two_cache();
    end

    if ~twoCache.valid
        oldM = 0;
        Snew = sparse(M,M);
        Hnew = sparse(M,M);
    else
        oldM = twoCache.M;
        oldKeys = twoCache.keys;
        prefixOK = (M >= oldM) && all(keys(1:oldM) == oldKeys);
        if ~prefixOK
            warning('Basis prefix changed. Sparse cache is rebuilt from scratch.');
            oldM = 0;
            Snew = sparse(M,M);
            Hnew = sparse(M,M);
        else
            if M == oldM
                S2 = twoCache.S2;
                H2 = twoCache.H2;
                info.reused = true;
                info.oldM = oldM;
                info.addedM = 0;
                return;
            end
            Snew = twoCache.S2;
            Hnew = twoCache.H2;
            Snew(M,M) = 0; Hnew(M,M) = 0;
        end
    end

    if oldM == 0
        rowsAll = 1:M;
    else
        rowsAll = oldM+1:M;
    end

    rbSize = assemblyOpts.rowBlockSize;
    if isempty(rbSize) || rbSize <= 0
        rbSize = min(numel(rowsAll), 64);
    end

    t0 = tic;
    for r0 = 1:rbSize:numel(rowsAll)
        r1 = min(numel(rowsAll), r0 + rbSize - 1);
        rows = rowsAll(r0:r1);

        [Srows,Hrows,eriCache] = assemble_twobody_sparse_rows_sp( ...
            twoBasis, rows, 1:M, oneCache, eriCache, assemblyOpts);

        Snew(rows,:) = Srows;
        Hnew(rows,:) = Hrows;
        Snew(:,rows) = Srows.';
        Hnew(:,rows) = Hrows.';

        if assemblyOpts.showRowProgress
            fprintf('    sparse block rows %d-%d / %d inserted, nnz(S)=%.3e, nnz(H)=%.3e, elapsed %.1f s\n', ...
                rows(1), rows(end), M, nnz(Snew), nnz(Hnew), toc(t0));
        end
        clear Srows Hrows;
    end

    if assemblyOpts.symmetrizeAfterInsert
        Snew = 0.5*(Snew + Snew.');
        Hnew = 0.5*(Hnew + Hnew.');
        Snew = sparsify_symmetric(Snew, assemblyOpts.dropTolS);
        Hnew = sparsify_symmetric(Hnew, assemblyOpts.dropTolH);
    end

    S2 = Snew;
    H2 = Hnew;

    if isfield(assemblyOpts,'keepMatrixCache') && ~assemblyOpts.keepMatrixCache
        twoCache = reset_two_cache();
        info.reused = false;
        info.oldM = 0;
        info.addedM = M;
    else
        twoCache.valid = true;
        twoCache.keys = keys;
        twoCache.S2 = Snew;
        twoCache.H2 = Hnew;
        twoCache.M = M;
        twoCache.isSparseStream = true;
        info.reused = oldM > 0;
        info.oldM = oldM;
        info.addedM = M - oldM;
    end
end

function twoCache = reset_two_cache()
    twoCache = struct();
    twoCache.valid = false;
    twoCache.keys = strings(0,1);
    twoCache.S2 = sparse([]);
    twoCache.H2 = sparse([]);
    twoCache.M = 0;
    twoCache.isSparseStream = true;
end

function oneCache = update_one_cache_sp(oneCache, oneFuncs, Zcharge, Rnuc, assemblyOpts)
    M = numel(oneFuncs);
    oldM = oneCache.M;

    if M == oldM && isfield(oneCache,'termData')
        return;
    end

    keys = onefunc_keys(oneFuncs);
    prefixOK = M >= oldM && all(keys(1:oldM) == oneCache.keys);

    if ~prefixOK
        oldM = 0;
        oneCache.S1 = [];
        oneCache.H1 = [];
    end

    Snew = zeros(M,M);
    Hnew = zeros(M,M);
    if oldM > 0
        Snew(1:oldM,1:oldM) = oneCache.S1;
        Hnew(1:oldM,1:oldM) = oneCache.H1;
    end

    for i = oldM+1:M
        for j = 1:M
            [Sij, Tij, Vij] = contracted_one_particle(oneFuncs(i), oneFuncs(j), Zcharge, Rnuc);
            Hij = Tij + Vij;
            Snew(i,j) = Sij; Hnew(i,j) = Hij;
            Snew(j,i) = Sij; Hnew(j,i) = Hij;
        end
    end

    oneCache.S1 = (Snew+Snew')/2;
    oneCache.H1 = (Hnew+Hnew')/2;
    oneCache.keys = keys;
    oneCache.M = M;
    oneCache.termData = build_term_data_sp(oneFuncs);

    if strcmpi(assemblyOpts.mode,'gpu_serial_sparse')
        try
            oneCache.gpuTermData = gpu_term_data_sp(oneCache.termData, assemblyOpts.useSingleGPUWorking);
        catch ME
            warning('GPU termData setup failed: %s. CPU assembly will be used.', ME);
            oneCache.gpuTermData = [];
        end
    end
end

function data = build_term_data_sp(oneFuncs)
    M = numel(oneFuncs);
    maxTerms = 0;
    for i = 1:M
        maxTerms = max(maxTerms, numel(oneFuncs(i).terms));
    end
    coef = zeros(M,maxTerms);
    alpha = zeros(M,maxTerms);
    cx = zeros(M,maxTerms); cy = zeros(M,maxTerms); cz = zeros(M,maxTerms);
    for i = 1:M
        for t = 1:numel(oneFuncs(i).terms)
            term = oneFuncs(i).terms(t);
            coef(i,t) = term.coef;
            alpha(i,t) = term.alpha;
            cx(i,t) = term.center(1);
            cy(i,t) = term.center(2);
            cz(i,t) = term.center(3);
        end
    end
    data = struct('M',M,'maxTerms',maxTerms,'coef',coef,'alpha',alpha, ...
        'cx',cx,'cy',cy,'cz',cz);
end

function gd = gpu_term_data_sp(td, useSingle)
    if useSingle
        gd.coef = gpuArray(single(td.coef));
        gd.alpha = gpuArray(single(td.alpha));
        gd.cx = gpuArray(single(td.cx)); gd.cy = gpuArray(single(td.cy)); gd.cz = gpuArray(single(td.cz));
    else
        gd.coef = gpuArray(td.coef);
        gd.alpha = gpuArray(td.alpha);
        gd.cx = gpuArray(td.cx); gd.cy = gpuArray(td.cy); gd.cz = gpuArray(td.cz);
    end
    gd.M = td.M; gd.maxTerms = td.maxTerms;
end

function [Srows,Hrows,eriCache] = assemble_twobody_sparse_rows_sp( ...
    twoBasis, rowIdx, colIdx, oneCache, eriCache, assemblyOpts)

    nr = numel(rowIdx); nc = numel(colIdx);
    S1 = oneCache.S1; H1 = oneCache.H1;
    td = oneCache.termData;

    leftCols  = [twoBasis(colIdx).left];
    rightCols = [twoBasis(colIdx).right];
    tb = twoBasis;

    mode = lower(string(assemblyOpts.mode));
    usePar = (mode == "cpu_parfor_sparse");
    useGPU = (mode == "gpu_serial_sparse");

    if usePar && assemblyOpts.startPool && isempty(gcp('nocreate'))
        try
            parpool;
        catch ME
            warning('Could not start parpool: %s. Falling back to cpu_serial_sparse.', ME);
            usePar = false;
        end
    end

    IScell = cell(nr,1); JScell = cell(nr,1); VScell = cell(nr,1);
    IHcell = cell(nr,1); JHcell = cell(nr,1); VHcell = cell(nr,1);

    if usePar
        parfor ir = 1:nr
            [js,vs,jh,vh] = assemble_one_sparse_row_cpu_sp(tb, rowIdx(ir), leftCols, rightCols, S1, H1, td, assemblyOpts);
            IScell{ir} = repmat(ir, numel(js), 1); JScell{ir} = js(:); VScell{ir} = vs(:);
            IHcell{ir} = repmat(ir, numel(jh), 1); JHcell{ir} = jh(:); VHcell{ir} = vh(:);
        end
    else
        for ir = 1:nr
            if useGPU && isfield(oneCache,'gpuTermData') && ~isempty(oneCache.gpuTermData)
                [js,vs,jh,vh] = assemble_one_sparse_row_gpu_sp(tb, rowIdx(ir), leftCols, rightCols, S1, H1, oneCache.gpuTermData, assemblyOpts);
            else
                [js,vs,jh,vh] = assemble_one_sparse_row_cpu_sp(tb, rowIdx(ir), leftCols, rightCols, S1, H1, td, assemblyOpts);
            end
            IScell{ir} = repmat(ir, numel(js), 1); JScell{ir} = js(:); VScell{ir} = vs(:);
            IHcell{ir} = repmat(ir, numel(jh), 1); JHcell{ir} = jh(:); VHcell{ir} = vh(:);
        end
    end

    if nr == 0
        Srows = sparse(nr,nc); Hrows = sparse(nr,nc); return;
    end

    IS = vertcat(IScell{:}); JS = vertcat(JScell{:}); VS = vertcat(VScell{:});
    IH = vertcat(IHcell{:}); JH = vertcat(JHcell{:}); VH = vertcat(VHcell{:});
    Srows = sparse(IS, JS, VS, nr, nc);
    Hrows = sparse(IH, JH, VH, nr, nc);
end

function [js,vs,jh,vh] = assemble_one_sparse_row_cpu_sp(tb, a, leftCols, rightCols, S1, H1, td, assemblyOpts)
    la = tb(a).left; ra = tb(a).right;
    Sl = S1(la, leftCols); Sr = S1(ra, rightCols);
    Hl = H1(la, leftCols); Hr = H1(ra, rightCols);
    srow = Sl .* Sr;
    hrow = Hl .* Sr + Sl .* Hr + eri_row_vectorized_cpu_sp(td, la, ra, leftCols, rightCols);
    maskS = abs(srow) >= assemblyOpts.dropTolS;
    maskH = abs(hrow) >= assemblyOpts.dropTolH;
    js = find(maskS); vs = srow(maskS);
    jh = find(maskH); vh = hrow(maskH);
end

function [js,vs,jh,vh] = assemble_one_sparse_row_gpu_sp(tb, a, leftCols, rightCols, S1, H1, gtd, assemblyOpts)
    la = tb(a).left; ra = tb(a).right;
    Sl = S1(la, leftCols); Sr = S1(ra, rightCols);
    Hl = H1(la, leftCols); Hr = H1(ra, rightCols);
    srow = Sl .* Sr;
    vgpu = eri_row_vectorized_gpu_sp(gtd, la, ra, leftCols, rightCols);
    hrow = Hl .* Sr + Sl .* Hr + double(gather(vgpu));
    maskS = abs(srow) >= assemblyOpts.dropTolS;
    maskH = abs(hrow) >= assemblyOpts.dropTolH;
    js = find(maskS); vs = srow(maskS);
    jh = find(maskH); vh = hrow(maskH);
end

function vrow = eri_row_vectorized_cpu_sp(td, la, ra, leftCols, rightCols)
    nc = numel(leftCols);
    vrow = zeros(1,nc);
    T = td.maxTerms;
    for ia = 1:T
        ca = td.coef(la,ia); if ca == 0, continue; end
        aA = td.alpha(la,ia); Ax = td.cx(la,ia); Ay = td.cy(la,ia); Az = td.cz(la,ia);
        for ic = 1:T
            cc = td.coef(ra,ic); if cc == 0, continue; end
            aC = td.alpha(ra,ic); Cx = td.cx(ra,ic); Cy = td.cy(ra,ic); Cz = td.cz(ra,ic);
            coefAC = ca * cc;
            for ib = 1:T
                cb = td.coef(leftCols,ib).';
                activeB = cb ~= 0;
                if ~any(activeB), continue; end
                aB_all = td.alpha(leftCols,ib).';
                Bx_all = td.cx(leftCols,ib).'; By_all = td.cy(leftCols,ib).'; Bz_all = td.cz(leftCols,ib).';
                for id = 1:T
                    cd = td.coef(rightCols,id).';
                    mask = activeB & (cd ~= 0);
                    if ~any(mask), continue; end
                    aB = aB_all(mask); Bx = Bx_all(mask); By = By_all(mask); Bz = Bz_all(mask);
                    aD = td.alpha(rightCols(mask),id).';
                    Dx = td.cx(rightCols(mask),id).'; Dy = td.cy(rightCols(mask),id).'; Dz = td.cz(rightCols(mask),id).';
                    val = eri_primitive_vec(aA,Ax,Ay,Az, aB,Bx,By,Bz, aC,Cx,Cy,Cz, aD,Dx,Dy,Dz);
                    vrow(mask) = vrow(mask) + coefAC .* cb(mask) .* cd(mask) .* val;
                end
            end
        end
    end
end

function vrow = eri_row_vectorized_gpu_sp(td, la, ra, leftCols, rightCols)
    % GPU row-wise vectorized ERI assembly.
    % leftCols/rightCols are CPU integer vectors. The indexed slices of td.*
    % are gpuArray because td.* are gpuArray. This avoids using gpuArray
    % indices/logical masks to index CPU vectors.
    nc = double(numel(leftCols));

    % Use zeros(...,'like',gpuArray) rather than gpuArray.zeros(...,'like',...),
    % because gpuArray.zeros does not support the 'like' syntax in several
    % MATLAB releases.
    vrow = zeros(1, nc, 'like', td.alpha);

    T = td.maxTerms;
    for ia = 1:T
        ca = td.coef(la,ia); if gather(ca == 0), continue; end
        aA = td.alpha(la,ia); Ax = td.cx(la,ia); Ay = td.cy(la,ia); Az = td.cz(la,ia);
        for ic = 1:T
            cc = td.coef(ra,ic); if gather(cc == 0), continue; end
            aC = td.alpha(ra,ic); Cx = td.cx(ra,ic); Cy = td.cy(ra,ic); Cz = td.cz(ra,ic);
            coefAC = ca * cc;
            for ib = 1:T
                cb = td.coef(leftCols,ib).';
                activeB = cb ~= 0;
                if ~gather(any(activeB)), continue; end

                aB_all = td.alpha(leftCols,ib).';
                Bx_all = td.cx(leftCols,ib).';
                By_all = td.cy(leftCols,ib).';
                Bz_all = td.cz(leftCols,ib).';

                for id = 1:T
                    cd = td.coef(rightCols,id).';
                    mask = activeB & (cd ~= 0);
                    if ~gather(any(mask)), continue; end

                    aB = aB_all(mask); Bx = Bx_all(mask); By = By_all(mask); Bz = Bz_all(mask);

                    % Pre-extract D-side arrays with CPU column indices, then GPU-mask them.
                    aD_all = td.alpha(rightCols,id).';
                    Dx_all = td.cx(rightCols,id).';
                    Dy_all = td.cy(rightCols,id).';
                    Dz_all = td.cz(rightCols,id).';
                    aD = aD_all(mask); Dx = Dx_all(mask); Dy = Dy_all(mask); Dz = Dz_all(mask);

                    val = eri_primitive_vec(aA,Ax,Ay,Az, aB,Bx,By,Bz, aC,Cx,Cy,Cz, aD,Dx,Dy,Dz);

                    % find(mask) on GPU may return a gpuArray. Gather for indexing safety.
                    idx = gather(find(mask));
                    vrow(idx) = vrow(idx) + coefAC .* cb(mask) .* cd(mask) .* val;
                end
            end
        end
    end
end

function [E0, coeff, nkeep, info] = solve_ground_generalized_sparseaware(H, S, mass_tol, verboseEig, targetEnergy, solverOpts)
    % v11 final eigensolver: NO generalized sparse eigs(H,S,...).
    %
    % The previous sparse generalized branch fails when S is singular or badly
    % scaled, and a wrong eigenvector directly destroys the adaptive M-curve.
    % This function restores the stable projected route used in the trusted
    % dense versions:
    %   1) diagonalize/project the mass matrix S and remove null directions;
    %   2) solve a standard Hermitian eigenproblem in the projected space;
    %   3) map the vector back and validate by Rayleigh quotient/residual.
    %
    % By default the projection is exact dense mass projection.  This is more
    % memory-demanding, but it avoids generalized sparse eigs completely and
    % should preserve the original M-growth curve.

    H = 0.5*(H+H');
    S = 0.5*(S+S');
    M = size(H,1);
    info = struct('method',"",'flag',NaN,'residual',NaN,'nkeep',NaN, ...
        'sigma',NaN,'rayleigh',NaN,'eigValue',NaN,'sNorm',NaN, ...
        'massMin',NaN,'massMax',NaN,'projectedMode',"");

    if M > solverOpts.projectedDenseLimit
        if isfield(solverOpts,'allowSparseMassProjectionFallback') && solverOpts.allowSparseMassProjectionFallback
            [E0, coeff, nkeep, info] = solve_by_sparse_mass_projection_noGE( ...
                H, S, mass_tol, verboseEig, targetEnergy, solverOpts);
            return;
        else
            error(['M=%d exceeds solverOpts.projectedDenseLimit=%d. ', ...
                   'Generalized sparse eigs is disabled in v11. ', ...
                   'Increase projectedDenseLimit only if RAM is sufficient, ', ...
                   'or enable allowSparseMassProjectionFallback for experimental sparse mass projection.'], ...
                   M, solverOpts.projectedDenseLimit);
        end
    end

    % Exact/stable projected branch.  Convert only inside the solver; the main
    % adaptive loop still stores sparse S,H and basis-only checkpoints.
    if verboseEig
        fprintf('    projected dense solver: M=%d, converting sparse S/H to full for mass projection...\n', M);
    end
    Hd = full(H);
    Sd = full(S);
    clear H S;

    tMass = tic;
    [U,D] = eig(Sd,'vector');
    lam = real(D(:));
    [lam,ord] = sort(lam,'descend');
    U = U(:,ord);
    lamMax = max(lam);
    keep = lam > mass_tol * lamMax;
    nkeep = nnz(keep);
    info.nkeep = nkeep;
    info.massMin = min(lam);
    info.massMax = lamMax;
    info.projectedMode = "full_mass_projection";

    if nkeep < 1
        error('Mass projection kept zero directions. Check mass_tol or S assembly.');
    end

    X = U(:,keep) .* (1 ./ sqrt(lam(keep))).';
    clear U D lam keep;

    if verboseEig
        fprintf('    mass eig/project: M=%d kept=%d min=%.3e max=%.3e time=%.2fs\n', ...
            M, nkeep, info.massMin, info.massMax, toc(tMass));
    end

    tProj = tic;
    HX = Hd * X;
    Horth = X' * HX;
    Horth = 0.5*(Horth+Horth');
    clear HX;
    if verboseEig
        fprintf('    projected Hamiltonian built: nkeep=%d time=%.2fs\n', nkeep, toc(tProj));
    end

    [E0, y0, eigInfo] = solve_standard_projected_H(Horth, targetEnergy, solverOpts, verboseEig);
    info.method = eigInfo.method;
    info.flag = eigInfo.flag;
    info.residual = eigInfo.residual;
    info.sigma = eigInfo.sigma;
    info.eigValue = E0;

    coeff = X * y0;
    clear X y0 Horth;
    sNorm = real(coeff' * Sd * coeff);
    if ~isfinite(sNorm) || sNorm <= solverOpts.minPositiveSNorm
        error('Projected solver returned invalid S-norm %.3e.', sNorm);
    end
    coeff = coeff / sqrt(sNorm);
    sNorm = real(coeff' * Sd * coeff);
    E_RQ = real(coeff' * Hd * coeff) / sNorm;
    r = Hd*coeff - E_RQ*(Sd*coeff);
    resFull = norm(r) / max(1, (norm(Hd,1)+abs(E_RQ)*norm(Sd,1))*norm(coeff,1));

    if abs(E_RQ - E0) > solverOpts.rayleighEigTol
        if verboseEig
            fprintf('    projected solver note: eig/RQ differ %.3e; using Rayleigh energy.\n', abs(E_RQ-E0));
        end
    end
    E0 = E_RQ;
    info.rayleigh = E_RQ;
    info.sNorm = sNorm;
    info.residual = max(info.residual, resFull);
    info.method = "projected-" + string(info.method);

    lowerBound = solverOpts.exactLowerBound - solverOpts.energyLowerTol;
    if solverOpts.rejectBelowExact && E0 < lowerBound
        error('Projected solver produced E=%.15f below exact guard %.15f. Stop to protect adaptive M-curve.', E0, lowerBound);
    end
end

function [E0, y0, info] = solve_standard_projected_H(Horth, targetEnergy, solverOpts, verboseEig)
    % Stable standard eigensolver in the S-orthonormalized subspace.
    nkeep = size(Horth,1);
    info = struct('method',"",'flag',NaN,'residual',NaN,'sigma',NaN);

    if nkeep <= solverOpts.directEigLimit
        tEig = tic;
        [Y,Eval] = eig(Horth);
        evals = real(diag(Eval));
        [evals,idx] = sort(evals,'ascend');
        y0 = real(Y(:,idx(1)));
        E0 = evals(1);
        info.method = "full-eig-direct";
        info.flag = 0;
        info.residual = norm(Horth*y0 - E0*y0)/max(1,norm(Horth,1)*norm(y0,1));
        if verboseEig
            fprintf('    standard eig direct: nkeep=%d E=%.15f res=%.3e time=%.2fs\n', ...
                nkeep, E0, info.residual, toc(tEig));
        end
        return;
    end

    solved = false;
    shiftGap = max(solverOpts.shiftAbsGap, solverOpts.shiftRelGap * max(1, abs(targetEnergy)));
    sigmaShift = targetEnergy - shiftGap;
    opts = struct('isreal',true,'tol',solverOpts.eigsTol, ...
        'maxit',solverOpts.eigsMaxit,'disp',0,'p',min(nkeep,solverOpts.eigsP));

    tEig = tic;
    try
        [y,e,flag] = eigs(Horth,1,sigmaShift,opts);
        y = real(y(:,1));
        E = real(e(1,1));
        res = norm(Horth*y - E*y)/max(1,norm(Horth,1)*norm(y,1));
        if isfinite(E) && isfinite(res) && (flag==0 || res < solverOpts.resTol)
            solved = true;
            y0 = y; E0 = E;
            info.method = sprintf("standard-shift-invert(%.8g)", sigmaShift);
            info.flag = flag;
            info.residual = res;
            info.sigma = sigmaShift;
            if verboseEig
                fprintf('    standard shift-invert: nkeep=%d E=%.15f res=%.3e flag=%d time=%.2fs\n', ...
                    nkeep, E0, res, flag, toc(tEig));
            end
        end
    catch ME
        if verboseEig
            fprintf('    standard shift-invert failed: %s\n', ME);
        end
    end

    if ~solved && nkeep <= solverOpts.fullFallbackLimit
        tEig = tic;
        [Y,Eval] = eig(Horth);
        evals = real(diag(Eval));
        [evals,idx] = sort(evals,'ascend');
        y0 = real(Y(:,idx(1)));
        E0 = evals(1);
        info.method = "full-eig-fallback";
        info.flag = 0;
        info.residual = norm(Horth*y0 - E0*y0)/max(1,norm(Horth,1)*norm(y0,1));
        solved = true;
        if verboseEig
            fprintf('    standard full fallback: nkeep=%d E=%.15f res=%.3e time=%.2fs\n', ...
                nkeep, E0, info.residual, toc(tEig));
        end
    end

    if ~solved
        % Last resort is a standard smallest-real solve on Horth only.  This is
        % not a generalized sparse eigensolve and is not affected by singular S.
        tEig = tic;
        [y,e,flag] = eigs(Horth,1,'smallestreal',opts);
        y0 = real(y(:,1));
        E0 = real(e(1,1));
        info.method = "standard-smallestreal";
        info.flag = flag;
        info.residual = norm(Horth*y0 - E0*y0)/max(1,norm(Horth,1)*norm(y0,1));
        if verboseEig
            fprintf('    standard smallestreal: nkeep=%d E=%.15f res=%.3e flag=%d time=%.2fs\n', ...
                nkeep, E0, info.residual, flag, toc(tEig));
        end
        if ~(isfinite(E0) && isfinite(info.residual))
            error('Standard projected eigensolver failed at nkeep=%d.', nkeep);
        end
    end
end

function [E0, coeff, nkeep, info] = solve_by_sparse_mass_projection_noGE(H, S, mass_tol, verboseEig, targetEnergy, solverOpts)
    % Optional experimental fallback: sparse mass projection, then standard H.
    % It still avoids generalized eigs(H,S,...), but uses eigs(S,...) to extract
    % the positive mass subspace.  Kept OFF by default because it can alter
    % small singular directions if k is not large enough.
    H = 0.5*(H+H');
    S = 0.5*(S+S');
    M = size(H,1);
    info = struct('method',"sparse-mass-projection",'flag',NaN,'residual',NaN,'nkeep',NaN, ...
        'sigma',NaN,'rayleigh',NaN,'eigValue',NaN,'sNorm',NaN, ...
        'massMin',NaN,'massMax',NaN,'projectedMode',"sparse_mass_projection");

    optsS = struct('isreal',true,'tol',solverOpts.massEigsTol, ...
        'maxit',solverOpts.massEigsMaxit,'disp',0,'p',solverOpts.massEigsP);
    [~,dmax] = eigs(S,1,'largestreal',optsS);
    lamMax = real(dmax(1,1));
    thresh = mass_tol * lamMax;
    k = min(M-2, solverOpts.massEigsInitialK);
    U = []; lam = [];
    while true
        if verboseEig
            fprintf('    sparse mass eigs: requesting k=%d, threshold=%.3e\n', k, thresh);
        end
        [Uk,Dk,flagS] = eigs(S,k,'largestreal',optsS); %#ok<ASGLU>
        lamk = real(diag(Dk));
        [lamk,ord] = sort(lamk,'descend'); Uk = real(Uk(:,ord));
        if lamk(end) <= thresh || k >= min(M-2,solverOpts.massEigsMaxK)
            U = Uk; lam = lamk;
            break;
        end
        kNew = min(M-2, min(solverOpts.massEigsMaxK, ceil(k*solverOpts.massEigsStepFactor)));
        if kNew <= k
            U = Uk; lam = lamk;
            break;
        end
        k = kNew;
    end
    keep = lam > thresh;
    U = U(:,keep); lam = lam(keep);
    nkeep = numel(lam);
    X = U .* (1 ./ sqrt(lam)).';
    clear U;
    HX = H * X;
    Horth = X' * HX; Horth = 0.5*(Horth+Horth');
    clear HX;
    [~, y0, eigInfo] = solve_standard_projected_H(Horth, targetEnergy, solverOpts, verboseEig);
    coeff = X * y0;
    sNorm = real(coeff' * S * coeff);
    coeff = coeff / sqrt(sNorm);
    sNorm = real(coeff' * S * coeff);
    E_RQ = real(coeff' * H * coeff) / sNorm;
    r = H*coeff - E_RQ*(S*coeff);
    res = norm(r) / max(1, (norm(H,1)+abs(E_RQ)*norm(S,1))*norm(coeff,1));

    E0 = E_RQ;
    info.method = "sparseMass-" + string(eigInfo.method);
    info.flag = eigInfo.flag;
    info.residual = max(eigInfo.residual,res);
    info.nkeep = nkeep;
    info.rayleigh = E_RQ;
    info.eigValue = E0;
    info.sNorm = sNorm;
    info.massMin = min(lam);
    info.massMax = lamMax;

    lowerBound = solverOpts.exactLowerBound - solverOpts.energyLowerTol;
    if solverOpts.rejectBelowExact && E0 < lowerBound
        error('Sparse mass projected solver produced E=%.15f below exact guard %.15f.', E0, lowerBound);
    end
end
