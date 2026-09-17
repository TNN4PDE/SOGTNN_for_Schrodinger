%% JH_HeTriplet_part2_build_coarsen_debug_v1.m
% J.H.-style adaptive reproduction for the lowest triplet state of He, (3S)He.
%
% Input:
%   JH_HeTriplet_HF_Q9_orbitals.mat  (run JH_HeTriplet_part1_HF_Q9.m first)
%
% Main settings from J.H. Section 5.2:
%   N=2, M_S=1, two spin-up electrons, fully antisymmetric spatial determinant.
%   c = 1/4 and L = 7 for the initial one-particle multiscale frame.
%   epsilon_kappa = 2^(-kappa), coarsening delta = 1/100 kcal/mol.
%
% Difference from singlet He M_S=0 code:
%   Singlet/opposite spin basis used products phi_i(r1) phi_j(r2).
%   Triplet/same spin basis uses normalized Slater determinants det[f_i,f_j].
%   Matrix elements are assembled using Löwdin determinant formulas:
%       S = S_ik S_jl - S_il S_jk
%       H1 = h_ik S_jl + S_ik h_jl - h_il S_jk - S_il h_jk
%       Vee = (ik|jl) - (il|jk)
%
% Output:
%   JH_HeTriplet_raw_assembled_M*.mat
%   JH_HeTriplet_coarsened_M*_nu*.mat
%   JH_HeTriplet_coarsened_<tag>_M*_nu*.mat
%
% This file intentionally stops after raw assembly and coarsening.

clear; clc; close all;
format long e;

%% ============================================================
% 0. Parameters
% =============================================================
hfFile = 'JH_HeTriplet_HF_Q9_orbitals.mat';
if ~exist(hfFile,'file')
    error('Cannot find %s. Run JH_HeTriplet_part1_HF_Q9.m first.', hfFile);
end
load(hfFile, 'hf');

Zcharge = hf.Z;
Rnuc = [0 0 0];
D = 3;
sigma0 = 1;
cscale = 1/4;
L0 = 7;
ZsetMode = 'signed';
centerConvention = 'eq428';
centerForNeighbors = 'actual';
neighborMode = 'cross';
updateMode = 'monotone_union';
importanceMode = 'partial_orthogonal';

% New audit/fix knob: in the particle-wise splitting the frame should represent
% the detail space W in V = U \oplus W, where U = span{psi_1,psi_2}.
% 'none' reproduces the previous unprojected calculation.
% 'orthogonal_to_U' replaces every non-reference one-particle frame function phi by
%     phi_W = phi - P_U phi,  P_U uses the L2 Gram matrix of the two HF orbitals.
% This is the most faithful interpretation of B_u^{(N,MS)} based on W, and is
% expected to strongly affect triplet coarsening/M-count.
detailProjectionMode = 'orthogonal_to_U';

% Critical debug knob for the antisymmetric 2x2 determinant basis.
% 'antisymmetrizer' means we use A[f_i,f_j] with one global antisymmetrizer scale,
% not individual L2 normalization. This keeps the same Galerkin subspace/energy,
% but changes coefficient magnitudes used by J.H.'s B(nu) coarsening.
% 'l2_normalized' reproduces the earlier normalized determinant implementation.
detScaleMode = 'l2_normalized';

% Critical for same-spin initial B_{p1,p2}^{[0]} in Eq. (4.55):
% both phi and phi_tilde are taken from the finite initial system b^{(L)}_{sigma,c}.
% If this is false, Ng(phi) creates outside-frame neighbors already in B^[0],
% which over-enriches the raw/coarsened system and gives M=314 instead of Table 5.5 M=141.
restrictInitialTwoNeighborsToFrame = false;   % J.H. Eq.(4.55): allow phi_tilde in Ng(phi), even outside finite b^(L)

% More accurate nonrelativistic reference for He 1s2s 3S is about this value.
% J.H. Table 5.8 rounds it to -2.17523.
E_ref_exact = -2.175229378236791;
E_ref_table = -2.17523;

% Controls.
kappa_max = 5;          % First-stage debug target: Table 5.5 rows kappa=1 and kappa=5.
max_inner = 40;
max_basis_allowed = 50000;
mass_tol = 1e-10;
verboseEig = false;

assemblyOpts = struct();
assemblyOpts.mode = 'cpu_parfor_sparse';  % 'serial_sparse' or 'cpu_parfor_sparse'
assemblyOpts.rowBlockSize = 128;        % stable for kappa<=5; increase to 256 later if desired
assemblyOpts.startPool = true;
assemblyOpts.dropTolS = 0;                % keep exact sparsity pattern from analytic zeros only
assemblyOpts.dropTolH = 0;
assemblyOpts.showRowProgress = true;
assemblyOpts.upperTriangle = true;      % same-spin determinant matrices are symmetric; compute i<=j only
assemblyOpts.detScaleMode = detScaleMode;

recordOpts = struct();
recordOpts.saveRecordsEveryInner = true;
recordOpts.saveBasisEveryMRecord = true;
recordOpts.saveEveryInnerPrefix = 'JH_HeTriplet_basis_byM';
recordOpts.computeBlockEnergyEachInner = true;

fprintf('\n============================================================\n');
fprintf('J.H. He triplet Table 5.5 adaptive code, fully antisymmetric same-spin determinants\n');
fprintf('HF file: %s\n', hfFile);
fprintf('Parameters: c=%.6g, L=%d, sigma=%.6g, neighbor=%s, update=%s\n', ...
    cscale, L0, sigma0, neighborMode, updateMode);
fprintf('Initial same-spin B55: phi_tilde in Ng(phi), restrict to initial frame = %d\n', restrictInitialTwoNeighborsToFrame);
fprintf('Determinant scaling mode: %s\n', detScaleMode);
fprintf('Detail projection mode: %s\n', detailProjectionMode);
fprintf('Target Table 5.5: kappa=1 M=141 E=-2.174698; kappa=12 M=20386 E=-2.175220.\n');
fprintf('First-stage run: kappa_max=5. Expected final kappa=5: M=147, E=-2.174698, Mone=82, Mtwo_uu=64.\n');
fprintf('============================================================\n\n');

%% ============================================================
% 1. Build one-particle pool and raw B_three^[0]
% =============================================================
oneFuncs = repmat(empty_onefunc(), 0, 1);
oneKeyMap = containers.Map('KeyType','char','ValueType','double');

% Register the two HF reference orbitals as one-particle functions 1 and 2.
for p = 1:2
    f = empty_onefunc();
    f.kind = 'ref';
    f.type = 'ref';
    f.level = -1;
    f.j = [0 0 0];
    f.z = [0 0 0];
    f.terms = hf.orbitals(p).terms;
    f.key = string(sprintf('ref%d',p));
    [oneFuncs, oneKeyMap, ~] = register_onefunc(oneFuncs, oneKeyMap, f);
end

% Projector onto the HF one-particle reference subspace U = span{psi_1,psi_2}.
% This is used to convert the multiscale frame into detail functions in W.
projU = build_reference_projector(oneFuncs(1), oneFuncs(2), Zcharge, Rnuc);
fprintf('Reference Gram S_U = [[%.6e %.6e]; [%.6e %.6e]], cond=%.3e\n', ...
    projU.G(1,1), projU.G(1,2), projU.G(2,1), projU.G(2,2), cond(projU.G));

% Initial one-particle multiscale frame b_{sigma,c}^{(L)}.
frame = make_initial_frame(sigma0, cscale, L0, D, ZsetMode, centerConvention);
for i = 1:numel(frame)
    fdetail = prepare_detail_onefunc(frame(i), projU, detailProjectionMode);
    [oneFuncs, oneKeyMap, ~] = register_onefunc(oneFuncs, oneKeyMap, fdetail);
end
firstFrameID = 3;
lastFrameID = numel(oneFuncs);
Mb = lastFrameID - firstFrameID + 1;

fprintf('One-particle frame count Mb = %d. For signed Zset, expected 27+26*(L-1)=183.\n', Mb);

% Build raw determinant basis.
rawBasis = repmat(empty_twobasis(), 0, 1);
twoKeyMap = containers.Map('KeyType','char','ValueType','double');

[rawBasis, twoKeyMap] = append_basis(rawBasis, twoKeyMap, make_det_basis('zero',1,2,0));

% V_one: replace either HF orbital by each frame function.
for fid = firstFrameID:lastFrameID
    [rawBasis, twoKeyMap] = append_basis(rawBasis, twoKeyMap, make_det_basis('one', fid, 2, 1));
    [rawBasis, twoKeyMap] = append_basis(rawBasis, twoKeyMap, make_det_basis('one', 1, fid, 2));
end

% V_two^{up up}: Eq. (4.55) for the same-spin pair.
% Important correction relative to v2:
% In the initial a-priori system B_three^[0], both phi and phi_tilde are restricted
% to the finite initial frame b^{(L)}_{sigma,c}.  Ng(phi) is used only to choose
% local partner functions inside this finite system.  Otherwise one already adds
% outside-frame functions during coarsening, which over-enriches the initial space.
for fid = firstFrameID:lastFrameID
    neigh = neighbors_Ng_onefunc(oneFuncs(fid), sigma0, cscale, D, ZsetMode, ...
        centerConvention, centerForNeighbors, neighborMode);
    for n = 1:numel(neigh)
        nf = prepare_detail_onefunc(neigh(n), projU, detailProjectionMode);
        nkey = char(nf.key);
        if restrictInitialTwoNeighborsToFrame
            if ~isKey(oneKeyMap, nkey)
                continue;
            end
            nid = oneKeyMap(nkey);
            if nid < firstFrameID || nid > lastFrameID
                continue;
            end
        else
            [oneFuncs, oneKeyMap, nid] = register_onefunc(oneFuncs, oneKeyMap, nf);
        end
        if nid ~= fid
            [rawBasis, twoKeyMap] = append_basis(rawBasis, twoKeyMap, make_det_basis('two_uu', fid, nid, 0));
        end
    end
end

rawCounts = count_blocks(rawBasis);
fprintf('\nRaw B_three^[0] before coarsening:\n');
fprintf('  M=%d | Mzero=%d | Mone=%d | Mtwo_uu=%d | oneFuncs=%d\n', ...
    rawCounts.total, rawCounts.zero, rawCounts.one_total, rawCounts.two_uu, numel(oneFuncs));
if restrictInitialTwoNeighborsToFrame
    fprintf('  Expected strict finite-frame B55 raw: M=1+2*183+342=709 (before linear dependence/coarsening).\n');
end

%% ============================================================
% 2. Assemble raw matrices and coarsen
% =============================================================
oneCache = reset_one_cache(Zcharge, Rnuc);
oneCache = update_one_cache(oneCache, oneFuncs, Zcharge, Rnuc);

tAsmRaw = tic;
[Sraw,Hraw,rawInfo] = assemble_twobody_matrices(rawBasis, oneCache, oneFuncs, assemblyOpts);
tAsmRaw = toc(tAsmRaw);

fprintf('\nRaw assembly done: M=%d, nnz(S/H)=%.3e/%.3e, time=%.2fs\n', ...
    numel(rawBasis), nnz(Sraw), nnz(Hraw), tAsmRaw);

% Zero determinant should reproduce the non-orthogonal triplet HF reference.
EzeroDirect = full(Hraw(1,1) / Sraw(1,1));
fprintf('Zero determinant check: H(1,1)/S(1,1)=%.15f | hf.E=%.15f | diff=%.3e\n', ...
    EzeroDirect, hf.E, EzeroDirect-hf.E);

[Er, cr, keptRaw, rawSolveInfo] = solve_ground_projected(Hraw, Sraw, mass_tol, verboseEig, E_ref_exact);
rawBE = compute_block_energy_contributions(cr, Hraw, rawBasis);
fprintf('Raw solve: E=%.15f | err=%.3e | kept=%d | residual=%.3e\n', ...
    Er, abs(Er-E_ref_exact), keptRaw, rawSolveInfo.residual);
fprintf('Raw row energy: Ezero=%.6f Eone=%.6f Etwo_uu=%.6f sum=%.15f\n', ...
    rawBE.row.zero, rawBE.row.one, rawBE.row.two_uu, rawBE.row.total);

partT_raw = struct();
partT_raw.hfFile = hfFile;
partT_raw.hf = hf;
partT_raw.params = struct('Zcharge',Zcharge,'Rnuc',Rnuc,'sigma0',sigma0,'cscale',cscale, ...
    'L0',L0,'D',D,'ZsetMode',ZsetMode,'centerConvention',centerConvention, ...
    'neighborMode',neighborMode,'restrictInitialTwoNeighborsToFrame',restrictInitialTwoNeighborsToFrame, ...
    'detScaleMode',detScaleMode, 'detailProjectionMode',detailProjectionMode, ...
    'E_ref_exact',E_ref_exact);
partT_raw.oneFuncs = oneFuncs;
partT_raw.twoBasis = rawBasis;
partT_raw.S = Sraw; partT_raw.H = Hraw;
partT_raw.E = Er; partT_raw.coeff = cr; partT_raw.kept = keptRaw;
partT_raw.counts = rawCounts; partT_raw.blockEnergy = rawBE; partT_raw.assemblyTime = tAsmRaw;
rawTag = sprintf('mode_%s_restrict%d_scale_%s_proj_%s', neighborMode, restrictInitialTwoNeighborsToFrame, detScaleMode, detailProjectionMode);
rawFile = sprintf('JH_HeTriplet_raw_assembled_%s_M%d.mat', rawTag, rawCounts.total);
save(rawFile, 'partT_raw', '-v7.3');
fprintf('Saved raw system: %s\n', rawFile);

% Coarsening.
hartree_per_kcalmol = 1/627.5094740631;
delta = 0.01 * hartree_per_kcalmol;
[coarseBasis, Scoarse, Hcoarse, Ecoarse, ccoarse, coarseRec, idxBest, nuBest] = ...
    coarsen_basis_by_coeff(rawBasis, Sraw, Hraw, cr, Er, delta, mass_tol, verboseEig, E_ref_exact);
coarseCounts = count_blocks(coarseBasis);
coarseBE = compute_block_energy_contributions(ccoarse, Hcoarse, coarseBasis);

fprintf('\n==================== SELECTED TRIPLET COARSE SYSTEM ====================\n');
fprintf('nu=%d | M=%d | Mzero=%d | Mone=%d | Mtwo_uu=%d\n', ...
    nuBest, coarseCounts.total, coarseCounts.zero, coarseCounts.one_total, coarseCounts.two_uu);
fprintf('E0=%.15f | rawE=%.15f | E0-raw=%.3e | delta=%.3e\n', Ecoarse, Er, Ecoarse-Er, delta);
fprintf('Table 5.5 kappa=1 target: M=141, E=-2.174698, Mone=76, Mtwo_uu=64\n');
fprintf('Target deviations: dM=%+d, dMone=%+d, dMtwo=%+d, dE=%.3e\n', ...
    coarseCounts.total-141, coarseCounts.one_total-76, coarseCounts.two_uu-64, Ecoarse-(-2.174698));
fprintf('Coarse row energy: Ezero=%.6f Eone=%.6f Etwo_uu=%.6f sum=%.15f\n', ...
    coarseBE.row.zero, coarseBE.row.one, coarseBE.row.two_uu, coarseBE.row.total);

partT = struct();
partT.rawFile = rawFile;
partT.hfFile = hfFile;
partT.hf = hf;
partT.params = partT_raw.params;
partT.delta = delta;
partT.recordsCoarsen = coarseRec;
partT.idxBest = idxBest;
partT.nuBest = nuBest;
partT.oneFuncs = oneFuncs;
partT.twoBasis0 = coarseBasis;
partT.S0 = Scoarse; partT.H0 = Hcoarse;
partT.E0 = Ecoarse; partT.coeff0 = ccoarse; partT.counts0 = coarseCounts; partT.blockEnergy0 = coarseBE;
tag = sprintf('mode_%s_restrict%d_scale_%s_proj_%s', neighborMode, restrictInitialTwoNeighborsToFrame, detScaleMode, detailProjectionMode);
coarseFile = sprintf('JH_HeTriplet_coarsened_%s_M%d_nu%d.mat', tag, coarseCounts.total, nuBest);
save(coarseFile, 'partT', '-v7.3');
fprintf('Saved coarsened system: %s\n', coarseFile);


%% End of coarsening/build stage. Adaptive is separated into JH_HeTriplet_part3_adaptive_from_coarse_debug_v1.m
%% ========================================================================
% Local functions: records
% ========================================================================
function r = empty_record()
    r = struct('kappa',0,'M',0,'E',NaN,'err',NaN,'Mzero',0,'Mone',0,'Mtwo_uu',0, ...
        'Ezero',NaN,'Eone',NaN,'Etwo_uu',NaN,'inner',0,'solverResidual',NaN);
end

function r = make_record(kappa, M, E, err, counts, be, inner, solverInfo)
    r = empty_record();
    r.kappa = kappa; r.M = M; r.E = E; r.err = err;
    r.Mzero = counts.zero; r.Mone = counts.one_total; r.Mtwo_uu = counts.two_uu;
    r.Ezero = be.row.zero; r.Eone = be.row.one; r.Etwo_uu = be.row.two_uu;
    r.inner = inner; r.solverResidual = solverInfo.residual;
end

function r = empty_m_record()
    r = struct('sampleID',0,'kappa',0,'inner',0,'epsilon',NaN,'M',0, ...
        'Mzero',0,'Mone',0,'Mtwo_uu',0,'E',NaN,'err',NaN, ...
        'Ezero',NaN,'Eone',NaN,'Etwo_uu',NaN,'kept',0,'solverResidual',NaN, ...
        'assemblyTime',NaN,'solveTime',NaN,'importanceTime',NaN,'blockEnergyTime',NaN, ...
        'growTime',NaN,'saveTime',NaN,'totalInnerTimeNoSave',NaN,'totalInnerTimeWithSave',NaN, ...
        'cumulativeTime',NaN,'nImportant',0,'scoreMax',NaN,'scoreMinNonzero',NaN, ...
        'nGrowRaw',0,'nAdded',0,'basisFile','');
end

function [recordsByM, tSave, basisFile] = append_M_record_and_checkpoint(recordsByM, sampleID, ...
    kappa, inner, epsK, twoBasis, E, err, ~, scores, important, growBasis, added, counts, be, solverInfo, ...
    assemblyTime, solveTime, importanceTime, blockEnergyTime, growTime, innerTimeNoSave, cumTime, oneFuncs, params, recordOpts)
    t0 = tic;
    r = empty_m_record();
    r.sampleID = sampleID; r.kappa = kappa; r.inner = inner; r.epsilon = epsK;
    r.M = numel(twoBasis); r.Mzero = counts.zero; r.Mone = counts.one_total; r.Mtwo_uu = counts.two_uu;
    r.E = E; r.err = err; r.Ezero = be.row.zero; r.Eone = be.row.one; r.Etwo_uu = be.row.two_uu;
    r.kept = solverInfo.kept; r.solverResidual = solverInfo.residual;
    r.assemblyTime = assemblyTime; r.solveTime = solveTime; r.importanceTime = importanceTime;
    r.blockEnergyTime = blockEnergyTime; r.growTime = growTime;
    r.totalInnerTimeNoSave = innerTimeNoSave; r.cumulativeTime = cumTime;
    r.nImportant = nnz(important); r.scoreMax = max(scores);
    sp = scores(scores>0); if isempty(sp), r.scoreMinNonzero = 0; else, r.scoreMinNonzero = min(sp); end
    r.nGrowRaw = numel(growBasis); r.nAdded = added;
    basisFile = '';
    if recordOpts.saveBasisEveryMRecord
        basisFile = sprintf('%s_sample%04d_kappa%d_inner%d_M%d.mat', recordOpts.saveEveryInnerPrefix, sampleID, kappa, inner, r.M);
        record = r; %#ok<NASGU>
        save(basisFile, 'oneFuncs', 'twoBasis', 'record', 'params', '-v7.3');
    end
    r.basisFile = basisFile;
    tSave = toc(t0);
    r.saveTime = tSave;
    r.totalInnerTimeWithSave = innerTimeNoSave + tSave;
    recordsByM(end+1) = r; %#ok<AGROW>
    if recordOpts.saveRecordsEveryInner
        save('JH_HeTriplet_records_live.mat', 'recordsByM', '-v7.3');
    end
end

function recordsUnique = unique_records_by_M_keep_best(recordsByM)
    if isempty(recordsByM), recordsUnique = recordsByM; return; end
    Mvals = [recordsByM.M].'; [Mu,~,ic] = unique(Mvals,'stable');
    recordsUnique = repmat(empty_m_record(), numel(Mu), 1);
    for i = 1:numel(Mu)
        idx = find(ic==i); [~,j] = min([recordsByM(idx).err]); recordsUnique(i) = recordsByM(idx(j));
    end
end

%% ========================================================================
% Local functions: basis and growth
% ========================================================================
function f = empty_onefunc()
    f = struct('kind','','type','','level',0,'j',[0 0 0],'z',[0 0 0], ...
        'terms',struct('coef',{},'alpha',{},'center',{}), ...
        'key',string(''),'rawKey',string(''),'projected',false);
end

function b = empty_twobasis()
    b = struct('block','','left',0,'right',0,'pReplaced',0,'key',string(''));
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

function projU = build_reference_projector(ref1, ref2, Zcharge, Rnuc)
    [S11,~,~] = contracted_one_particle(ref1, ref1, Zcharge, Rnuc);
    [S22,~,~] = contracted_one_particle(ref2, ref2, Zcharge, Rnuc);
    [S12,~,~] = contracted_one_particle(ref1, ref2, Zcharge, Rnuc);
    G = [S11 S12; S12 S22];
    projU = struct('ref1',ref1,'ref2',ref2,'G',G,'Ginv',inv(G),'Zcharge',Zcharge,'Rnuc',Rnuc);
end

function f = prepare_detail_onefunc(fraw, projU, mode)
    % Keep the multiscale metadata (type/level/j/z) for center and neighbor mapping,
    % but replace the actual analytic terms by the detail function in W if requested.
    if string(fraw.kind) ~= "frame"
        f = fraw;
        if strlength(f.rawKey)==0, f.rawKey = f.key; end
        return;
    end
    switch lower(mode)
        case {'none','off','raw'}
            f = fraw;
            if strlength(f.rawKey)==0, f.rawKey = f.key; end
            f.projected = false;
        case {'orthogonal_to_u','projectw','w','orthogonal'}
            f = project_onefunc_to_reference_complement(fraw, projU);
        otherwise
            error('Unknown detailProjectionMode: %s', mode);
    end
end

function f = project_onefunc_to_reference_complement(fraw, projU)
    [b1,~,~] = contracted_one_particle(projU.ref1, fraw, projU.Zcharge, projU.Rnuc);
    [b2,~,~] = contracted_one_particle(projU.ref2, fraw, projU.Zcharge, projU.Rnuc);
    coeffU = projU.Ginv * [b1; b2];

    % phi_W = phi - coeffU(1)*psi1 - coeffU(2)*psi2
    terms = fraw.terms;
    for a = 1:numel(projU.ref1.terms)
        t = projU.ref1.terms(a);
        t.coef = -coeffU(1) * t.coef;
        terms(end+1) = t; %#ok<AGROW>
    end
    for a = 1:numel(projU.ref2.terms)
        t = projU.ref2.terms(a);
        t.coef = -coeffU(2) * t.coef;
        terms(end+1) = t; %#ok<AGROW>
    end
    f = fraw;
    f.rawKey = fraw.key;
    f.key = string("W|" + string(fraw.key));
    f.terms = compress_terms(terms, 1e-14);
    f.projected = true;
end

function termsOut = compress_terms(termsIn, tol)
    % Combine exactly matching primitive terms.  This keeps projected detail functions
    % reasonably compact, because the same HF primitives are appended many times.
    if isempty(termsIn), termsOut = termsIn; return; end
    keys = strings(numel(termsIn),1);
    for i = 1:numel(termsIn)
        c = termsIn(i).center;
        keys(i) = string(sprintf('a%.16e_C%.16e_%.16e_%.16e', termsIn(i).alpha, c(1), c(2), c(3)));
    end
    [ukeys,~,ic] = unique(keys,'stable');
    termsOut = repmat(termsIn(1), 0, 1);
    for k = 1:numel(ukeys)
        idx = find(ic==k);
        coef = sum([termsIn(idx).coef]);
        if abs(coef) > tol
            t = termsIn(idx(1));
            t.coef = coef;
            termsOut(end+1) = t; %#ok<AGROW>
        end
    end
end

function [B, map] = append_basis(B, map, b)
    if isempty(b) || b.left == b.right, return; end
    key = char(b.key);
    if ~isKey(map,key)
        B(end+1) = b; 
        map(key) = numel(B);
    end
end

function b = make_det_basis(block, idA, idB, pReplaced)
    b = empty_twobasis();
    if idA == idB
        b.left = idA; b.right = idB; b.key = string('ZERO_DET'); return;
    end
    ids = sort([idA idB]);
    b.block = char(block);
    b.left = ids(1);
    b.right = ids(2);
    b.pReplaced = pReplaced;
    b.key = string(sprintf('%s_D%d_%d_p%d', char(block), b.left, b.right, pReplaced));
end

function [Buniq, map] = unique_twobasis(B)
    if isempty(B), Buniq = B; map = containers.Map('KeyType','char','ValueType','double'); return; end
    keys = strings(numel(B),1);
    for i = 1:numel(B), keys(i)=string(B(i).key); end
    [~,ia] = unique(keys,'stable'); Buniq = B(ia);
    map = containers.Map('KeyType','char','ValueType','double');
    for i = 1:numel(Buniq), map(char(Buniq(i).key)) = i; end
end

function counts = count_blocks(twoBasis)
    M = numel(twoBasis);
    blocks = strings(M,1);
    for i = 1:M, blocks(i)=string(twoBasis(i).block); end
    counts = struct();
    counts.zero = nnz(blocks=="zero");
    counts.one_total = nnz(blocks=="one");
    counts.two_uu = nnz(blocks=="two_uu");
    counts.total = M;
end

function [growBasis, oneFuncs, oneKeyMap] = grow_important_det_basis(importantBasis, oneFuncs, oneKeyMap, ...
    sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode, projU, detailProjectionMode)
    out = {}; k = 0;
    ref1 = 1; ref2 = 2;
    for ib = 1:numel(importantBasis)
        B = importantBasis(ib);
        switch string(B.block)
            case "zero"
                k = k+1; out{k} = B; %#ok<AGROW>
            case "one"
                if B.pReplaced == 1
                    active = setdiff([B.left B.right], ref2);
                    if isempty(active), continue; end
                    active = active(1); other = ref2;
                elseif B.pReplaced == 2
                    active = setdiff([B.left B.right], ref1);
                    if isempty(active), continue; end
                    active = active(1); other = ref1;
                else
                    % Fallback: grow whichever entry is not a reference.
                    ids = [B.left B.right]; active = ids(ids>2); if isempty(active), continue; end
                    active = active(1); other = ids(ids<=2); if isempty(other), other = ref1; else, other=other(1); end
                end
                neigh = onefunc_neighbors_both(oneFuncs(active), sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode);
                for n = 1:numel(neigh)
                    nf = prepare_detail_onefunc(neigh(n), projU, detailProjectionMode);
                    [oneFuncs, oneKeyMap, nid] = register_onefunc(oneFuncs, oneKeyMap, nf);
                    k = k+1; out{k} = make_det_basis('one', nid, other, B.pReplaced); %#ok<AGROW>
                end
            case "two_uu"
                fL = oneFuncs(B.left); fR = oneFuncs(B.right);
                neighL = onefunc_neighbors_both(fL, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode);
                neighR = onefunc_neighbors_both(fR, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode);
                for n = 1:numel(neighL)
                    nf = prepare_detail_onefunc(neighL(n), projU, detailProjectionMode);
                    [oneFuncs, oneKeyMap, nid] = register_onefunc(oneFuncs, oneKeyMap, nf);
                    if nid ~= B.right
                        k = k+1; out{k} = make_det_basis('two_uu', nid, B.right, 0); %#ok<AGROW>
                    end
                end
                for n = 1:numel(neighR)
                    nf = prepare_detail_onefunc(neighR(n), projU, detailProjectionMode);
                    [oneFuncs, oneKeyMap, nid] = register_onefunc(oneFuncs, oneKeyMap, nf);
                    if nid ~= B.left
                        k = k+1; out{k} = make_det_basis('two_uu', B.left, nid, 0); %#ok<AGROW>
                    end
                end
        end
    end
    if isempty(out), growBasis = importantBasis([]); else, growBasis = [out{:}]; end
end

%% ========================================================================
% Local functions: frame and neighbors
% ========================================================================
function frame = make_initial_frame(sigma0, cscale, L, D, ZsetMode, centerConvention)
    % Same finite b^{(L)}_{sigma,c} convention as the verified singlet He code:
    %   27 coarse phi functions plus 26 second-order wavelet functions per level.
    % Hence signed Zset gives 27 + 26*(L-1), e.g. 183 for L=7.
    cellF = {}; k = 0;
    vals = -1:1;
    [J1,J2,J3] = ndgrid(vals,vals,vals);
    JJ = [J1(:),J2(:),J3(:)];
    for i = 1:size(JJ,1)
        k=k+1; cellF{k}=make_phi(sigma0,cscale,0,JJ(i,:),D); %#ok<AGROW>
    end
    if L >= 2
        Zset = generate_z_set(D,ZsetMode);
        for l = 0:(L-2)
            for iz = 1:size(Zset,1)
                z = Zset(iz,:);
                j = -z;
                k=k+1; cellF{k}=make_psi(sigma0,cscale,l,j,z,D,centerConvention); %#ok<AGROW>
            end
        end
    end
    frame = unique_onefuncs([cellF{:}]);
end

function neigh = onefunc_neighbors_both(f, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode)
    if ~isfield(f,'kind') || string(f.kind) ~= "frame"
        neigh = f([]); return;
    end
    neigh = unique_onefuncs([neighbors_Ng_onefunc(f,sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode), ...
                             neighbors_Ngplus_onefunc(f,sigma0,cscale,D,ZsetMode,centerConvention,centerForNeighbors,neighborMode)]);
end

function neigh = neighbors_Ng_onefunc(f, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode)
    Z0 = generate_neighbor_set(neighborMode);
    C0 = center_of_basis(f,cscale,centerForNeighbors);
    cellF = {}; k = 0;
    switch string(f.type)
        case "phi"
            step = 1/cscale;
            for n = 1:size(Z0,1)
                fs = functions_at_center(C0+step*Z0(n,:), sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, 'phi', 0);
                for j = 1:numel(fs), k=k+1; cellF{k}=fs(j); end %#ok<AGROW>
            end
        case "psi"
            l = f.level; step = 1/(cscale*2^(l+1));
            for n = 1:size(Z0,1)
                fs = functions_at_center(C0+step*Z0(n,:), sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, 'psi', l);
                for j = 1:numel(fs), k=k+1; cellF{k}=fs(j); end %#ok<AGROW>
            end
    end
    if isempty(cellF), neigh = f([]); else, neigh = unique_onefuncs([cellF{:}]); end
end

function neigh = neighbors_Ngplus_onefunc(f, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, neighborMode)
    Z0 = generate_neighbor_set(neighborMode);
    C0 = center_of_basis(f,cscale,centerForNeighbors);
    cellF = {}; k = 0;
    switch string(f.type)
        case "phi"
            step = 1/(cscale*2); level = 0;
            for n = 1:size(Z0,1)
                fs = functions_at_center(C0+step*Z0(n,:), sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, 'psi', level);
                for j = 1:numel(fs), k=k+1; cellF{k}=fs(j); end %#ok<AGROW>
            end
        case "psi"
            level = f.level + 1; step = 1/(cscale*2^(f.level+2));
            for n = 1:size(Z0,1)
                fs = functions_at_center(C0+step*Z0(n,:), sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, 'psi', level);
                for j = 1:numel(fs), k=k+1; cellF{k}=fs(j); end %#ok<AGROW>
            end
    end
    if isempty(cellF), neigh = f([]); else, neigh = unique_onefuncs([cellF{:}]); end
end

function fs = functions_at_center(target, sigma0, cscale, D, ZsetMode, centerConvention, centerForNeighbors, kind, level)
    fsCell = {}; k = 0;
    switch string(kind)
        case "phi"
            scale = cscale * 2^level;
            j = round(target * scale);
            f = make_phi(sigma0,cscale,level,j,D);
            if norm(center_of_basis(f,cscale,centerForNeighbors)-target) < 1e-10
                k=k+1; fsCell{k}=f;
            end
        case "psi"
            Zset = generate_z_set(D,ZsetMode);
            scale = cscale * 2^level;
            for iz = 1:size(Zset,1)
                z = Zset(iz,:);
                switch lower(centerConvention)
                    case 'eq428'
                        j = round(target*scale - 0.5*z);
                    case 'cmap'
                        j = round(target*scale + 0.5*z);
                    otherwise
                        error('Unknown centerConvention.');
                end
                f = make_psi(sigma0,cscale,level,j,z,D,centerConvention);
                if norm(center_of_basis(f,cscale,centerForNeighbors)-target) < 1e-10
                    k=k+1; fsCell{k}=f; %#ok<AGROW>
                end
            end
    end
    if isempty(fsCell), fs = empty_onefunc(); fs = fs([]); else, fs = unique_onefuncs([fsCell{:}]); end
end

function Z0 = generate_neighbor_set(mode)
    switch lower(mode)
        case 'cross'
            Z0 = [0 0 0; 1 0 0; -1 0 0; 0 1 0; 0 -1 0; 0 0 1; 0 0 -1];
        case 'full'
            vals = -1:1; [A,B,C] = ndgrid(vals,vals,vals); Z0=[A(:),B(:),C(:)];
        otherwise
            error('Unknown neighborMode.');
    end
end

function Zset = generate_z_set(D, mode)
    switch lower(mode)
        case 'signed'
            vals = -1:1; [A,B,C] = ndgrid(vals,vals,vals); Zset=[A(:),B(:),C(:)]; Zset=Zset(any(Zset~=0,2),:);
        case 'binary'
            Zset = dec2bin(0:(2^D-1))-'0'; Zset=Zset(any(Zset~=0,2),:);
        otherwise
            error('Unknown ZsetMode.');
    end
end

function f = make_phi(sigma0,cscale,l,j,D)
    scale = cscale*2^l; width = sigma0/scale; alpha = 1/width^2; center = j/scale;
    term = struct('coef',1.0,'alpha',alpha,'center',center);
    f = empty_onefunc();
    f.kind='frame'; f.type='phi'; f.level=l; f.j=round(j); f.z=zeros(1,D); f.terms=term;
    f.key = canonical_key_from_center('phi',l,center,cscale);
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
    f = empty_onefunc();
    f.kind='frame'; f.type='psi'; f.level=l; f.j=round(j); f.z=round(z); f.terms=[term1,term2];
    f.key = canonical_key_from_center('psi',l,center,cscale);
end

function key = canonical_key_from_center(type, level, center, cscale)
    scale = cscale*2^level; idx2 = round(2*scale*center);
    key = string(sprintf('%s_l%d_C2_%d_%d_%d', type, level, idx2(1), idx2(2), idx2(3)));
end

function C = center_of_basis(f,cscale,mode)
    switch lower(mode)
        case 'actual'
            C = f.terms(1).center;
        case 'paper_minus'
            if string(f.type)=="phi", C=f.j/cscale; else, C=(f.j-0.5*f.z)/(cscale*2^f.level); end
        otherwise
            error('Unknown center mode.');
    end
end

function B = unique_onefuncs(B)
    if isempty(B), return; end
    keys = strings(numel(B),1); for i=1:numel(B), keys(i)=string(B(i).key); end
    [~,ia] = unique(keys,'stable'); B = B(ia);
end

%% ========================================================================
% Local functions: assembly
% ========================================================================
function oneCache = reset_one_cache(Zcharge,Rnuc)
    oneCache = struct('S1',[],'H1',[],'M',0,'Zcharge',Zcharge,'Rnuc',Rnuc);
end

function oneCache = update_one_cache(oneCache, oneFuncs, Zcharge, Rnuc)
    Mnew = numel(oneFuncs); Mold = oneCache.M;
    if Mold == Mnew, return; end
    S1 = zeros(Mnew,Mnew); H1 = zeros(Mnew,Mnew);
    if Mold > 0
        S1(1:Mold,1:Mold) = oneCache.S1;
        H1(1:Mold,1:Mold) = oneCache.H1;
    end
    for i = 1:Mnew
        jStart = max(i, Mold+1);
        for j = jStart:Mnew
            [S,T,Ven] = contracted_one_particle(oneFuncs(i),oneFuncs(j),Zcharge,Rnuc);
            S1(i,j)=S; S1(j,i)=S;
            H1(i,j)=T+Ven; H1(j,i)=H1(i,j);
        end
    end
    oneCache.S1=S1; oneCache.H1=H1; oneCache.M=Mnew; oneCache.Zcharge=Zcharge; oneCache.Rnuc=Rnuc;
end

function [S2,H2,info] = assemble_twobody_matrices(twoBasis, oneCache, oneFuncs, opts)
    M = numel(twoBasis);
    S1 = oneCache.S1; H1 = oneCache.H1;
    td = build_term_data(oneFuncs);
    normInv = determinant_norm_inv(twoBasis,S1,opts);

    if strcmpi(opts.mode,'cpu_parfor_sparse') && opts.startPool
        try
            p = gcp('nocreate'); if isempty(p), parpool; end
        catch ME
            warning('Could not start parpool. Falling back to serial_sparse: %s', ME.message);
            opts.mode = 'serial_sparse';
        end
    end

    rowBlockSize = opts.rowBlockSize;
    S2 = sparse(M,M); H2 = sparse(M,M);
    t0 = tic;
    for r0 = 1:rowBlockSize:M
        r1 = min(M, r0+rowBlockSize-1);
        rows = r0:r1;
        [Srows,Hrows] = assemble_det_sparse_rows(twoBasis, rows, S1, H1, td, normInv, opts);
        S2(rows,:) = Srows;
        H2(rows,:) = Hrows;
        if opts.showRowProgress
            fprintf('    det sparse rows %d-%d / %d inserted, nnz(S/H)=%.3e/%.3e, elapsed %.1fs\n', ...
                r0,r1,M,nnz(S2),nnz(H2),toc(t0));
        end
    end
    if isfield(opts,'upperTriangle') && opts.upperTriangle
        S2 = triu(S2); H2 = triu(H2);
        S2 = S2 + triu(S2,1)';
        H2 = H2 + triu(H2,1)';
    else
        S2 = (S2+S2')/2; H2=(H2+H2')/2;
    end
    info = struct('M',M,'nnzS',nnz(S2),'nnzH',nnz(H2),'elapsed',toc(t0));
end

function normInv = determinant_norm_inv(tb,S1,opts)
    M = numel(tb);
    mode = 'l2_normalized';
    if isfield(opts,'detScaleMode'), mode = lower(char(string(opts.detScaleMode))); end
    switch mode
        case {'antisymmetrizer','unnormalized','raw_a'}
            normInv = ones(1,M);
        case {'l2_normalized','normalized_det','normalized'}
            normInv = zeros(1,M);
            for i = 1:M
                a=tb(i).left; b=tb(i).right;
                n2 = S1(a,a)*S1(b,b) - S1(a,b)*S1(b,a);
                if n2 < 1e-14, n2 = 1e-14; end
                normInv(i) = 1/sqrt(n2);
            end
        otherwise
            error('Unknown detScaleMode: %s', mode);
    end
end

function [Srows,Hrows] = assemble_det_sparse_rows(tb, rowIdx, S1, H1, td, normInv, opts)
    nr = numel(rowIdx); M = numel(tb);
    leftCols = [tb.left]; rightCols = [tb.right]; ncol = 1:M;
    switch lower(opts.mode)
        case 'cpu_parfor_sparse'
            IScell=cell(nr,1); JScell=cell(nr,1); VScell=cell(nr,1);
            IHcell=cell(nr,1); JHcell=cell(nr,1); VHcell=cell(nr,1);
            parfor ir = 1:nr
                [js,vs,jh,vh] = assemble_one_det_row(tb,rowIdx(ir),leftCols,rightCols,S1,H1,td,normInv,opts);
                IScell{ir}=ir*ones(numel(js),1); JScell{ir}=js(:); VScell{ir}=vs(:);
                IHcell{ir}=ir*ones(numel(jh),1); JHcell{ir}=jh(:); VHcell{ir}=vh(:);
            end
            IS=vertcat(IScell{:}); JS=vertcat(JScell{:}); VS=vertcat(VScell{:});
            IH=vertcat(IHcell{:}); JH=vertcat(JHcell{:}); VH=vertcat(VHcell{:});
        otherwise
            IS=[];JS=[];VS=[];IH=[];JH=[];VH=[];
            for ir = 1:nr
                [js,vs,jh,vh] = assemble_one_det_row(tb,rowIdx(ir),leftCols,rightCols,S1,H1,td,normInv,opts);
                IS=[IS; ir*ones(numel(js),1)]; JS=[JS; js(:)]; VS=[VS; vs(:)]; %#ok<AGROW>
                IH=[IH; ir*ones(numel(jh),1)]; JH=[JH; jh(:)]; VH=[VH; vh(:)]; %#ok<AGROW>
            end
    end
    Srows = sparse(IS,JS,VS,nr,M);
    Hrows = sparse(IH,JH,VH,nr,M);
end

function [js,vs,jh,vh] = assemble_one_det_row(tb,a,leftCols,rightCols,S1,H1,td,normInv,opts)
    i = tb(a).left; j = tb(a).right;
    k = leftCols; l = rightCols;
    if isfield(opts,'upperTriangle') && opts.upperTriangle
        upperMask = (1:numel(leftCols)) >= a;
    else
        upperMask = true(size(leftCols));
    end
    Sik = S1(i,k); Sjl = S1(j,l); Sil = S1(i,l); Sjk = S1(j,k);
    Hik = H1(i,k); Hjl = H1(j,l); Hil = H1(i,l); Hjk = H1(j,k);

    Sraw = Sik.*Sjl - Sil.*Sjk;
    Hone = Hik.*Sjl + Sik.*Hjl - Hil.*Sjk - Sil.*Hjk;
    Jdir = eri_row_vectorized_cpu(td, i, k, j, l);
    Jex  = eri_row_vectorized_cpu(td, i, l, j, k);
    Hraw = Hone + Jdir - Jex;
    fac = normInv(a) .* normInv;
    srow = Sraw .* fac;
    hrow = Hraw .* fac;
    maskS = upperMask & (abs(srow) >= opts.dropTolS);
    maskH = upperMask & (abs(hrow) >= opts.dropTolH);
    js = find(maskS); vs = srow(maskS);
    jh = find(maskH); vh = hrow(maskH);
end

function td = build_term_data(oneFuncs)
    M = numel(oneFuncs); maxT = 0;
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
    nc = numel(ibVec); vrow = zeros(1,nc); T=td.maxTerms;
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
    S=0;T=0;Ven=0;
    for a=1:numel(f.terms)
        A=f.terms(a);
        for b=1:numel(g.terms)
            B=g.terms(b);
            [s,t,v]=one_particle_primitive(A.alpha,A.center,B.alpha,B.center,Z,Rnuc);
            coef=A.coef*B.coef;
            S=S+coef*s; T=T+coef*t; Ven=Ven+coef*v;
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
    ts=t(small); F(small)=1-ts/3+ts.^2/10-ts.^3/42+ts.^4/216-ts.^5/1320;
    tl=t(~small); F(~small)=0.5*sqrt(pi)./sqrt(tl).*erf(sqrt(tl));
end

function F = boys0(t)
    if t<1e-10, F=1-t/3+t^2/10-t^3/42+t^4/216-t^5/1320; else, F=0.5*sqrt(pi)/sqrt(t)*erf(sqrt(t)); end
end

%% ========================================================================
% Local functions: solve, coarsen, diagnostics
% ========================================================================
function [E0,coeff,nkeep,info] = solve_ground_projected(H,S,mass_tol,verbose,targetEnergy)
    M=size(S,1); t0=tic;
    Sf=full((S+S')/2); Hf=full((H+H')/2);
    [U,D]=eig(Sf); d=real(diag(D)); keep=d>mass_tol*max(d);
    nkeep=nnz(keep);
    X=U(:,keep)*diag(1./sqrt(d(keep)));
    Horth=(X'*Hf*X); Horth=(Horth+Horth')/2;
    [V,Ediag]=eig(Horth); evals=real(diag(Ediag)); [E0,idx]=min(evals);
    y=V(:,idx); coeff=X*y; coeff=coeff/sqrt(real(coeff'*S*coeff));
    res=norm(H*coeff-E0*(S*coeff))/(max(1,norm(H*coeff))+max(1,abs(E0)*norm(S*coeff)));
    info=struct('kept',nkeep,'residual',res,'time',toc(t0),'method','full-mass-projection');
    if verbose
        fprintf('    mass eig M=%d kept=%d E=%.15f targetDiff=%.3e res=%.3e\n',M,nkeep,E0,E0-targetEnergy,res);
    end
end

function [coarseBasis,S0,H0,E0,c0,records,idxBest,nuBest] = coarsen_basis_by_coeff(rawBasis,Sraw,Hraw,craw,Eraw,delta,mass_tol,verbose,Eref)
    coefAbs=abs(craw(:)); nuList=0:80;
    records=struct('nu',{},'thr',{},'M',{},'Mzero',{},'Mone',{},'Mtwo',{},'E',{},'dE',{},'idx',{});
    fprintf('\nCoarsening scan B(nu) = {|v_mu| >= 2^{-nu}}:\n');
    fprintf('   nu       threshold          M     Mzero   Mone   Mtwo_uu        E(nu)              dE\n');
    for inu=1:numel(nuList)
        nu=nuList(inu); thr=2^(-nu);
        idx=find(coefAbs>=thr); idx=unique([1;idx(:)],'stable');
        [Esub,~,~,~]=solve_ground_projected(Hraw(idx,idx),Sraw(idx,idx),mass_tol,verbose,Eref);
        dE=Esub-Eraw; counts=count_blocks(rawBasis(idx));
        records(inu).nu=nu; records(inu).thr=thr; records(inu).M=numel(idx); records(inu).Mzero=counts.zero;
        records(inu).Mone=counts.one_total; records(inu).Mtwo=counts.two_uu; records(inu).E=Esub; records(inu).dE=dE; records(inu).idx=idx;
        fprintf('%5d   %.6e   %6d   %5d  %5d  %7d   %.15f   %.3e\n',nu,thr,numel(idx),counts.zero,counts.one_total,counts.two_uu,Esub,dE);
    end
    valid=find([records.dE] <= delta+1e-12 & [records.dE] >= -1e-10);
    if isempty(valid)
        warning('No B(nu) satisfies dE <= delta; selecting largest candidate.'); [~,pos]=max([records.M]); selected=records(pos);
    else
        [~,loc]=min([records(valid).M]); selected=records(valid(loc));
    end
    idxBest=selected.idx; nuBest=selected.nu;
    coarseBasis=rawBasis(idxBest); S0=Sraw(idxBest,idxBest); H0=Hraw(idxBest,idxBest);
    [E0,c0,~,~]=solve_ground_projected(H0,S0,mass_tol,verbose,Eref);
end

function scores = partial_orthogonal_importance(c,S)
    c=c(:); M=numel(c);
    if M==1, scores=abs(c); return; end
    B11=full(S(1,1)); B12=full(S(1,2:end));
    z=c; z(1)=c(1)+(B12/B11)*c(2:end);
    diagPO=zeros(M,1); diagPO(1)=B11;
    diagPO(2:end)=full(diag(S(2:end,2:end))) - (B12(:).^2)/B11;
    diagPO=max(real(diagPO),0);
    scores=abs(sqrt(diagPO).*z);
end

function be = empty_block_energy()
    z=struct('zero',NaN,'one',NaN,'two_uu',NaN,'total',NaN);
    be=struct('row',z,'diag',z);
end

function be = compute_block_energy_contributions(c,H,twoBasis)
    be=empty_block_energy();
    blocks=strings(numel(twoBasis),1); for i=1:numel(twoBasis), blocks(i)=string(twoBasis(i).block); end
    idxZero=find(blocks=="zero"); idxOne=find(blocks=="one"); idxTwo=find(blocks=="two_uu");
    be.row.zero=row_energy(c,H,idxZero); be.row.one=row_energy(c,H,idxOne); be.row.two_uu=row_energy(c,H,idxTwo);
    be.row.total=be.row.zero+be.row.one+be.row.two_uu;
    be.diag.zero=diag_energy(c,H,idxZero); be.diag.one=diag_energy(c,H,idxOne); be.diag.two_uu=diag_energy(c,H,idxTwo);
    be.diag.total=be.diag.zero+be.diag.one+be.diag.two_uu;
end

function e=row_energy(c,H,idx)
    if isempty(idx), e=0; else, e=real(c(idx)'*(H(idx,:)*c)); end
end
function e=diag_energy(c,H,idx)
    if isempty(idx), e=0; else, e=real(c(idx)'*(H(idx,idx)*c(idx))); end
end

function gb=estimate_sparse_pair_gb(S,H)
    ns=nnz(S); nh=nnz(H); m=size(S,1); gb=(16*(ns+nh)+16*(m+1))/1024^3;
end
