%% JH_BeGround_part2_coarsening_v3_termCountFast_LiStyle.m
% Be ground state, J.H. Table 5.10 style: raw B_three^[0] union B_other^[0]
% construction + coarsening for N=4, M_S=0, N_up=2, N_down=2.
% v3 termCountFast-LiStyle:
%   - keeps the v1/v2 canonical-Ng Be basis logic;
%   - removes the largest redundancy in v2: ERI primitive loops no longer run
%     over maxTerms for every frame function; each one-particle function uses
%     its own nTerms, exactly as the Li-style contraction logic should;
%   - uses a lighter raw assembly for coarsening selection and optional exact
%     selected reassembly for production output.
%
% This script is intentionally written as the Be analogue of the debugged Li
% coarsening script, but with the essential Be changes:
%   1) Be uses two alpha and two beta orbitals:
%        A_up[orb1,orb2] * A_down[orb3,orb4].
%   2) J.H. Sec. 5.3 uses c = 1/4 and L = 8 for Be.
%   3) Since N>3, the initial system is B_three^[0] union B_other^[0]
%      according to Eq. (4.61)--(4.63); there is no general V_four block in
%      the finite-order-three construction, only the two special B_other
%      functions from Eq. (4.62).
%   4) The Ng(psi) canonical-neighbor fix from the Li v3 script is kept:
%      each neighboring center C(phi)+h*z0 produces exactly one canonical
%      one-particle frame representative.
%
% Required input:
%   JH_BeGround_HF_Q9_orbitals.mat   or
%   JH_BeGround_HF_Q12_orbitals.mat  or any compatible hf file with
%   hf.orbitals(1:4).terms.
%
% Outputs:
%   JH_BeGround_BthreeOther0_raw_assembled_v3termFast_M*.mat
%   JH_BeGround_BthreeOther0_coarsened_v3termFast_M*_nu*.mat
%   JH_BeGround_BthreeOther0_coarsening_scan_v3termFast.csv

clear; clc; close all;
format long e;

%% ============================================================
% 0. Parameters from J.H. Sec. 5.3 and general Sec. 4.4
% =============================================================
hfFileCandidates = { ...
    'JH_BeGround_HF_Q12_orbitals.mat', ...
    'JH_BeGround_HF_Q9_orbitals.mat', ...
    'JH_BeGround_HF_Q9_RHF_SCF_DIIS_fast_compatible.mat', ...
    'JH_BeGround_HF_Q9_RHF_SCF_DIIS_orbitals.mat'};

hfFile = '';
for k = 1:numel(hfFileCandidates)
    if exist(hfFileCandidates{k}, 'file')
        hfFile = hfFileCandidates{k};
        break;
    end
end
if isempty(hfFile)
    error('Cannot find a compatible Be HF file. Run Be HF first and save hf.orbitals(1:4).');
end
load(hfFile, 'hf');
if ~exist('hf','var') || ~isfield(hf,'orbitals') || numel(hf.orbitals) < 4
    error('HF file %s does not contain hf.orbitals(1:4).', hfFile);
end

Zcharge = 4;
if isfield(hf,'Z'), Zcharge = hf.Z; end
Rnuc = [0 0 0];
D = 3;

% J.H. Sec. 5.3 Be ground state: c=1/4, L=8, sigma=1.
sigma0 = 1;
cscale = 1/4;
L0 = 8;

ZsetMode = 'signed';
centerConvention = 'eq428';
centerForNeighbors = 'actual';
neighborMode = 'cross';       % Z0={0, +-e1, +-e2, +-e3}

detScaleMode = 'l2_normalized';

% Diagnostics from J.H. Table 5.10 and 5.13.
E_ref_exact = -14.66736;
E_ref_HF_table = -14.57296;
E_ref_HF_limit = -14.57302;
E_ref_tilde_table = -14.65978;
E_table_kappa1 = -14.644213;

hartree_per_kcalmol = 1/627.5094740631;
delta = 0.01 * hartree_per_kcalmol;      % 1/100 kcal/mol

mass_tol = 1e-10;
verboseEig = false;
nuList = 0:90;

dryRunAfterBasisBuild = false;

assemblyOpts = struct();
assemblyOpts.mode = 'cpu_parfor_sparse';
% Be raw space is much larger than Li.  Exact raw assembly is prohibitively
% expensive on a workstation, so the recommended workflow is:
%   screened raw assembly -> coarsening selection -> optional exact reassembly
%   of the selected much smaller coarse space.
assemblyOpts.rowBlockSize = 8;
assemblyOpts.startPool = true;
assemblyOpts.dropTolS = 1e-13;
assemblyOpts.dropTolH = 1e-12;
assemblyOpts.screenERIByOverlap = true;
assemblyOpts.screenTolSForERI = 5e-9;
assemblyOpts.screenTolHCheapForERI = 5e-8;
assemblyOpts.eriCoeffTol = 1e-11;      % skip ERI terms multiplied by tiny cofactors
assemblyOpts.forceDiagonal = true;
assemblyOpts.showRowProgress = true;
assemblyOpts.progressEveryBlocks = 2;
assemblyOpts.upperTriangle = true;
assemblyOpts.detScaleMode = detScaleMode;

% For the final file used by adaptive, it is safer to rebuild S0/H0 on the
% selected coarse system without overlap screening.  Set false for a quick
% exploratory run, then true for the final production run.
exactReassembleSelected = false;  % first fast pass; set true for final production exact S0/H0
exactSelectedRowBlockSize = 8;
exactSelectedEriCoeffTol = 0;

fprintf('\n============================================================\n');
fprintf('J.H. Be ground-state coarsening: 2-alpha + 2-beta determinants\n');
fprintf('HF file: %s\n', hfFile);
fprintf('Parameters: Z=%g, sigma=%g, c=%g, L=%d, neighbor=%s, detScale=%s\n', ...
    Zcharge, sigma0, cscale, L0, neighborMode, detScaleMode);
fprintf('Initial system: B_three^[0] union B_other^[0] for N=4.\n');
fprintf('Expected Table 5.10 kappa=1: M=1417, E=-14.644213, Mone=524, Mud=562, Muu=264, Mthree=64, Mother=2.\n');
fprintf('Coarsening delta = %.6e hartree = 1/100 kcal/mol.\n', delta);
fprintf('Assembly: canonical Ng, dropTolS=%.1e, dropTolH=%.1e, screenERI=%d, rowBlock=%d, eriCoeffTol=%.1e.\n', ...
    assemblyOpts.dropTolS, assemblyOpts.dropTolH, assemblyOpts.screenERIByOverlap, assemblyOpts.rowBlockSize, assemblyOpts.eriCoeffTol);
fprintf('============================================================\n\n');

%% ============================================================
% 1. Register HF orbitals and one-particle initial frame b^{(L)}_{sigma,c}
% =============================================================
oneFuncs = repmat(empty_onefunc(), 0, 1);
oneKeyMap = containers.Map('KeyType','char','ValueType','double');

% HF orbital order: 1=alpha_core, 2=alpha_valence, 3=beta_core, 4=beta_valence.
for p = 1:4
    f = empty_onefunc();
    f.kind = 'ref'; f.type = 'ref'; f.level = -1;
    f.j = [0 0 0]; f.z = [0 0 0];
    f.terms = hf.orbitals(p).terms;
    f.key = string(sprintf('ref%d',p));
    [oneFuncs, oneKeyMap, ~] = register_onefunc(oneFuncs, oneKeyMap, f);
end

frameL = make_initial_frame(sigma0, cscale, L0, D, ZsetMode, centerConvention);
for i = 1:numel(frameL)
    [oneFuncs, oneKeyMap, ~] = register_onefunc(oneFuncs, oneKeyMap, frameL(i));
end
firstFrameID = 5;
lastFrameID = numel(oneFuncs);
frameIDs = firstFrameID:lastFrameID;
Mb = numel(frameIDs);
fprintf('One-particle frame b^(%d): Mb=%d. Expected signed count 27+26*(L-1)=%d.\n', ...
    L0, Mb, 27+26*(L0-1));

frame2 = make_initial_frame(sigma0, cscale, 2, D, ZsetMode, centerConvention);
frame2IDs = zeros(1,numel(frame2));
for i = 1:numel(frame2)
    [oneFuncs, oneKeyMap, frame2IDs(i)] = register_onefunc(oneFuncs, oneKeyMap, frame2(i));
end
fprintf('Three-block frame b^(2): M2=%d.\n', numel(frame2IDs));

% B_other functions in Eq. (4.62): phi_{0,0} and psi with z=+/-1 vector.
phi0 = make_phi(sigma0,cscale,0,[0 0 0],D);
[oneFuncs, oneKeyMap, phi0ID] = register_onefunc(oneFuncs, oneKeyMap, phi0);
psiPlus = make_psi(sigma0,cscale,0,[0 0 0],[1 1 1],D,centerConvention);
psiMinus = make_psi(sigma0,cscale,0,[0 0 0],[-1 -1 -1],D,centerConvention);
[oneFuncs, oneKeyMap, psiPlusID] = register_onefunc(oneFuncs, oneKeyMap, psiPlus);
[oneFuncs, oneKeyMap, psiMinusID] = register_onefunc(oneFuncs, oneKeyMap, psiMinus);

%% ============================================================
% 2. Build raw B_three^[0] union B_other^[0] according to Eq. (4.61)-(4.62)
% =============================================================
rawBasis = repmat(empty_bebasis(), 0, 1);
basisMap = containers.Map('KeyType','char','ValueType','double');

refA1 = 1; refA2 = 2; refB1 = 3; refB2 = 4;

% V_zero: HF determinant A_up[psi1,psi2] * A_down[psi3,psi4].
[rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('zero', refA1, refA2, refB1, refB2, 'zero'));

% V_one: replace each occupied one-particle function by b^(L).
for fid = frameIDs
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('one', fid, refA2, refB1, refB2, 'one_a1'));
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('one', refA1, fid, refB1, refB2, 'one_a2'));
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('one', refA1, refA2, fid, refB2, 'one_b1'));
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('one', refA1, refA2, refB1, fid, 'one_b2'));
end

% V_two^{up down}: replace one alpha and one beta by the same phi in b^(L).
for fid = frameIDs
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('two_ud', fid, refA2, fid, refB2, 'two_a1b1'));
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('two_ud', fid, refA2, refB1, fid, 'two_a1b2'));
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('two_ud', refA1, fid, fid, refB2, 'two_a2b1'));
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('two_ud', refA1, fid, refB1, fid, 'two_a2b2'));
end

% V_two^{up up}: replace both alpha or both beta functions by neighboring frame functions.
for fid = frameIDs
    neigh = neighbors_Ng_onefunc(oneFuncs(fid), sigma0, cscale, D, ZsetMode, ...
        centerConvention, centerForNeighbors, neighborMode);
    for n = 1:numel(neigh)
        [oneFuncs, oneKeyMap, nid] = register_onefunc(oneFuncs, oneKeyMap, neigh(n));
        if nid == fid, continue; end
        % alpha-alpha block
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('two_uu', nid, fid, refB1, refB2, 'two_aa_1'));
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('two_uu', fid, nid, refB1, refB2, 'two_aa_2'));
        % beta-beta block
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('two_uu', refA1, refA2, nid, fid, 'two_bb_1'));
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('two_uu', refA1, refA2, fid, nid, 'two_bb_2'));
    end
end

% V_three: triples with two particles of one spin and one of the opposite spin.
% Use b^(2) and Ng(phi), independent of L, exactly as Eq. (4.56) generalized by (4.61).
for fid = frame2IDs
    neigh = neighbors_Ng_onefunc(oneFuncs(fid), sigma0, cscale, D, ZsetMode, ...
        centerConvention, centerForNeighbors, neighborMode);
    for n = 1:numel(neigh)
        [oneFuncs, oneKeyMap, nid] = register_onefunc(oneFuncs, oneKeyMap, neigh(n));
        if nid == fid, continue; end
        % alpha-alpha + beta1; beta2 stays reference
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('three', nid, fid, fid, refB2, 'three_aa_b1_1'));
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('three', fid, nid, fid, refB2, 'three_aa_b1_2'));
        % alpha-alpha + beta2; beta1 stays reference
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('three', nid, fid, refB1, fid, 'three_aa_b2_1'));
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('three', fid, nid, refB1, fid, 'three_aa_b2_2'));
        % beta-beta + alpha1; alpha2 stays reference
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('three', fid, refA2, nid, fid, 'three_bb_a1_1'));
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('three', fid, refA2, fid, nid, 'three_bb_a1_2'));
        % beta-beta + alpha2; alpha1 stays reference
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('three', refA1, fid, nid, fid, 'three_bb_a2_1'));
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('three', refA1, fid, fid, nid, 'three_bb_a2_2'));
    end
end

% B_other: the two special N-particle functions from Eq. (4.62), atomic case.
[rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('other', phi0ID, psiPlusID, phi0ID, psiPlusID, 'other_plus'));
[rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_be_basis('other', phi0ID, psiMinusID, phi0ID, psiMinusID, 'other_minus'));

rawCounts = count_blocks(rawBasis);
fprintf('\nRaw B_three^[0] union B_other^[0] before coarsening:\n');
fprintf('  M=%d | Mzero=%d | Mone=%d | Mtwo_ud=%d | Mtwo_uu=%d | Mthree=%d | Mother=%d | oneFuncs=%d\n', ...
    rawCounts.total, rawCounts.zero, rawCounts.one_total, rawCounts.two_ud, rawCounts.two_uu, rawCounts.three, rawCounts.other, numel(oneFuncs));
fprintf('  Be Table 5.10 selected kappa=1 target after coarsening: M=1417, Mone=524, Mud=562, Muu=264, Mthree=64, Mother=2.\n');

% Pure combinatorial check for the raw construction.  These are not Table
% 5.10 coarsened counts; they are the a-priori B_three^[0] union B_other^[0]
% counts before coefficient-threshold coarsening.
M2pairs = rawCounts.three / 4;
MuuPairsPerSpin = rawCounts.two_uu / 2;
fprintf('  Raw-count diagnostic: Mone=4*Mb=%d, Mud=4*Mb=%d, Muu=2*%d=%d, Mthree=4*%d=%d.\n', ...
    4*Mb, 4*Mb, MuuPairsPerSpin, rawCounts.two_uu, M2pairs, rawCounts.three);
fprintf('  Interpretation: counts are larger than Li because Be has four one-particle replacement slots, four up-down pairs, two same-spin pairs, and four triples.\n');

if dryRunAfterBasisBuild
    basisPreview = struct('hf',hf,'params',struct('Zcharge',Zcharge,'sigma0',sigma0,'cscale',cscale,'L0',L0, ...
        'ZsetMode',ZsetMode,'centerConvention',centerConvention,'neighborMode',neighborMode), ...
        'oneFuncs',oneFuncs,'basis',rawBasis,'counts',rawCounts);
    save('JH_BeGround_BthreeOther0_basis_preview_v2fast.mat','basisPreview','-v7.3');
    fprintf('dryRunAfterBasisBuild=true: saved basis preview and stopped before assembly.\n');
    return;
end

%% ============================================================
% 3. Assemble raw matrices
% =============================================================
oneCache = reset_one_cache(Zcharge, Rnuc);
oneCache = update_one_cache(oneCache, oneFuncs, Zcharge, Rnuc);

tAsmRaw = tic;
[Sraw,Hraw,rawInfo] = assemble_be_matrices(rawBasis, oneCache, oneFuncs, assemblyOpts);
tAsmRaw = toc(tAsmRaw);

fprintf('\nRaw assembly done: M=%d, nnz(S/H)=%.3e/%.3e, time=%.2fs\n', ...
    numel(rawBasis), nnz(Sraw), nnz(Hraw), tAsmRaw);

EzeroDirect = full(Hraw(1,1) / Sraw(1,1));
fprintf('Zero determinant check: H(1,1)/S(1,1)=%.15f | hf.E=%.15f | diff=%.3e\n', ...
    EzeroDirect, get_hf_energy(hf), EzeroDirect-get_hf_energy(hf));

[Er, cr, keptRaw, rawSolveInfo] = solve_ground_projected(Hraw, Sraw, mass_tol, verboseEig, E_ref_exact);
rawBE = compute_block_energy_contributions(cr, Hraw, rawBasis);
fprintf('Raw solve: E=%.15f | err(table exact)=%.3e | kept=%d | residual=%.3e\n', ...
    Er, abs(Er-E_ref_exact), keptRaw, rawSolveInfo.residual);
if Er < E_ref_exact - 0.5
    warning(['Raw energy is far below the physical variational diagnostic. ', ...
        'This indicates remaining linear-dependence or mapping errors. ', ...
        'Try increasing mass_tol and inspect raw block counts.']);
end
fprintf('Raw row energy: Ezero=%.6f Eone=%.6f Etwo_ud=%.6f Etwo_uu=%.6e Ethree=%.6e Eother=%.6e sum=%.15f\n', ...
    rawBE.row.zero, rawBE.row.one, rawBE.row.two_ud, rawBE.row.two_uu, rawBE.row.three, rawBE.row.other, rawBE.row.total);

part2 = struct();
part2.hfFile = hfFile; part2.hf = hf;
part2.params = struct('Zcharge',Zcharge,'Rnuc',Rnuc,'sigma0',sigma0,'cscale',cscale,'L0',L0, ...
    'D',D,'ZsetMode',ZsetMode,'centerConvention',centerConvention,'centerForNeighbors',centerForNeighbors, ...
    'neighborMode',neighborMode,'detScaleMode',detScaleMode,'E_ref_exact',E_ref_exact, ...
    'E_ref_HF_table',E_ref_HF_table,'E_ref_HF_limit',E_ref_HF_limit,'E_ref_tilde_table',E_ref_tilde_table, ...
    'E_table_kappa1',E_table_kappa1,'delta',delta);
part2.oneFuncs = oneFuncs;
part2.basis = rawBasis;
part2.S = Sraw; part2.H = Hraw;
part2.E_raw = Er; part2.coeff_raw = cr; part2.keptRaw = keptRaw;
part2.counts_raw = rawCounts; part2.blockEnergy_raw = rawBE;
part2.assemblyInfo = rawInfo; part2.assemblyTime = tAsmRaw;

rawFile = sprintf('JH_BeGround_BthreeOther0_raw_assembled_v3termFast_M%d.mat', rawCounts.total);
save(rawFile, 'part2', '-v7.3');
fprintf('Saved raw system: %s\n', rawFile);

%% ============================================================
% 4. Coarsening scan B(nu) = {|v_mu| >= 2^{-nu}}
% =============================================================
coefAbs = abs(cr(:));
records = struct('nu',{},'thr',{},'M',{},'Mzero',{},'Mone',{},'Mtwo_ud',{},'Mtwo_uu',{},'Mthree',{},'Mother',{}, ...
                 'E',{},'dE',{},'err',{},'keptDim',{},'idx',{});

fprintf('\nCoarsening scan B(nu) = {|v_mu| >= 2^{-nu}}:\n');
fprintf('   nu       threshold          M   Mzero   Mone  Mtwo_ud Mtwo_uu Mthree Mother        E(nu)              dE\n');

for inu = 1:numel(nuList)
    nu = nuList(inu); thr = 2^(-nu);
    idx = find(coefAbs >= thr);
    idx = unique([1; idx(:)], 'stable');

    Ssub = Sraw(idx,idx); Hsub = Hraw(idx,idx);
    [Esub, csub, keptDim] = solve_ground_projected(Hsub, Ssub, mass_tol, false, E_ref_exact); %#ok<ASGLU>
    dE = Esub - Er;
    cnt = count_blocks(rawBasis(idx));

    records(inu).nu=nu; records(inu).thr=thr; records(inu).M=numel(idx);
    records(inu).Mzero=cnt.zero; records(inu).Mone=cnt.one_total; records(inu).Mtwo_ud=cnt.two_ud;
    records(inu).Mtwo_uu=cnt.two_uu; records(inu).Mthree=cnt.three; records(inu).Mother=cnt.other;
    records(inu).E=Esub; records(inu).dE=dE; records(inu).err=abs(Esub-E_ref_exact); records(inu).keptDim=keptDim; records(inu).idx=idx;

    fprintf('%5d   %.6e   %6d   %5d  %5d  %7d %7d %6d %6d   %.15f   %.3e\n', ...
        nu, thr, numel(idx), cnt.zero, cnt.one_total, cnt.two_ud, cnt.two_uu, cnt.three, cnt.other, Esub, dE);
end

valid = find([records.dE] <= delta + 1e-12 & [records.dE] >= -1e-10);
if isempty(valid)
    warning('No B(nu) satisfies dE <= delta. Selecting the largest candidate.');
    [~,bestPos] = max([records.M]); selected = records(bestPos);
else
    [~,local] = min([records(valid).M]); selected = records(valid(local));
end

idxBest = selected.idx;
S0 = Sraw(idxBest,idxBest);
H0 = Hraw(idxBest,idxBest);
basis0 = rawBasis(idxBest);
selectedExactInfo = struct('used',false);
if exactReassembleSelected
    fprintf('\nExact reassembly of selected coarse Be system before final save: M0=%d ...\n', numel(basis0));
    optsExact = assemblyOpts;
    optsExact.screenERIByOverlap = false;
    optsExact.eriCoeffTol = exactSelectedEriCoeffTol;
    optsExact.rowBlockSize = exactSelectedRowBlockSize;
    optsExact.showRowProgress = true;
    tExactSel = tic;
    [S0,H0,selectedExactInfo] = assemble_be_matrices(basis0, oneCache, oneFuncs, optsExact);
    selectedExactInfo.used = true;
    selectedExactInfo.time = toc(tExactSel);
    fprintf('Exact selected reassembly done: time=%.2fs, nnz(S/H)=%.3e/%.3e\n', ...
        selectedExactInfo.time, nnz(S0), nnz(H0));
end
[E0, coeff0, kept0, solveInfo0] = solve_ground_projected(H0, S0, mass_tol, verboseEig, E_ref_exact);
counts0 = count_blocks(basis0);
ener0 = compute_block_energy_contributions(coeff0, H0, basis0);

fprintf('\n==================== SELECTED Be COARSE SYSTEM ====================\n');
fprintf('nu_selected = %d\n', selected.nu);
fprintf('threshold   = %.6e\n', selected.thr);
fprintf('M0          = %d\n', numel(idxBest));
fprintf('M_zero      = %d\n', counts0.zero);
fprintf('M_one       = %d\n', counts0.one_total);
fprintf('M_two_ud    = %d\n', counts0.two_ud);
fprintf('M_two_uu    = %d\n', counts0.two_uu);
fprintf('M_three     = %d\n', counts0.three);
fprintf('M_other     = %d\n', counts0.other);
fprintf('E0          = %.15f\n', E0);
fprintf('E_raw       = %.15f\n', Er);
fprintf('E0-E_raw    = %.6e hartree\n', E0 - Er);
fprintf('delta       = %.6e hartree\n', delta);
fprintf('err to J.H. exact diagnostic %.5f = %.6e\n', E_ref_exact, abs(E0-E_ref_exact));
fprintf('keptDim     = %d / %d | residual=%.3e\n', kept0, numel(idxBest), solveInfo0.residual);
fprintf('\nTable 5.10 row/action energy decomposition:\n');
fprintf('E_zero(row)   = %.15f\n', ener0.row.zero);
fprintf('E_one(row)    = %.15f\n', ener0.row.one);
fprintf('E_two_ud(row) = %.15f\n', ener0.row.two_ud);
fprintf('E_two_uu(row) = %.15e\n', ener0.row.two_uu);
fprintf('E_three(row)  = %.15e\n', ener0.row.three);
fprintf('E_other(row)  = %.15e\n', ener0.row.other);
fprintf('sum(row)      = %.15f\n', ener0.row.total);
fprintf('E0            = %.15f\n', E0);
fprintf('\nTable 5.10 kappa=1 target: M=1417, E=-14.644213, Mone=524, Mtwo_ud=562, Mtwo_uu=264, Mthree=64, Mother=2.\n');
fprintf('Target deviations: dM=%+d, dMone=%+d, dMtwo_ud=%+d, dMtwo_uu=%+d, dMthree=%+d, dMother=%+d, dE=%+.3e\n', ...
    counts0.total-1417, counts0.one_total-524, counts0.two_ud-562, counts0.two_uu-264, counts0.three-64, counts0.other-2, E0-E_table_kappa1);

fprintf('\nNeighborhood around selected candidate:\n');
for k = max(1, selected.nu-3):min(numel(records)-1, selected.nu+3)
    r = records(k+1); marker = ' ';
    if r.nu == selected.nu, marker = '*'; end
    fprintf('%s nu=%2d | M=%5d | Mone=%4d | Mud=%4d | Muu=%4d | M3=%4d | Mo=%2d | E=%.15f | dE=%.3e\n', ...
        marker, r.nu, r.M, r.Mone, r.Mtwo_ud, r.Mtwo_uu, r.Mthree, r.Mother, r.E, r.dE);
end

part3 = struct();
part3.inputFile = rawFile;
part3.hf = hf;
part3.params = part2.params;
part3.delta = delta;
part3.hartree_per_kcalmol = hartree_per_kcalmol;
part3.records = records;
part3.selected = selected;
part3.idxBest = idxBest;
part3.nuBest = selected.nu;
part3.oneFuncs = oneFuncs;
part3.basis0 = basis0;
part3.S0 = S0; part3.H0 = H0;
part3.selectedExactInfo = selectedExactInfo;
part3.E0 = E0; part3.coeff0 = coeff0; part3.kept0 = kept0; part3.solveInfo0 = solveInfo0;
part3.counts0 = counts0; part3.blockEnergy0 = ener0;
part3.rawSummary = struct('Mraw',rawCounts.total,'Eraw',Er,'countsRaw',rawCounts,'assemblyTime',tAsmRaw);

outFile = sprintf('JH_BeGround_BthreeOther0_coarsened_v2fast_M%d_nu%d.mat', counts0.total, selected.nu);
save(outFile, 'part3', '-v7.3');
fprintf('\nSaved coarsened system: %s\n', outFile);

Tcsv = struct2table(rmfield(records,'idx'));
writetable(Tcsv, 'JH_BeGround_BthreeOther0_coarsening_scan_v3termFast.csv');
fprintf('Saved coarsening scan CSV: JH_BeGround_BthreeOther0_coarsening_scan_v3termFast.csv\n');

%% ========================================================================
% Local functions
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
        B(end+1) = b; %#ok<AGROW>
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
        frame(end+1)=make_phi(sigma0,cscale,0,js(n,:),D); %#ok<AGROW>
    end
    for l=0:(L-2)
        for iz=1:size(Zset,1)
            z=Zset(iz,:); j=-z;
            frame(end+1)=make_psi(sigma0,cscale,l,j,z,D,centerConvention); %#ok<AGROW>
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
                neigh(end+1)=make_phi_at_center(sigma0,cscale,0,center,D); %#ok<AGROW>
            end
        case 'psi'
            l=f.level;
            for k=1:size(Z0,1)
                center = target + (1/(cscale*2^(l+1)))*Z0(k,:);
                cand = make_psi_at_center(sigma0,cscale,l,center,D,ZsetMode,centerConvention);
                if ~isempty(cand), neigh(end+1)=cand; end %#ok<AGROW>
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
    rowBlock = opts.rowBlockSize; blocks = 1:rowBlock:M;
    S = sparse(M,M); H = sparse(M,M); t0=tic;
    if strcmpi(opts.mode,'cpu_parfor_sparse') && opts.startPool
        try
            pool = gcp('nocreate'); if isempty(pool), parpool; end %#ok<NASGU>
        catch ME
            warning('Could not start parallel pool, continuing serial/parfor fallback: %s', ME.message);
        end
    end
    for ib=1:numel(blocks)
        rows = blocks(ib):min(M,blocks(ib)+rowBlock-1);
        [Sr,Hr] = assemble_be_sparse_rows(B,rows,S1,H1,td,normInv,opts);
        S(rows,:) = S(rows,:) + Sr;
        H(rows,:) = H(rows,:) + Hr;
        if opts.showRowProgress
            pe = get_opt(opts,'progressEveryBlocks',1);
            if mod(ib,pe)==0 || ib==numel(blocks)
                fprintf('    Be sparse rows %d-%d / %d inserted, nnz(S/H)=%.3e/%.3e, elapsed %.1fs\n', ...
                    rows(1), rows(end), M, nnz(S), nnz(H), toc(t0));
            end
        end
    end
    if isfield(opts,'upperTriangle') && opts.upperTriangle
        S = S + triu(S,1).'; H = H + triu(H,1).';
    end
    S = 0.5*(S+S'); H = 0.5*(H+H');
    info = struct('nnzS',nnz(S),'nnzH',nnz(H),'M',M,'upperTriangle',opts.upperTriangle);
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

function [Srows,Hrows] = assemble_be_sparse_rows(B,rowIdx,S1,H1,td,normInv,opts)
    nr=numel(rowIdx); M=numel(B);
    avec=[B.a]; bvec=[B.b]; cvec=[B.c]; dvec=[B.d];
    switch lower(opts.mode)
        case 'cpu_parfor_sparse'
            IScell=cell(nr,1); JScell=cell(nr,1); VScell=cell(nr,1);
            IHcell=cell(nr,1); JHcell=cell(nr,1); VHcell=cell(nr,1);
            parfor ir=1:nr
                [js,vs,jh,vh]=assemble_one_be_row(B,rowIdx(ir),avec,bvec,cvec,dvec,S1,H1,td,normInv,opts);
                IScell{ir}=ir*ones(numel(js),1); JScell{ir}=js(:); VScell{ir}=vs(:);
                IHcell{ir}=ir*ones(numel(jh),1); JHcell{ir}=jh(:); VHcell{ir}=vh(:);
            end
            IS=vertcat(IScell{:}); JS=vertcat(JScell{:}); VS=vertcat(VScell{:});
            IH=vertcat(IHcell{:}); JH=vertcat(JHcell{:}); VH=vertcat(VHcell{:});
        otherwise
            IS=[];JS=[];VS=[];IH=[];JH=[];VH=[];
            for ir=1:nr
                [js,vs,jh,vh]=assemble_one_be_row(B,rowIdx(ir),avec,bvec,cvec,dvec,S1,H1,td,normInv,opts);
                IS=[IS; ir*ones(numel(js),1)]; JS=[JS; js(:)]; VS=[VS; vs(:)]; %#ok<AGROW>
                IH=[IH; ir*ones(numel(jh),1)]; JH=[JH; jh(:)]; VH=[VH; vh(:)]; %#ok<AGROW>
            end
    end
    Srows=sparse(IS,JS,VS,nr,M); Hrows=sparse(IH,JH,VH,nr,M);
end

function [js,vs,jh,vh] = assemble_one_be_row(B,row,avec,bvec,cvec,dvec,S1,H1,td,normInv,opts)
    a=B(row).a; b=B(row).b; c=B(row).c; d0=B(row).d;
    M = numel(avec);
    e=avec; f=bvec; g=cvec; h=dvec;
    if isfield(opts,'upperTriangle') && opts.upperTriangle
        upperMask = (1:M) >= row;
    else
        upperMask = true(1,M);
    end

    Sae=S1(a,e); Saf=S1(a,f); Sbe=S1(b,e); Sbf=S1(b,f);
    Hae=H1(a,e); Haf=H1(a,f); Hbe=H1(b,e); Hbf=H1(b,f);
    Scg=S1(c,g); Sch=S1(c,h); Sdg=S1(d0,g); Sdh=S1(d0,h);
    Hcg=H1(c,g); Hch=H1(c,h); Hdg=H1(d0,g); Hdh=H1(d0,h);

    detA = Sae.*Sbf - Saf.*Sbe;
    detB = Scg.*Sdh - Sch.*Sdg;
    Sraw = detA .* detB;

    HoneA = (Hae.*Sbf + Sae.*Hbf - Haf.*Sbe - Saf.*Hbe) .* detB;
    HoneB = detA .* (Hcg.*Sdh + Scg.*Hdh - Hch.*Sdg - Sch.*Hdg);
    Hcheap = HoneA + HoneB;

    doScreen = isfield(opts,'screenERIByOverlap') && opts.screenERIByOverlap;
    if doScreen
        tolSpre = get_opt(opts,'screenTolSForERI',1e-12);
        tolHpre = get_opt(opts,'screenTolHCheapForERI',1e-10);
        preMask = upperMask & (abs(Sraw) > tolSpre | abs(Hcheap) > tolHpre);
        if isfield(opts,'forceDiagonal') && opts.forceDiagonal && row <= M
            preMask(row) = true;
        end
        idx = find(preMask);
    else
        idx = find(upperMask);
    end

    Hraw = Hcheap;
    if ~isempty(idx)
        ei=e(idx); fi=f(idx); gi=g(idx); hi=h(idx);
        detA_i=detA(idx); detB_i=detB(idx);

        coeffTol = get_opt(opts,'eriCoeffTol',0);
        nidx = numel(idx);
        Vaa_i = zeros(1,nidx); Vbb_i = zeros(1,nidx);

        maskVaa = true(1,nidx);
        maskVbb = true(1,nidx);
        if coeffTol > 0
            maskVaa = abs(detB_i) > coeffTol;
            maskVbb = abs(detA_i) > coeffTol;
        end
        if any(maskVaa)
            Vaa_i(maskVaa) = (eri_row_vectorized_cpu(td,a,ei(maskVaa),b,fi(maskVaa)) ...
                - eri_row_vectorized_cpu(td,a,fi(maskVaa),b,ei(maskVaa))) .* detB_i(maskVaa);
        end
        if any(maskVbb)
            Vbb_i(maskVbb) = detA_i(maskVbb) .* ...
                (eri_row_vectorized_cpu(td,c,gi(maskVbb),d0,hi(maskVbb)) ...
                - eri_row_vectorized_cpu(td,c,hi(maskVbb),d0,gi(maskVbb)));
        end

        % Cofactors of the alpha and beta 2x2 overlap matrices.
        CA11=Sbf(idx); CA12=-Sbe(idx); CA21=-Saf(idx); CA22=Sae(idx);
        CB11=Sdh(idx); CB12=-Sdg(idx); CB21=-Sch(idx); CB22=Scg(idx);

        Vab_i = zeros(1,nidx);
        Aleft = [a b];
        Bleft = [c d0];
        % Terms for alpha row i / col j and beta row k / col l.
        CAmat = {CA11, CA12; CA21, CA22};
        CBmat = {CB11, CB12; CB21, CB22};
        Aright = {ei, fi}; Bright = {gi, hi};
        for ia = 1:2
            for ja = 1:2
                caij = CAmat{ia,ja};
                for ib = 1:2
                    for jb = 1:2
                        cbij = CBmat{ib,jb};
                        pref = caij .* cbij;
                        termMask = true(1,nidx);
                        if coeffTol > 0
                            termMask = abs(pref) > coeffTol;
                        end
                        if any(termMask)
                            Vab_i(termMask) = Vab_i(termMask) + pref(termMask) .* ...
                                eri_row_vectorized_cpu(td,Aleft(ia),Aright{ja}(termMask),Bleft(ib),Bright{jb}(termMask));
                        end
                    end
                end
            end
        end
        Hraw(idx) = Hraw(idx) + Vaa_i + Vbb_i + Vab_i;
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
    % Compact term data with per-function nTerms.  This is crucial for Be:
    % HF orbitals have Q=12 terms, but every frame function has only 1 term.
    % The old v2 ERI routine looped over maxTerms^4 even for frame-frame ERIs.
    M=numel(oneFuncs); maxT=0; nTerms=zeros(M,1);
    for i=1:M
        nTerms(i)=numel(oneFuncs(i).terms);
        maxT=max(maxT,nTerms(i));
    end
    coef=zeros(M,maxT); alpha=zeros(M,maxT); cx=zeros(M,maxT); cy=zeros(M,maxT); cz=zeros(M,maxT);
    for i=1:M
        for t=1:nTerms(i)
            term=oneFuncs(i).terms(t);
            coef(i,t)=term.coef; alpha(i,t)=term.alpha;
            cx(i,t)=term.center(1); cy(i,t)=term.center(2); cz(i,t)=term.center(3);
        end
    end
    td=struct('coef',coef,'alpha',alpha,'cx',cx,'cy',cy,'cz',cz, ...
              'maxTerms',maxT,'nTerms',nTerms);
end

function vrow = eri_row_vectorized_cpu(td, ia, ibVec, ic, idVec)
    % Vectorized contracted ERI row with per-function term counts.
    %
    % This removes the major redundancy of v1/v2.  In Be coarsening, most
    % one-particle functions are one-term frame Gaussians, while only the HF
    % reference orbitals have many terms.  Looping over td.maxTerms for every
    % function costs roughly Q^4 even for frame-frame-frame-frame integrals.
    % Here each fixed/column function only contributes through its true nTerms.
    nc=numel(ibVec); vrow=zeros(1,nc);
    if nc==0, return; end

    nA = td.nTerms(ia); nC = td.nTerms(ic);
    nBv = reshape(td.nTerms(ibVec),1,[]);
    nDv = reshape(td.nTerms(idVec),1,[]);
    maxTB = max(nBv); maxTD = max(nDv);

    for ta=1:nA
        ca=td.coef(ia,ta); if ca==0, continue; end
        aA=td.alpha(ia,ta); Ax=td.cx(ia,ta); Ay=td.cy(ia,ta); Az=td.cz(ia,ta);
        for tc=1:nC
            cc=td.coef(ic,tc); if cc==0, continue; end
            aC=td.alpha(ic,tc); Cx=td.cx(ic,tc); Cy=td.cy(ic,tc); Cz=td.cz(ic,tc);
            coefAC=ca*cc;

            for tb=1:maxTB
                activeB = nBv >= tb;
                if ~any(activeB), continue; end
                cb=td.coef(ibVec,tb).';
                activeB = activeB & (cb~=0);
                if ~any(activeB), continue; end
                aB_all=td.alpha(ibVec,tb).'; Bx_all=td.cx(ibVec,tb).'; By_all=td.cy(ibVec,tb).'; Bz_all=td.cz(ibVec,tb).';

                for td2=1:maxTD
                    activeD = nDv >= td2;
                    mask=activeB & activeD;
                    if ~any(mask), continue; end
                    cd=td.coef(idVec,td2).';
                    mask=mask & (cd~=0);
                    if ~any(mask), continue; end

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
    for k=1:numel(fields), row.(fields{k})=0; diagv.(fields{k})=0; end %#ok<AGROW>
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
