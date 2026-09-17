%% JH_LiGround_part2_coarsening_v3_canonicalNg.m
% Li ground state, J.H. Table 5.9 style: raw B_three^[0] construction + coarsening.
% v3 canonical-Ng:
%   1) Fixes the most important mapping bug in v1/v2: Ng(psi) must create
%      one nearest-neighbor frame function per neighboring center C(phi)+h*z0.
%      The previous version enumerated all z-directions for every neighbor
%      center and therefore over-generated many near-duplicate same-spin/three
%      functions.  This caused Mraw~8205 and an ill-conditioned raw problem.
%   2) Keeps the J.H. construction of B_three^[0] in Eq. (4.52)-(4.56), but
%      uses a canonical representative of a wavelet-like Gaussian at a given
%      center/level.
%   3) Uses exact Coulomb assembly by default after the basis-size reduction.
%      Screening can still be enabled for exploratory tests.
%
% Input:
%   JH_LiGround_HF_Q9_orbitals.mat
%
% Output:
%   JH_LiGround_Bthree0_raw_assembled_M*.mat
%   JH_LiGround_Bthree0_coarsened_M*_nu*.mat
%
% This script intentionally stops after coarsening.  The adaptive main program
% should load the coarsened file and continue Algorithm 4.1 without rebuilding
% the raw a-priori system.

clear; clc; close all;
format long e;

%% ============================================================
% 0. Parameters from J.H. Sec. 5.3 and general Sec. 4.4
% =============================================================
hfFile = 'JH_LiGround_HF_Q9_orbitals.mat';
if ~exist(hfFile,'file')
    error('Cannot find %s. Run Li HF first.', hfFile);
end
load(hfFile, 'hf');

Zcharge = 3;
if isfield(hf,'Z'), Zcharge = hf.Z; end
Rnuc = [0 0 0];
D = 3;

% J.H. Sec. 5.3 Li ground state: c=1, L=6, sigma=1.
sigma0 = 1;
cscale = 1;
L0 = 6;

ZsetMode = 'signed';             % reproduces 27 + 26*(L-1)
centerConvention = 'eq428';      % same as the He/He-triplet code path
centerForNeighbors = 'actual';
neighborMode = 'cross';          % Z'={z: |z|_1<=1} in Eq. (4.54)

% Coarsening follows J.H.: basis set is L2-normalized, then threshold raw coeffs.
detScaleMode = 'l2_normalized';

% J.H. Sec. 5.3 / Table 5.13 diagnostics.
E_ref_exact = -7.47806;
E_ref_HF_table = -7.43271;
E_ref_tilde_table = -7.47702;

% J.H. delta = 1/100 kcal/mol.
hartree_per_kcalmol = 1/627.5094740631;
delta = 0.01 * hartree_per_kcalmol;

mass_tol = 1e-10;
verboseEig = false;
nuList = 0:80;

% Debug switch: set true to stop after raw basis construction and inspect counts
% before spending time on H/S assembly.
dryRunAfterBasisBuild = false;

assemblyOpts = struct();
assemblyOpts.mode = 'cpu_parfor_sparse';    % 'serial_sparse' or 'cpu_parfor_sparse'
assemblyOpts.rowBlockSize = 256;             % canonical Ng makes raw much smaller; 16/24/32 are all OK
assemblyOpts.startPool = true;
assemblyOpts.dropTolS = 1e-13;             % use stricter sparse drop; set 0 for a final exact check
assemblyOpts.dropTolH = 1e-12;             % use stricter sparse drop; set 0 for a final exact check
assemblyOpts.screenERIByOverlap = false;   % default exact Coulomb assembly; enable only for exploratory speed tests
assemblyOpts.screenTolSForERI = 1e-13;
assemblyOpts.screenTolHCheapForERI = 1e-12;
assemblyOpts.forceDiagonal = true;         % always keep diagonal entries
assemblyOpts.showRowProgress = true;
assemblyOpts.upperTriangle = true;
assemblyOpts.detScaleMode = detScaleMode;

fprintf('\n============================================================\n');
fprintf('J.H. Li ground-state coarsening, partially antisymmetric 2-alpha + 1-beta basis\n');
fprintf('HF file: %s\n', hfFile);
fprintf('Parameters: Z=%g, sigma=%g, c=%g, L=%d, neighbor=%s, detScale=%s\n', ...
    Zcharge, sigma0, cscale, L0, neighborMode, detScaleMode);
fprintf('Expected Table 5.9 kappa=1: M=610, E=-7.471645, Mone=253, Mtwo_ud=190, Mtwo_uu=108, Mthree=58.\n');
fprintf('Coarsening delta = %.6e hartree = 1/100 kcal/mol.\n', delta);
fprintf('Assembly: canonical Ng, dropTolS=%.1e, dropTolH=%.1e, screenERI=%d.\n', ...
    assemblyOpts.dropTolS, assemblyOpts.dropTolH, assemblyOpts.screenERIByOverlap);
fprintf('============================================================\n\n');

%% ============================================================
% 1. Register HF orbitals and one-particle initial frame b^{(L)}_{sigma,c}
% =============================================================
oneFuncs = repmat(empty_onefunc(), 0, 1);
oneKeyMap = containers.Map('KeyType','char','ValueType','double');

% HF orbital order: 1=alpha_core, 2=alpha_valence, 3=beta_core.
for p = 1:3
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
firstFrameID = 4;
lastFrameID = numel(oneFuncs);
frameIDs = firstFrameID:lastFrameID;
Mb = numel(frameIDs);
fprintf('One-particle frame b^(%d): Mb=%d. Expected signed count 27+26*(L-1)=%d.\n', ...
    L0, Mb, 27+26*(L0-1));

% b^(2) used only for the initial three-particle block Eq. (4.56), independent of L.
frame2 = make_initial_frame(sigma0, cscale, 2, D, ZsetMode, centerConvention);
frame2IDs = zeros(1,numel(frame2));
for i = 1:numel(frame2)
    [oneFuncs, oneKeyMap, frame2IDs(i)] = register_onefunc(oneFuncs, oneKeyMap, frame2(i));
end
fprintf('Three-block frame b^(2): M2=%d.\n', numel(frame2IDs));

%% ============================================================
% 2. Build raw B_three^[0] according to Eq. (4.52)-(4.56)
% =============================================================
rawBasis = repmat(empty_libasis(), 0, 1);
basisMap = containers.Map('KeyType','char','ValueType','double');

refA1 = 1; refA2 = 2; refB = 3;

% V_zero: HF determinant A_up[psi1_alpha,psi2_alpha] * psi_beta.
[rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_li_basis('zero', refA1, refA2, refB, 'zero'));

% V_one: replace each occupied one-particle function by b^(L).
for fid = frameIDs
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_li_basis('one', fid, refA2, refB, 'one_a1'));
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_li_basis('one', refA1, fid, refB, 'one_a2'));
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_li_basis('one', refA1, refA2, fid, 'one_b'));
end

% V_two^{up down}: replace one alpha and the beta by the same phi in b^(L), Eq. (4.53).
for fid = frameIDs
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_li_basis('two_ud', fid, refA2, fid, 'two_a1b'));
    [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_li_basis('two_ud', refA1, fid, fid, 'two_a2b'));
end

% V_two^{up up}: replace both alpha functions by neighboring frame functions, Eq. (4.55).
for fid = frameIDs
    neigh = neighbors_Ng_onefunc(oneFuncs(fid), sigma0, cscale, D, ZsetMode, ...
        centerConvention, centerForNeighbors, neighborMode);
    for n = 1:numel(neigh)
        [oneFuncs, oneKeyMap, nid] = register_onefunc(oneFuncs, oneKeyMap, neigh(n));
        if nid == fid, continue; end
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_li_basis('two_uu', nid, fid, refB, 'two_aa_1'));
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_li_basis('two_uu', fid, nid, refB, 'two_aa_2'));
    end
end

% V_three: for Li N_up=2,N_down=1, replace the two alpha and the beta.
% Eq. (4.56): phi in b^(2), phi_tilde in Ng(phi), independent of L.
for fid = frame2IDs
    neigh = neighbors_Ng_onefunc(oneFuncs(fid), sigma0, cscale, D, ZsetMode, ...
        centerConvention, centerForNeighbors, neighborMode);
    for n = 1:numel(neigh)
        [oneFuncs, oneKeyMap, nid] = register_onefunc(oneFuncs, oneKeyMap, neigh(n));
        if nid == fid, continue; end
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_li_basis('three', nid, fid, fid, 'three_1'));
        [rawBasis, basisMap] = append_basis(rawBasis, basisMap, make_li_basis('three', fid, nid, fid, 'three_2'));
    end
end

rawCounts = count_blocks(rawBasis);
fprintf('\nRaw B_three^[0] before coarsening:\n');
fprintf('  M=%d | Mzero=%d | Mone=%d | Mtwo_ud=%d | Mtwo_uu=%d | Mthree=%d | oneFuncs=%d\n', ...
    rawCounts.total, rawCounts.zero, rawCounts.one_total, rawCounts.two_ud, rawCounts.two_uu, rawCounts.three, numel(oneFuncs));
fprintf('  Canonical Ng sanity: Mtwo_uu and Mthree should be far smaller than the v2 over-generated values 5973 and 1446.\n');
if dryRunAfterBasisBuild
    basisPreview = struct('hf',hf,'params',struct('Zcharge',Zcharge,'sigma0',sigma0,'cscale',cscale,'L0',L0, ...
        'ZsetMode',ZsetMode,'centerConvention',centerConvention,'neighborMode',neighborMode), ...
        'oneFuncs',oneFuncs,'basis',rawBasis,'counts',rawCounts);
    save('JH_LiGround_Bthree0_basis_preview_v3canon.mat','basisPreview','-v7.3');
    fprintf('dryRunAfterBasisBuild=true: saved basis preview and stopped before assembly.\n');
    return;
end

%% ============================================================
% 3. Assemble raw matrices
% =============================================================
oneCache = reset_one_cache(Zcharge, Rnuc);
oneCache = update_one_cache(oneCache, oneFuncs, Zcharge, Rnuc);

tAsmRaw = tic;
[Sraw,Hraw,rawInfo] = assemble_li_matrices(rawBasis, oneCache, oneFuncs, assemblyOpts);
tAsmRaw = toc(tAsmRaw);

fprintf('\nRaw assembly done: M=%d, nnz(S/H)=%.3e/%.3e, time=%.2fs\n', ...
    numel(rawBasis), nnz(Sraw), nnz(Hraw), tAsmRaw);

EzeroDirect = full(Hraw(1,1) / Sraw(1,1));
fprintf('Zero determinant check: H(1,1)/S(1,1)=%.15f | hf.E=%.15f | diff=%.3e\n', ...
    EzeroDirect, hf.E, EzeroDirect-hf.E);

[Er, cr, keptRaw, rawSolveInfo] = solve_ground_projected(Hraw, Sraw, mass_tol, verboseEig, E_ref_exact);
rawBE = compute_block_energy_contributions(cr, Hraw, rawBasis);
fprintf('Raw solve: E=%.15f | err(table exact)=%.3e | kept=%d | residual=%.3e\n', ...
    Er, abs(Er-E_ref_exact), keptRaw, rawSolveInfo.residual);
if Er < E_ref_exact - 0.5
    warning(['Raw energy is far below the physical variational bound. ', ...
        'This indicates remaining linear-dependence or mapping errors. ', ...
        'Try increasing mass_tol to 1e-8 and check raw block counts.']);
end
fprintf('Raw row energy: Ezero=%.6f Eone=%.6f Etwo_ud=%.6f Etwo_uu=%.6e Ethree=%.6e sum=%.15f\n', ...
    rawBE.row.zero, rawBE.row.one, rawBE.row.two_ud, rawBE.row.two_uu, rawBE.row.three, rawBE.row.total);

part2 = struct();
part2.hfFile = hfFile; part2.hf = hf;
part2.params = struct('Zcharge',Zcharge,'Rnuc',Rnuc,'sigma0',sigma0,'cscale',cscale,'L0',L0, ...
    'D',D,'ZsetMode',ZsetMode,'centerConvention',centerConvention,'centerForNeighbors',centerForNeighbors, ...
    'neighborMode',neighborMode,'detScaleMode',detScaleMode,'E_ref_exact',E_ref_exact, ...
    'E_ref_HF_table',E_ref_HF_table,'E_ref_tilde_table',E_ref_tilde_table,'delta',delta);
part2.oneFuncs = oneFuncs;
part2.basis = rawBasis;
part2.S = Sraw; part2.H = Hraw;
part2.E_raw = Er; part2.coeff_raw = cr; part2.keptRaw = keptRaw;
part2.counts_raw = rawCounts; part2.blockEnergy_raw = rawBE;
part2.assemblyInfo = rawInfo; part2.assemblyTime = tAsmRaw;

rawFile = sprintf('JH_LiGround_Bthree0_raw_assembled_v3canon_M%d.mat', rawCounts.total);
save(rawFile, 'part2', '-v7.3');
fprintf('Saved raw system: %s\n', rawFile);

%% ============================================================
% 4. Coarsening scan B(nu) = {|v_mu| >= 2^{-nu}}
% =============================================================
coefAbs = abs(cr(:));
records = struct('nu',{},'thr',{},'M',{},'Mzero',{},'Mone',{},'Mtwo_ud',{},'Mtwo_uu',{},'Mthree',{}, ...
                 'E',{},'dE',{},'err',{},'keptDim',{},'idx',{});

fprintf('\nCoarsening scan B(nu) = {|v_mu| >= 2^{-nu}}:\n');
fprintf('   nu       threshold          M   Mzero   Mone  Mtwo_ud Mtwo_uu Mthree        E(nu)              dE\n');

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
    records(inu).Mtwo_uu=cnt.two_uu; records(inu).Mthree=cnt.three;
    records(inu).E=Esub; records(inu).dE=dE; records(inu).err=abs(Esub-E_ref_exact); records(inu).keptDim=keptDim; records(inu).idx=idx;

    fprintf('%5d   %.6e   %6d   %5d  %5d  %7d %7d %6d   %.15f   %.3e\n', ...
        nu, thr, numel(idx), cnt.zero, cnt.one_total, cnt.two_ud, cnt.two_uu, cnt.three, Esub, dE);
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
[E0, coeff0, kept0, solveInfo0] = solve_ground_projected(H0, S0, mass_tol, verboseEig, E_ref_exact);
counts0 = count_blocks(basis0);
ener0 = compute_block_energy_contributions(coeff0, H0, basis0);

fprintf('\n==================== SELECTED Li COARSE SYSTEM ====================\n');
fprintf('nu_selected = %d\n', selected.nu);
fprintf('threshold   = %.6e\n', selected.thr);
fprintf('M0          = %d\n', numel(idxBest));
fprintf('M_zero      = %d\n', counts0.zero);
fprintf('M_one       = %d\n', counts0.one_total);
fprintf('M_two_ud    = %d\n', counts0.two_ud);
fprintf('M_two_uu    = %d\n', counts0.two_uu);
fprintf('M_three     = %d\n', counts0.three);
fprintf('E0          = %.15f\n', E0);
fprintf('E_raw       = %.15f\n', Er);
fprintf('E0-E_raw    = %.6e hartree\n', E0 - Er);
fprintf('delta       = %.6e hartree\n', delta);
fprintf('err to J.H. exact diagnostic %.5f = %.6e\n', E_ref_exact, abs(E0-E_ref_exact));
fprintf('keptDim     = %d / %d | residual=%.3e\n', kept0, numel(idxBest), solveInfo0.residual);
fprintf('\nTable 5.9 row/action energy decomposition:\n');
fprintf('E_zero(row)   = %.15f\n', ener0.row.zero);
fprintf('E_one(row)    = %.15f\n', ener0.row.one);
fprintf('E_two_ud(row) = %.15f\n', ener0.row.two_ud);
fprintf('E_two_uu(row) = %.15e\n', ener0.row.two_uu);
fprintf('E_three(row)  = %.15e\n', ener0.row.three);
fprintf('sum(row)      = %.15f\n', ener0.row.total);
fprintf('E0            = %.15f\n', E0);
fprintf('\nTable 5.9 kappa=1 target: M=610, E=-7.471645, Mone=253, Mtwo_ud=190, Mtwo_uu=108, Mthree=58.\n');
fprintf('Target deviations: dM=%+d, dMone=%+d, dMtwo_ud=%+d, dMtwo_uu=%+d, dMthree=%+d, dE=%+.3e\n', ...
    counts0.total-610, counts0.one_total-253, counts0.two_ud-190, counts0.two_uu-108, counts0.three-58, E0-(-7.471645));

fprintf('\nNeighborhood around selected candidate:\n');
for k = max(1, selected.nu-3):min(numel(records)-1, selected.nu+3)
    r = records(k+1); marker = ' ';
    if r.nu == selected.nu, marker = '*'; end
    fprintf('%s nu=%2d | M=%5d | Mone=%4d | Mud=%4d | Muu=%4d | M3=%4d | E=%.15f | dE=%.3e\n', ...
        marker, r.nu, r.M, r.Mone, r.Mtwo_ud, r.Mtwo_uu, r.Mthree, r.E, r.dE);
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
part3.E0 = E0; part3.coeff0 = coeff0; part3.kept0 = kept0; part3.solveInfo0 = solveInfo0;
part3.counts0 = counts0; part3.blockEnergy0 = ener0;
part3.rawSummary = struct('Mraw',rawCounts.total,'Eraw',Er,'countsRaw',rawCounts,'assemblyTime',tAsmRaw);

outFile = sprintf('JH_LiGround_Bthree0_coarsened_v3canon_M%d_nu%d.mat', counts0.total, selected.nu);
save(outFile, 'part3', '-v7.3');
fprintf('\nSaved coarsened system: %s\n', outFile);

% A CSV summary for quick inspection.
Tcsv = struct2table(rmfield(records,'idx'));
writetable(Tcsv, 'JH_LiGround_Bthree0_coarsening_scan_v3canon.csv');
fprintf('Saved coarsening scan CSV: JH_LiGround_Bthree0_coarsening_scan_v3canon.csv\n');

%% ========================================================================
% Local functions
% ========================================================================
function f = empty_onefunc()
    f = struct('kind','','type','','level',0,'j',[0 0 0],'z',[0 0 0], ...
        'terms',struct('coef',{},'alpha',{},'center',{}),'key',string(''));
end

function b = empty_libasis()
    b = struct('block','','a',0,'b',0,'c',0,'tag','','key',string(''));
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
    if b.a == b.b, return; end       % alpha determinant vanishes
    key = char(b.key);
    if ~isKey(map,key)
        B(end+1) = b; %#ok<AGROW>
        map(key) = numel(B);
    end
end

function b = make_li_basis(block, ia, ib, ic, tag)
    b = empty_libasis();
    ab = sort([ia ib]);
    b.block = char(block); b.a = ab(1); b.b = ab(2); b.c = ic; b.tag = char(tag);
    % Keep the block label in the key.  This mirrors the He code and preserves
    % the intended particle-wise direct-sum diagnostics even if subspaces overlap.
    b.key = string(sprintf('%s_A%d_%d_B%d', char(block), b.a, b.b, b.c));
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
    % J.H. Eq. (4.54): Ng(phi) is defined by neighboring centers C(phi)+h*z0.
    % For a wavelet-like psi, the old implementation enumerated all z-directions
    % for every target center.  That is not the intended finite nearest-neighbor
    % set and over-generates many almost dependent functions.  Here we create a
    % single canonical frame representative for each target center.
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
    scale = cscale*2^l;
    j = round(center * scale);
    center2 = j / scale;
    f = make_phi(sigma0,cscale,l,j,D);
    % Keep the exact represented center in the key/terms to avoid drift.
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
    % Return one canonical psi_{l,j}^{[z]} with C(psi)=center.
    % For Eq. (4.28), center = (j + 0.5*z)/(c*2^l) under eq428.
    scale = cscale*2^l;
    m = round(2 * scale * center);  % m = 2*j + z, componentwise integer
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
    if all(z==0)
        % No wavelet-like psi with z=0 exists in b_{sigma,c}; this center is
        % not represented by a psi at this level.  Do not fabricate a nearby
        % center, since doing so was the source of over-generation and mapping
        % errors in the previous implementation.
        f = [];
        return;
    end
    f = make_psi(sigma0,cscale,l,round(j),round(z),D,centerConvention);
    % Force canonical exact center from represented j,z.
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

function [S,H,info] = assemble_li_matrices(B, oneCache, oneFuncs, opts)
    M = numel(B); S1=oneCache.S1; H1=oneCache.H1; td=build_term_data(oneFuncs);
    normInv = li_norm_inv(B,S1,opts);
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
    % Fast screened Li row assembly.
    % The cheap one-particle and overlap part is evaluated for all columns.
    % The expensive two-particle Coulomb terms are evaluated only on a screened
    % column subset.  This is the main speed-up for raw Li M~8k.
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
        if isfield(opts,'forceDiagonal') && opts.forceDiagonal && row <= M
            preMask(row) = true;
        end
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
