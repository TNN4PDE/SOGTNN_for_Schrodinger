%% JH_LiGround_direct_assemble_project_solve_from_checkpoint.m
% Direct matrix assembly + mass projection + lowest eigen solve for a saved Li adaptive basis.
%
% Purpose:
%   Load a saved adaptive checkpoint basis, e.g.
%       JH_LiGround_Table59_adaptive_checkpoint_nu12_M508_kappa10_M22630.mat
%   then assemble S/H on exactly this basis and solve the projected generalized
%   eigenvalue problem without any further adaptive growth.
%
% This script intentionally follows the timing / memory / precision reporting
% style of Adaptive_Li.m, but strips away all growth logic.

clear; clc; close all;
format long e;

%% ============================================================
% 0. User controls
% =============================================================
checkpointFile = 'JH_LiGround_Table59_adaptive_checkpoint_nu12_M508_kappa10_M22630.mat';
if ~exist(checkpointFile,'file')
    % Fallback: latest matching checkpoint in current folder.
    d = dir('JH_LiGround_Table59_adaptive_checkpoint_nu12_M508_kappa10_M*.mat');
    if isempty(d)
        d = dir('JH_LiGround_Table59_adaptive_checkpoint_*_M22630.mat');
    end
    if isempty(d)
        error('Checkpoint file not found. Put the M=22630 checkpoint in the current folder or edit checkpointFile.');
    end
    [~,ord] = max([d.datenum]);
    checkpointFile = d(ord).name;
end

% References / diagnostics for Li ground-state Table 5.9/5.13.
E_ref_exact = -7.47806;
E_ref_tilde_table = -7.47702;
E_ref_HF_table = -7.43271;

% Exact assembly controls.  Keep these aligned with Adaptive_Li.m unless you
% deliberately want a screened exploratory run.
assemblyOpts = struct();
assemblyOpts.mode = 'cpu_parfor_sparse';      % 'cpu_parfor_sparse' or 'cpu_serial_sparse'
assemblyOpts.rowBlockSize = 512;              % smaller = more progress prints and lower peak row memory
assemblyOpts.startPool = true;
assemblyOpts.dropTolS = 1e-13;
assemblyOpts.dropTolH = 1e-12;
assemblyOpts.screenERIByOverlap = false;      % false = exact-style assembly used for final validation
assemblyOpts.screenTolSForERI = 1e-13;
assemblyOpts.screenTolHCheapForERI = 1e-12;
assemblyOpts.forceDiagonal = true;
assemblyOpts.showRowProgress = true;
assemblyOpts.upperTriangle = true;
assemblyOpts.detScaleMode = 'l2_normalized';

% Solver controls.  This is the same dense mass-projection solver style as
% Adaptive_Li.m.  For M=22630 it may require very large RAM; the script prints
% an estimate before solving.
mass_tol = 1e-10;
verboseEig = true;
solverOpts = struct();
solverOpts.projectedDenseLimit = 25000;
solverOpts.rejectBelowExact = true;
solverOpts.energyLowerTol = 5e-4;
solverOpts.exactLowerBound = E_ref_exact;
solverOpts.allowDenseSolveDespiteMemoryRisk = true;  % set false to stop after memory precheck
solverOpts.symmetrizeFullMatrices = false;           % matrices from assembler are already symmetrized

% Output controls.
saveSparseMatrices = true;       % set false if the assembled S/H file is too large
saveDenseProjection = false;      % usually false; Hp/X are huge for M=22630
outputPrefix = 'JH_LiGround_Table59_direct_M22630';

%% ============================================================
% 1. Load checkpoint and normalize interface
% =============================================================
fprintf('\n============================================================\n');
fprintf('J.H. Li direct assembly + mass projection solve from checkpoint\n');
fprintf('Checkpoint: %s\n', checkpointFile);
fprintf('============================================================\n');

Sload = load(checkpointFile);
[state, stateName] = extract_checkpoint_state(Sload, checkpointFile);

basis = state.basis(:).';
oneFuncs = state.oneFuncs(:).';
params = state.params;
if isfield(state,'assemblyOpts') && isfield(state.assemblyOpts,'detScaleMode')
    assemblyOpts.detScaleMode = state.assemblyOpts.detScaleMode;
end

% Robust parameter fallback.
if isfield(params,'Zcharge'), Zcharge = params.Zcharge; elseif isfield(state,'hf') && isfield(state.hf,'Z'), Zcharge = state.hf.Z; else, Zcharge = 3; end
if isfield(params,'Rnuc'), Rnuc = params.Rnuc; else, Rnuc = [0 0 0]; end
if isfield(params,'sigma0'), sigma0 = params.sigma0; else, sigma0 = 1; end
if isfield(params,'cscale'), cscale = params.cscale; else, cscale = 1; end
if isfield(params,'L0'), L0 = params.L0; else, L0 = 6; end
if isfield(params,'neighborMode'), neighborMode = params.neighborMode; else, neighborMode = 'cross'; end
if isfield(params,'detScaleMode'), assemblyOpts.detScaleMode = params.detScaleMode; end

counts = count_blocks(basis);
M = numel(basis);
fprintf('Loaded variable: %s\n', stateName);
fprintf('Basis: M=%d | Mzero=%d | Mone=%d | Mtwo_ud=%d | Mtwo_uu=%d | Mthree=%d | oneFuncs=%d\n', ...
    M, counts.zero, counts.one_total, counts.two_ud, counts.two_uu, counts.three, numel(oneFuncs));
fprintf('Parameters: Z=%g, sigma=%g, c=%g, L=%d, neighbor=%s, detScale=%s\n', ...
    Zcharge, sigma0, cscale, L0, char(string(neighborMode)), char(string(assemblyOpts.detScaleMode)));
fprintf('Assembly: rowBlock=%d, dropTolS=%.1e, dropTolH=%.1e, screenERI=%d\n', ...
    assemblyOpts.rowBlockSize, assemblyOpts.dropTolS, assemblyOpts.dropTolH, assemblyOpts.screenERIByOverlap);
fprintf('References: HF %.8f | J.H. tilde final %.8f | exact diagnostic %.8f\n', ...
    E_ref_HF_table, E_ref_tilde_table, E_ref_exact);
print_memory_report('after load');
print_dense_projection_estimate(M, solverOpts);

if M > solverOpts.projectedDenseLimit
    error('M=%d exceeds projectedDenseLimit=%d. Increase solverOpts.projectedDenseLimit if intentional.', M, solverOpts.projectedDenseLimit);
end

%% ============================================================
% 2. One-particle cache
% =============================================================
tCache = tic;
oneCache = reset_one_cache(Zcharge, Rnuc);
oneCache = update_one_cache(oneCache, oneFuncs, Zcharge, Rnuc);
tCache = toc(tCache);
fprintf('\nOne-particle cache: %.2f s | n=%d | S1/H1 size=%d x %d\n', ...
    tCache, oneCache.n, size(oneCache.S1,1), size(oneCache.S1,2));
print_memory_report('after one-particle cache');

%% ============================================================
% 3. Sparse matrix assembly on the saved M=22630 basis
% =============================================================
tAsm = tic;
[S,H,asmInfo] = assemble_li_matrices(basis, oneCache, oneFuncs, assemblyOpts);
tAsm = toc(tAsm);

nnzS = nnz(S); nnzH = nnz(H); sparseGB = estimate_sparse_pair_gb(S,H);
fprintf('\nRaw assembly done: M=%d, nnz(S/H)=%.3e/%.3e, sparse est %.3f GB, time=%.2fs\n', ...
    M, nnzS, nnzH, sparseGB, tAsm);
if M >= 1
    EzeroDet = full(H(1,1)/S(1,1));
    fprintf('Zero determinant check: H(1,1)/S(1,1)=%.15f | HF table=%.15f | diff=%.3e\n', ...
        EzeroDet, E_ref_HF_table, EzeroDet-E_ref_HF_table);
end
print_memory_report('after sparse assembly');

%% ============================================================
% 4. Mass projection and lowest eigen solve
% =============================================================
tSol = tic;
[E0, coeff, nkeep, solverInfo] = solve_ground_projected_direct(H, S, mass_tol, verboseEig, E_ref_exact, solverOpts);
tSol = toc(tSol);
errExact = abs(E0 - E_ref_exact);
errTilde = abs(E0 - E_ref_tilde_table);
fprintf('\nProjected solve done: %.2f s | E=%.15f | err(exact)=%.3e | err(J.H. tilde)=%.3e | kept=%d | residual=%.3e\n', ...
    tSol, E0, errExact, errTilde, nkeep, solverInfo.residual);
fprintf('Mass spectrum: max=%.3e | minKept=%.3e | minAll=%.3e | nSmall=%d | condKept=%.3e\n', ...
    solverInfo.massMax, solverInfo.massMinKept, solverInfo.massMinAll, solverInfo.nSmall, solverInfo.condKept);
if solverOpts.rejectBelowExact && E0 < E_ref_exact - solverOpts.energyLowerTol
    warning('Energy %.12f is below diagnostic exact %.12f by %.3e; check basis/solver.', ...
        E0, E_ref_exact, E_ref_exact-E0);
end
print_memory_report('after projected solve');

%% ============================================================
% 5. Energy decomposition and save outputs
% =============================================================
tBE = tic;
blockEnergy = compute_block_energy_contributions(coeff, H, basis);
tBE = toc(tBE);
fprintf('\nTable 5.9-style row energy decomposition:\n');
fprintf('E_zero(row)   = %.15f\n', blockEnergy.row.zero);
fprintf('E_one(row)    = %.15f\n', blockEnergy.row.one);
fprintf('E_two_ud(row) = %.15f\n', blockEnergy.row.two_ud);
fprintf('E_two_uu(row) = %.15e\n', blockEnergy.row.two_uu);
fprintf('E_three(row)  = %.15e\n', blockEnergy.row.three);
fprintf('sum(row)      = %.15f\n', blockEnergy.row.total);
fprintf('E0            = %.15f\n', E0);

result = struct();
result.checkpointFile = checkpointFile;
result.M = M;
result.counts = counts;
result.E = E0;
result.errExact = errExact;
result.errTilde = errTilde;
result.coeff = coeff;
result.nkeep = nkeep;
result.solverInfo = solverInfo;
result.blockEnergy = blockEnergy;
result.times = struct('oneCache',tCache,'assembly',tAsm,'solve',tSol,'blockEnergy',tBE,'total',tCache+tAsm+tSol+tBE);
result.nnzS = nnzS;
result.nnzH = nnzH;
result.sparseGB = sparseGB;
result.assemblyOpts = assemblyOpts;
result.solverOpts = solverOpts;
result.params = params;

summaryCsv = sprintf('%s_summary.csv', outputPrefix);
write_direct_summary_csv(summaryCsv, result);
fprintf('\nSaved summary CSV: %s\n', summaryCsv);

outMat = sprintf('%s_result.mat', outputPrefix);
if saveSparseMatrices
    fprintf('Saving result with sparse S/H to %s ...\n', outMat);
    save(outMat, 'result', 'S', 'H', 'basis', 'oneFuncs', 'params', '-v7.3');
else
    fprintf('Saving result without S/H to %s ...\n', outMat);
    save(outMat, 'result', 'basis', 'oneFuncs', 'params', '-v7.3');
end
fprintf('Saved MAT: %s\n', outMat);

fprintf('\n==================== DIRECT M=22630 SOLVE SUMMARY ====================\n');
fprintf('M=%d | E=%.15f | errExact=%.3e | kept=%d | residual=%.3e\n', ...
    M, E0, errExact, nkeep, solverInfo.residual);
fprintf('time: cache=%.1fs | assembly=%.1fs | solve=%.1fs | blockEnergy=%.1fs | total=%.1fs\n', ...
    tCache, tAsm, tSol, tBE, result.times.total);
fprintf('nnz(S/H)=%.3e/%.3e | sparseEst=%.3f GB\n', nnzS, nnzH, sparseGB);
fprintf('======================================================================\n');

%% ========================================================================
% Local helpers specific to this direct script
% ========================================================================
function [state, stateName] = extract_checkpoint_state(Sload, fname)
    if isfield(Sload,'adaptiveState')
        state = Sload.adaptiveState; stateName = 'adaptiveState';
    elseif isfield(Sload,'finalState')
        state = Sload.finalState; stateName = 'finalState';
    elseif isfield(Sload,'part3')
        p = Sload.part3;
        state = struct('basis',p.basis0,'oneFuncs',p.oneFuncs,'params',p.params);
        if isfield(p,'hf'), state.hf = p.hf; end
        stateName = 'part3 -> state';
    else
        vars = fieldnames(Sload);
        error('File %s does not contain adaptiveState/finalState/part3. Variables: %s', fname, strjoin(vars.', ', '));
    end
    req = {'basis','oneFuncs','params'};
    for k = 1:numel(req)
        if ~isfield(state, req{k})
            error('Loaded %s does not contain required field %s.', stateName, req{k});
        end
    end
end

function print_dense_projection_estimate(M, solverOpts)
    denseOneGB = 8*M*M/1024^3;
    densePairGB = 2*denseOneGB;
    roughPeakGB = 7*denseOneGB;  % H,S,U,X,Hp,HX plus overhead, rough lower-bound
    fprintf('\nDense projection memory estimate for M=%d:\n', M);
    fprintf('  one full double matrix: %.2f GB\n', denseOneGB);
    fprintf('  full H+S only:          %.2f GB\n', densePairGB);
    fprintf('  rough eig/proj peak:    %.2f GB or more\n', roughPeakGB);
    if ~solverOpts.allowDenseSolveDespiteMemoryRisk
        mem = get_memory_info();
        if ~isnan(mem.availableGB) && mem.availableGB < roughPeakGB
            error('Available memory %.2f GB is below rough dense projection estimate %.2f GB.', mem.availableGB, roughPeakGB);
        end
    end
end

function print_memory_report(label)
    mem = get_memory_info();
    if isnan(mem.usedGB)
        fprintf('MEM %-28s | MATLAB memory info unavailable on this platform.\n', label);
    else
        fprintf('MEM %-28s | used %.2f GB | available %.2f GB | max array %.2f GB\n', ...
            label, mem.usedGB, mem.availableGB, mem.maxArrayGB);
    end
end

function mem = get_memory_info()
    mem = struct('usedGB',NaN,'availableGB',NaN,'maxArrayGB',NaN);
    try
        m = memory;
        mem.usedGB = m.MemUsedMATLAB/1024^3;
        mem.availableGB = m.MemAvailableAllArrays/1024^3;
        mem.maxArrayGB = m.MaxPossibleArrayBytes/1024^3;
    catch
        % memory() is primarily supported on Windows.  Keep NaN fallback.
    end
end

function [E0,coeff,nkeep,info] = solve_ground_projected_direct(H,S,mass_tol,verbose,targetEnergy,solverOpts)
    M = size(S,1);
    if M > solverOpts.projectedDenseLimit
        error('M=%d exceeds projectedDenseLimit=%d.', M, solverOpts.projectedDenseLimit);
    end
    print_dense_projection_estimate(M, solverOpts);

    tFull = tic;
    Hfull = full(H);
    Sfull = full(S);
    if isfield(solverOpts,'symmetrizeFullMatrices') && solverOpts.symmetrizeFullMatrices
        Hfull = 0.5*(Hfull+Hfull.');
        Sfull = 0.5*(Sfull+Sfull.');
    end
    tFull = toc(tFull);
    fprintf('    full conversion: %.2f s\n', tFull);

    tMass = tic;
    try
        [U,d] = eig(Sfull,'vector');
    catch
        [U,D] = eig(Sfull); d = diag(D);
    end
    d = real(d(:));
    [d,ord] = sort(d,'descend');
    U = U(:,ord);
    massMax = max(d);
    massMinAll = min(d);
    keep = d > mass_tol*massMax;
    nkeep = nnz(keep);
    if nkeep == 0, error('Mass matrix projection removed all dimensions.'); end
    nSmall = M - nkeep;
    massMinKept = min(d(keep));
    condKept = massMax / massMinKept;
    tMass = toc(tMass);
    fprintf('    mass eig: %.2f s | kept=%d/%d | nSmall=%d | condKept=%.3e\n', ...
        tMass, nkeep, M, nSmall, condKept);

    tProj = tic;
    Uk = U(:,keep);
    dk = d(keep);
    clear U d;
    X = bsxfun(@times, Uk, (1./sqrt(dk)).');
    clear Uk dk;
    HX = Hfull * X;
    Hp = X' * HX;
    clear HX;
    Hp = 0.5*(Hp+Hp.');
    tProj = toc(tProj);
    fprintf('    projection Hp=X''HX: %.2f s | projected dim=%d\n', tProj, nkeep);

    tEig = tic;
    try
        [Y,evals] = eig(Hp,'vector');
    catch
        [Y,Dp] = eig(Hp); evals = diag(Dp);
    end
    evals = real(evals(:));
    [E0,pos] = min(evals);
    y0 = Y(:,pos);
    clear Y Hp evals;
    coeff = X*y0;
    clear X y0;
    normS = sqrt(real(coeff'*(Sfull*coeff)));
    coeff = coeff / normS;
    res = norm(Hfull*coeff - E0*Sfull*coeff) / max(1,norm(Hfull*coeff));
    tEig = toc(tEig);
    fprintf('    projected eig: %.2f s | E=%.15f | residual=%.3e\n', tEig, E0, res);

    info = struct();
    info.method = 'dense_mass_projection_direct_M_checkpoint';
    info.targetEnergy = targetEnergy;
    info.residual = res;
    info.nkeep = nkeep;
    info.massMax = massMax;
    info.massMinAll = massMinAll;
    info.massMinKept = massMinKept;
    info.nSmall = nSmall;
    info.condKept = condKept;
    info.timeFull = tFull;
    info.timeMassEig = tMass;
    info.timeProjection = tProj;
    info.timeProjectedEig = tEig;
    if verbose
        fprintf('    projected solve total internal: %.2f s\n', tFull+tMass+tProj+tEig);
    end
end

function write_direct_summary_csv(fname, result)
    fid = fopen(fname,'w');
    if fid < 0
        warning('Could not open CSV for writing: %s', fname); return;
    end
    fprintf(fid,'M,E,errExact,errTilde,nkeep,residual,nnzS,nnzH,sparseGB,timeCache,timeAssembly,timeSolve,timeBlockEnergy,timeTotal,Mzero,Mone,Mtwo_ud,Mtwo_uu,Mthree,massMax,massMinKept,massMinAll,nSmall,condKept\n');
    c = result.counts; si = result.solverInfo; t = result.times;
    fprintf(fid,'%d,%.16g,%.16g,%.16g,%d,%.16g,%d,%d,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g,%d,%d,%d,%d,%d,%.16g,%.16g,%.16g,%d,%.16g\n', ...
        result.M, result.E, result.errExact, result.errTilde, result.nkeep, si.residual, ...
        result.nnzS, result.nnzH, result.sparseGB, t.oneCache, t.assembly, t.solve, t.blockEnergy, t.total, ...
        c.zero, c.one_total, c.two_ud, c.two_uu, c.three, si.massMax, si.massMinKept, si.massMinAll, si.nSmall, si.condKept);
    fclose(fid);
end


%% ========================================================================
% Assembly / integral functions imported from Adaptive_Li.m
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

%% ========================================================================
% Diagnostics functions imported from Adaptive_Li.m
% ========================================================================
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
