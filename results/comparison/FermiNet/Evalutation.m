%% analyze_ferminet_eval_logs.m
% Fixed-theta FermiNet VMC evaluation postprocessing.
%
% This script follows the common FermiNet/VMCalike post-optimization
% evaluation logic:
%   1. read batch-mean local energies from fixed-theta inference logs;
%   2. average the energy time series;
%   3. estimate the standard error with a blocking analysis to account
%      for autocorrelation.
%
% Recommended input logs are generated with:
%   EVAL_STEPS=10000, MCMC_STEPS=10, STATS_FREQ=1, BATCH_SIZE=4096.
%
% Author: generated for sxw SOG-TNN/FermiNet benchmark

clear; clc;

%% ===================== User settings =====================
files = {
    'eval_He_singlet_eval_prr_style.log',  'He_singlet';
    'eval_He_triplet_eval_prr_style.log',  'He_triplet';
    'eval_Li_doublet_eval_prr_style.log', 'Li_doublet';
    'eval_Be_singlet_eval_prr_style.log',  'Be_singlet';
};

% If logs are not in the current MATLAB folder, set log_dir explicitly.
log_dir = pwd;

% Primary PRR-style summary uses all fixed-theta samples.
discard_fraction = 0.0;

% Robustness check: discard the first 10% logged points.
discard_fraction_check = 0.10;

% Blocking rule:
% use a conservative SE, namely the maximum blocking SE among levels with
% at least min_blocks block means remaining.
min_blocks = 16;

%% ===================== Main analysis =====================
summary = table();

for i = 1:size(files, 1)
    fname = fullfile(log_dir, files{i, 1});
    system_name = files{i, 2};

    [step, energy, variance, stdval, pmove] = parse_ferminet_log(fname);

    if isempty(energy)
        warning('No Step energy lines found in %s. Skip.', fname);
        continue;
    end

    % All samples
    result_all = summarize_energy_series(step, energy, variance, pmove, ...
        discard_fraction, min_blocks);

    % 10% discard robustness check
    result_d10 = summarize_energy_series(step, energy, variance, pmove, ...
        discard_fraction_check, min_blocks);

    one = table( ...
        string(system_name), ...
        numel(energy), min(step), max(step), ...
        result_all.mean_E, result_all.naive_SE, result_all.blocking_SE, ...
        result_all.block_level, result_all.block_size, result_all.n_blocks, ...
        result_all.mean_variance, result_all.mean_local_std, result_all.mean_pmove, ...
        result_d10.mean_E, result_d10.naive_SE, result_d10.blocking_SE, ...
        result_d10.block_level, result_d10.block_size, result_d10.n_blocks, ...
        'VariableNames', { ...
            'system', 'n_logged', 'step_min', 'step_max', ...
            'mean_E_all', 'naive_SE_all', 'blocking_SE_all', ...
            'blocking_level_all', 'blocking_block_size_all', 'blocking_nblocks_all', ...
            'mean_variance', 'mean_local_std', 'mean_pmove', ...
            'mean_E_discard10pct', 'naive_SE_discard10pct', 'blocking_SE_discard10pct', ...
            'blocking_level_discard10pct', 'blocking_block_size_discard10pct', ...
            'blocking_nblocks_discard10pct' ...
        });

    summary = [summary; one]; %#ok<AGROW>

    fprintf('\n============================================================\n');
    fprintf('%s\n', system_name);
    fprintf('n_logged = %d, step range = %d ... %d\n', numel(energy), min(step), max(step));
    fprintf('Primary, all samples:\n');
    fprintf('  E = %.16f Eh\n', result_all.mean_E);
    fprintf('  naive SE = %.6e Eh\n', result_all.naive_SE);
    fprintf('  blocking SE = %.6e Eh  [level=%d, block_size=%d, n_blocks=%d]\n', ...
        result_all.blocking_SE, result_all.block_level, result_all.block_size, result_all.n_blocks);
    fprintf('  mean variance = %.6e Eh^2, mean local std = %.6e Eh, pmove = %.6f\n', ...
        result_all.mean_variance, result_all.mean_local_std, result_all.mean_pmove);

    fprintf('Robustness check, discard first 10%% logged points:\n');
    fprintf('  E = %.16f Eh\n', result_d10.mean_E);
    fprintf('  blocking SE = %.6e Eh  [level=%d, block_size=%d, n_blocks=%d]\n', ...
        result_d10.blocking_SE, result_d10.block_level, result_d10.block_size, result_d10.n_blocks);
end

fprintf('\n================ Summary table ================\n');
disp(summary);

writetable(summary, 'ferminet_eval_summary_from_matlab.csv');
fprintf('\nSaved summary to ferminet_eval_summary_from_matlab.csv\n');

%% ===================== Local functions =====================
function [step, energy, variance, stdval, pmove] = parse_ferminet_log(fname)
    txt = fileread(fname);

    % Match lines like:
    % Step 00009999: energy=-7.480... E_h, variance=... E_h^2, std=... E_h, ... pmove=...
    expr = ['Step\s+(\d+):\s+energy=([-+0-9.eE]+)\s+E_h,\s+' ...
            'variance=([-+0-9.eE]+)\s+E_h\^2,\s+' ...
            'std=([-+0-9.eE]+)\s+E_h,.*?pmove=([-+0-9.eE]+)'];

    tokens = regexp(txt, expr, 'tokens');

    n = numel(tokens);
    step = zeros(n, 1);
    energy = zeros(n, 1);
    variance = zeros(n, 1);
    stdval = zeros(n, 1);
    pmove = zeros(n, 1);

    for k = 1:n
        step(k) = str2double(tokens{k}{1});
        energy(k) = str2double(tokens{k}{2});
        variance(k) = str2double(tokens{k}{3});
        stdval(k) = str2double(tokens{k}{4});
        pmove(k) = str2double(tokens{k}{5});
    end
end

function result = summarize_energy_series(step, energy, variance, pmove, discard_fraction, min_blocks)
    n = numel(energy);
    istart = floor(discard_fraction * n) + 1;

    e = energy(istart:end);
    v = variance(istart:end);
    p = pmove(istart:end);
    st = step(istart:end); %#ok<NASGU>

    mean_E = mean(e);
    naive_SE = std(e, 1) / sqrt(numel(e));  % population std convention; close for large n
    if numel(e) > 1
        naive_SE = std(e, 0) / sqrt(numel(e));
    end

    block_table = blocking_table(e);

    valid = block_table(block_table.n_blocks >= min_blocks, :);
    if isempty(valid)
        valid = block_table;
    end

    [blocking_SE, idx] = max(valid.SE);
    chosen = valid(idx, :);

    result.mean_E = mean_E;
    result.naive_SE = naive_SE;
    result.blocking_SE = blocking_SE;
    result.block_level = chosen.level;
    result.block_size = chosen.block_size;
    result.n_blocks = chosen.n_blocks;
    result.mean_variance = mean(v);
    result.mean_local_std = sqrt(max(mean(v), 0));
    result.mean_pmove = mean(p);
end

function T = blocking_table(x)
    x = x(:);
    level = [];
    n_blocks = [];
    block_size = [];
    mean_val = [];
    SE = [];

    y = x;
    bs = 1;
    lev = 0;

    while numel(y) >= 8
        n = numel(y);
        level(end+1, 1) = lev; %#ok<AGROW>
        n_blocks(end+1, 1) = n; %#ok<AGROW>
        block_size(end+1, 1) = bs; %#ok<AGROW>
        mean_val(end+1, 1) = mean(y); %#ok<AGROW>
        if n > 1
            SE(end+1, 1) = std(y, 0) / sqrt(n); %#ok<AGROW>
        else
            SE(end+1, 1) = NaN; %#ok<AGROW>
        end

        if mod(numel(y), 2) == 1
            y = y(1:end-1);
        end
        y = 0.5 * (y(1:2:end) + y(2:2:end));
        bs = bs * 2;
        lev = lev + 1;
    end

    T = table(level, n_blocks, block_size, mean_val, SE);
end
