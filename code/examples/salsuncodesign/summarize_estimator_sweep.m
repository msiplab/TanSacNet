function T = summarize_estimator_sweep(options)
%SUMMARIZE_ESTIMATOR_SWEEP Accuracy and cost of the estimator configurations
%
%   T = summarize_estimator_sweep() collects the results of
%   run_estimator_sweep and of the baseline (3 residual blocks of width 2:
%   results/none.mat for seed 0 and results/seed<s>/none.mat) and returns
%   a table with, per configuration, the multiply-adds of the estimators
%   per block (fully connected layers), their ratio to the baseline, and
%   the MSE over the seeds (image statistics, frames 2-150).
%
% Requirements: MATLAB R2026b
%
% Copyright (c) 2026, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%
arguments
    options.ResultsDir = fullfile(fileparts(mfilename('fullpath')),'results')
end
rows = {};
% baseline
mse = [];
f0 = fullfile(options.ResultsDir,'none.mat');
if isfile(f0), R = load(f0,'mse'); mse(end+1) = R.mse.image; end
for s = 1:9
    f = fullfile(options.ResultsDir,sprintf('seed%d',s),'none.mat');
    if isfile(f), R = load(f,'mse'); mse(end+1) = R.mse.image; end %#ok<AGROW>
end
rows(end+1,:) = {3, 2, mse, 9, 0};
% sweep
d = dir(fullfile(options.ResultsDir,'arch','R*_W*'));
for i = 1:numel(d)
    v = sscanf(d(i).name,'R%d_W%f');
    nb = sscanf(regexp(d(i).name,'N\d+x\d+','match','once'),'N%dx%d');
    if isempty(nb), nb = [3;3]; end
    pb = sscanf(regexp(d(i).name,'P\d+','match','once'),'P%d');
    if isempty(pb), pb = 0; end
    mse = [];
    s = dir(fullfile(d(i).folder,d(i).name,'seed*','none.mat'));
    for k = 1:numel(s)
        R = load(fullfile(s(k).folder,s(k).name),'mse');
        mse(end+1) = R.mse.image; %#ok<AGROW>
    end
    rows(end+1,:) = {v(1), v(2), mse, prod(nb), pb}; %#ok<AGROW>
end
base = estimatorMacs(3,2,9,0);
n = size(rows,1);
T = table('Size',[n 10],'VariableTypes',repmat({'double'},1,10), ...
    'VariableNames',{'ResBlocks','Width','Neighbors','Bottleneck','MACs','MACsRatio','Seeds','MeanMSE','StdMSE','MSERatio'});
mBase = mean(rows{1,3});
for i = 1:n
    m = rows{i,3};
    mc = estimatorMacs(rows{i,1},rows{i,2},rows{i,4},rows{i,5});
    T(i,:) = {rows{i,1}, rows{i,2}, rows{i,4}, rows{i,5}, mc, mc/base, numel(m), mean(m), std(m), mean(m)/mBase};
end
T = sortrows(T,'MACs','descend');
end

function m = estimatorMacs(r,w,nNeighbors,p)
% Multiply-adds per block of the fully connected layers of the five
% estimators: features 15 or 8 channels x nNeighbors blocks, angles 49
% and 28, optionally projected to p features first
nF = [15 8 8 8 8]*nNeighbors;
nA = [49 28 28 28 28];
m = 0;
for k = 1:5
    if p > 0
        m = m + nF(k)*p;
        f = p;
    else
        f = nF(k);
    end
    h = round(w*f);
    m = m + r*2*f*h + nA(k)*f;
end
end
