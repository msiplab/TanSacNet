function profiles = measure_receptive_field(options)
%MEASURE_RECEPTIVE_FIELD Spatial extent of the influence of one input block
%
%   measure_receptive_field() perturbs one 4 x 4 block of a random image
%   and measures, for each angle estimator, the masked coefficients and
%   the reconstructed output, how far (in blocks, Chebyshev distance with
%   circular wrap) the change reaches, with a network of randomly
%   perturbed parameters (salsun2d_create_test_network).
%
%   measure_receptive_field(ParamsFile=file) uses the trained parameters
%   saved by train_salsun2d_variants (field params of the .mat file) and
%   a frame of the wave equation data (option Frame, default 11) as the
%   image.
%
%   The reference implementation in double precision is used, so that
%   the profile is not masked by rounding.
%
%   Expected from the structure (see the Slack discussion): the footprint
%   of an input block grows by one block per estimator (3 x 3 neighbors)
%   and by one block per atom extension in its direction. Chebyshev
%   radius: estimator 1: 1, estimator 2: 3, estimator 3: 4, estimator 4:
%   5, estimator 5: 6, coefficients: 6, output: 7. Beyond that radius a
%   small change remains everywhere, because the state standardization
%   uses the statistics of the whole image. With trained parameters the
%   change decays by orders of magnitude within 2 to 3 blocks, well
%   inside the structural radius.
%
%   profiles(k,d+1) is the maximum change at distance d relative to the
%   change of the perturbed block, for estimators 1..5, the coefficients
%   and the output (rows 1..7).
%
% Requirements: MATLAB R2026b, Deep Learning Toolbox (random network)
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
    options.ParamsFile {mustBeTextScalar} = ''
    options.InputSize (1,2) double = [128 128]   % random image, 32 x 32 blocks
    options.Frame (1,1) double = 11              % frame of the wave data
end
here = fileparts(mfilename('fullpath'));
addpath(here,fullfile(here,'..','..'),fullfile(here,'..','salsun'));
L = salsun2d_hls_layout();

if isempty(options.ParamsFile)
    rng(11)
    nCoefs = 2;
    coefMask = reshape([ones(nCoefs,1); zeros(16-nCoefs,1)],2,[]).';
    net = salsun2d_create_test_network(options.InputSize,coefMask(:));
    params = salsun2d_cast_params(salsun2d_extract_params(net),'double');
    x = rand(options.InputSize);
    fprintf('Random network, %d x %d random image\n',options.InputSize);
else
    R = load(options.ParamsFile,'params');
    params = salsun2d_cast_params(R.params,'double');
    u = salsun2d_wave_data();
    x = double(u(:,:,options.Frame));
    fprintf('Trained parameters from %s, wave data frame %d\n',options.ParamsFile,options.Frame);
end
nBlk = size(x)./L.Stride;
center = floor(nBlk/2) + 1;          % perturbed block

xp = x;
rows = (center(1)-1)*L.Stride(1) + (1:L.Stride(1));
cols = (center(2)-1)*L.Stride(2) + (1:L.Stride(2));
rng(1)
xp(rows,cols) = xp(rows,cols) + 0.5*std(x(:))*(rand(L.Stride) - 0.5);

[y0,c0,th0] = salsun2d_infer(x,params);
[y1,c1,th1] = salsun2d_infer(xp,params);

% Chebyshev distance of every block from the perturbed block (circular)
[R,C] = ndgrid(1:nBlk(1),1:nBlk(2));
dr = min(mod(R-center(1),nBlk(1)),mod(center(1)-R,nBlk(1)));
dc = min(mod(C-center(2),nBlk(2)),mod(center(2)-C,nBlk(2)));
dist = max(dr,dc);
maxDist = max(dist(:));

names = {'estimator 1','estimator 2','estimator 3','estimator 4','estimator 5', ...
    'coefficients','output'};
expected = [1 3 4 5 6 6 7];
profiles = zeros(numel(names),maxDist+1);
% thetas are nAngles x nBlocks with the blocks in column-major order of
% the grid, so they reshape to nBlk
for k = 1:5
    profiles(k,:) = profileOf(reshape(max(abs(th1{k}-th0{k}),[],1),nBlk),dist,maxDist);
end
profiles(6,:) = profileOf(reshape(max(abs(c1-c0),[],1),nBlk),dist,maxDist);
dy = abs(y1-y0);
dyBlk = squeeze(max(max(reshape(dy,L.Stride(1),nBlk(1),L.Stride(2),nBlk(2)),[],1),[],3));
profiles(7,:) = profileOf(dyBlk,dist,maxDist);
profiles = profiles./profiles(:,1);

%% Report: max change per distance, relative to the change at distance 0
nShow = min(maxDist,9);
fprintf('Max relative change vs. Chebyshev block distance (perturbed block = 1)\n');
fprintf('%-14s %9s', 'quantity', 'expected');
fprintf(' %8d', 0:nShow);
fprintf('   floor\n');
for k = 1:numel(names)
    p = profiles(k,:);
    fprintf('%-14s %9d', names{k}, expected(k));
    fprintf(' %8.1e', p(1:nShow+1));
    fprintf('   %.1e\n', median(p(end-3:end)));
end

%% Local extent: last distance with a change above 1% of the perturbed
% block, and the global floor (the change far from the perturbed block,
% caused by the whole-image statistics of the standardization). When the
% floor itself is above 1%, the extent cannot be told apart.
fprintf('\nLocal extent (last distance with a change above 1%% of the perturbed block):\n');
for k = 1:numel(names)
    p = profiles(k,:);
    floorLevel = median(p(end-3:end));
    if floorLevel > 1e-2
        fprintf('%-14s not distinguishable  expected %2d  floor %.1e\n', names{k}, expected(k), floorLevel);
    else
        ext = find(p > 1e-2,1,'last') - 1;
        fprintf('%-14s measured %2d  expected %2d  floor %.1e\n', names{k}, ext, expected(k), floorLevel);
    end
end
end

function p = profileOf(blockChange,dist,maxDist)
p = zeros(1,maxDist+1);
for d = 0:maxDist
    p(d+1) = max(blockChange(dist == d));
end
end
