function results = evaluate_salsun2d_wordlength(options)
%EVALUATE_SALSUN2D_WORDLENGTH Reconstruction error vs. fixed-point word length
%
%   results = evaluate_salsun2d_wordlength() runs the trained SA-LSUN
%   (ParamsFile, default results/none.mat) on frames of the wave equation
%   data in double precision with simulated fixed-point rounding
%   (salsun2d_fixed_quantizer, salsun2d_quantize_params) and reports, for
%   each setting, the reconstruction MSE relative to the double reference
%   and the largest deviations of the output and of the last estimator's
%   angles from the double reference:
%
%   A. parameters only, B. all intermediate signals only, C. both with
%   the same word length, D. one signal class at a time at TagWordLength
%   bits (sensitivity).
%
%   Options: ParamsFile, Frames (default 11:15), WordLengths (default
%   [8 10 12 14 16 18]), TagWordLength (12).
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
    options.ParamsFile {mustBeTextScalar} = fullfile(fileparts(mfilename('fullpath')),'results','none.mat')
    options.Frames (1,:) double = 11:15
    options.WordLengths (1,:) double = [8 10 12 14 16 18]
    options.TagWordLength (1,1) double = 12
end
here = fileparts(mfilename('fullpath'));
addpath(here);
R = load(options.ParamsFile,'params');
params = salsun2d_cast_params(R.params,'double');
u = salsun2d_wave_data();
u = u(:,:,options.Frames);
nFrames = size(u,3);
tags = {'coefs','rotation','features','ln','z1','act','angles'};

% Double reference
[yRef,thRef] = run(u,params,@(v,tag) v);
mseRef = mean((u - yRef).^2,'all');
fprintf('Trained parameters: %s, %d frames of %d x %d\n',options.ParamsFile,nFrames,size(u,1),size(u,2));
fprintf('double reference: MSE %.4g\n\n',mseRef);

results = struct('Case',{},'WordLength',{},'Tag',{},'Mse',{},'MseRatio',{},'MaxDiff',{},'AngleErr',{});
header = sprintf('%-28s %5s %10s %9s %10s %10s','case','bits','MSE','MSE/ref','max|dy|','max|dth5|');
cases = {
    'A parameters only',   @(wl) deal(salsun2d_quantize_params(params,wl),@(v,tag) v)
    'B signals only',      @(wl) deal(params,salsun2d_fixed_quantizer(wl))
    'C parameters+signals',@(wl) deal(salsun2d_quantize_params(params,wl),salsun2d_fixed_quantizer(wl))};
for iCase = 1:size(cases,1)
    fprintf('%s\n',header);
    for wl = options.WordLengths
        [p,q] = cases{iCase,2}(wl);
        r = evaluateOne(u,p,q,yRef,thRef,mseRef,cases{iCase,1},wl,'all');
        results(end+1) = r; %#ok<AGROW>
        fprintf('%-28s %5d %10.4g %9.3f %10.3g %10.3g\n',r.Case,wl,r.Mse,r.MseRatio,r.MaxDiff,r.AngleErr);
    end
    fprintf('\n');
end

% D. sensitivity per signal class
fprintf('%s\n',header);
for iTag = 1:numel(tags)
    wls = struct(tags{iTag},options.TagWordLength);
    r = evaluateOne(u,params,salsun2d_fixed_quantizer(wls),yRef,thRef,mseRef, ...
        sprintf('D %s only',tags{iTag}),options.TagWordLength,tags{iTag});
    results(end+1) = r; %#ok<AGROW>
    fprintf('%-28s %5d %10.4g %9.3f %10.3g %10.3g\n',r.Case,options.TagWordLength,r.Mse,r.MseRatio,r.MaxDiff,r.AngleErr);
end
wls = struct('weights',options.TagWordLength);  %#ok<NASGU>
r = evaluateOne(u,salsun2d_quantize_params(params,options.TagWordLength),@(v,tag) v,yRef,thRef,mseRef, ...
    'D parameters only',options.TagWordLength,'weights');
results(end+1) = r;
fprintf('%-28s %5d %10.4g %9.3f %10.3g %10.3g\n',r.Case,options.TagWordLength,r.Mse,r.MseRatio,r.MaxDiff,r.AngleErr);
end

function [y,th5] = run(u,params,q)
nFrames = size(u,3);
y = zeros(size(u));
th5 = cell(nFrames,1);
for t = 1:nFrames
    [y(:,:,t),~,thetas] = salsun2d_infer(u(:,:,t),params,Quantizer=q);
    th5{t} = thetas{end};
end
end

function r = evaluateOne(u,params,q,yRef,thRef,mseRef,name,wl,tag)
[y,th5] = run(u,params,q);
r.Case = name;
r.WordLength = wl;
r.Tag = tag;
r.Mse = mean((u - y).^2,'all');
r.MseRatio = r.Mse/mseRef;
r.MaxDiff = max(abs(y(:) - yRef(:)));
r.AngleErr = max(cellfun(@(a,b) max(abs(a(:)-b(:))),th5,thRef));
end
