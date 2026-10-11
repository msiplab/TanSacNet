function T = evaluate_heldout_sweep(options)
%EVALUATE_HELDOUT_SWEEP Held-out reconstruction error of the trained networks
%
%   T = evaluate_heldout_sweep() finds every results/arch*/**/none.mat
%   (run_estimator_sweep) and the baseline results/none.mat and
%   results/seed*/none.mat, and computes the reconstruction MSE (image
%   statistics) on frames of the wave simulation that the training did not
%   use (HeldOutFrames, default simulation frames 551-700, every Step-th
%   frame) and on the training frames (401-550, every Step-th frame). The
%   result is saved next to each file as heldout.mat (skipped if present)
%   and returned as a table.
%
%   The networks fit the training frames much better than unseen frames
%   (held-out MSE 4 to 15 times the training MSE, growing with the number
%   of epochs), so configurations are compared on the held-out MSE.
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
    options.HeldOutFrames (1,2) double = [551 700]
    options.Step (1,1) double {mustBeInteger,mustBePositive} = 3
end
here = fileparts(mfilename('fullpath'));
addpath(here,fullfile(here,'..','..'),fullfile(here,'..','salsun'));
uTr = single(salsun2d_wave_data());
uTr = uTr(:,:,2:options.Step:end);
uHo = single(salsun2d_wave_data(Frames=options.HeldOutFrames));
uHo = uHo(:,:,1:options.Step:end);

files = [dir(fullfile(options.ResultsDir,'arch*','**','none.mat')); ...
    dir(fullfile(options.ResultsDir,'none.mat')); ...
    dir(fullfile(options.ResultsDir,'seed*','none.mat'))];
names = strings(numel(files),1);
trainMse = nan(numel(files),1);
heldOutMse = nan(numel(files),1);
for i = 1:numel(files)
    f = fullfile(files(i).folder,files(i).name);
    names(i) = string(erase(f,[options.ResultsDir filesep]));
    out = fullfile(files(i).folder,'heldout.mat');
    if isfile(out)
        H = load(out);
    else
        R = load(f,'params');
        mse = @(u) mean(arrayfun(@(t) mean((u(:,:,t) - salsun2d_infer(u(:,:,t),R.params)).^2,'all'),1:size(u,3)));
        H.trainMse = mse(uTr);
        H.heldOutMse = mse(uHo);
        H.heldOutFrames = options.HeldOutFrames;
        H.step = options.Step;
        save(out,'-struct','H');
    end
    trainMse(i) = H.trainMse;
    heldOutMse(i) = H.heldOutMse;
    fprintf('%-55s train %.4g  held-out %.4g\n',names(i),trainMse(i),heldOutMse(i));
end
T = table(names,trainMse,heldOutMse,'VariableNames',{'File','TrainMSE','HeldOutMSE'});
end
