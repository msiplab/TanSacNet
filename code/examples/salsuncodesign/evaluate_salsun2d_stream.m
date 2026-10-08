function results = evaluate_salsun2d_stream(options)
%EVALUATE_SALSUN2D_STREAM Error of band-wise streaming vs. the halo size
%
%   results = evaluate_salsun2d_stream() runs the trained SA-LSUN
%   (ParamsFile, default results/none.mat from train_salsun2d_variants)
%   on the wave equation data with salsun2d_infer_sequence in the
%   streaming configuration (causal statistics, bands of BandRows block
%   rows) for several halo sizes, and compares with whole-frame
%   processing:
%
%   - reconstruction MSE against the original frames
%   - RMS and maximum difference against the whole-frame result with the
%     same causal statistics (the error caused by the halo alone)
%   - the compute overhead (BandRows + 2*Halo)/BandRows
%
%   Options: ParamsFile, Frames (default 2:51), Halos (default
%   [1 2 3 4 6]), BandRows (5), Statistics ('fir2'), StatsRho (0.9).
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
    options.Frames (1,:) double = 2:51
    options.Halos (1,:) double = [1 2 3 4 6]
    options.BandRows (1,1) double = 5
    options.Statistics {mustBeMember(options.Statistics,{'previous','ema','fir2'})} = 'fir2'
    options.StatsRho (1,1) double = 0.9
end
here = fileparts(mfilename('fullpath'));
addpath(here);
R = load(options.ParamsFile,'params');
params = R.params;
u = salsun2d_wave_data();
u = single(u(:,:,[1 options.Frames]));   % frame 1 initializes the statistics
eval = 2:size(u,3);

fprintf('Trained parameters: %s\n',options.ParamsFile);
fprintf('%d frames of %d x %d, statistics %s, bands of %d block rows\n', ...
    numel(eval),size(u,1),size(u,2),options.Statistics,options.BandRows);

% Whole-frame processing with the same causal statistics
tic
yWhole = salsun2d_infer_sequence(u,params,Statistics=options.Statistics,StatsRho=options.StatsRho);
tWhole = toc;
yImage = salsun2d_infer_sequence(u,params,Statistics='image');
mseWhole = mean((u(:,:,eval) - yWhole(:,:,eval)).^2,'all');
mseImage = mean((u(:,:,eval) - yImage(:,:,eval)).^2,'all');
fprintf('whole frame, image statistics:  MSE %.4g\n',mseImage);
fprintf('whole frame, %s statistics: MSE %.4g  (%.1f s)\n',options.Statistics,mseWhole,tWhole);

results = struct('Halo',{},'Overhead',{},'Mse',{},'RmsDiff',{},'MaxDiff',{},'Time',{});
fprintf('%5s %9s %11s %12s %12s %8s\n','halo','overhead','MSE','RMS diff','max diff','time[s]');
for iHalo = 1:numel(options.Halos)
    H = options.Halos(iHalo);
    tic
    y = salsun2d_infer_sequence(u,params,Statistics=options.Statistics, ...
        StatsRho=options.StatsRho,Halo=H,BandRows=options.BandRows);
    t = toc;
    d = y(:,:,eval) - yWhole(:,:,eval);
    r.Halo = H;
    r.Overhead = (options.BandRows + 2*H)/options.BandRows;
    r.Mse = mean((u(:,:,eval) - y(:,:,eval)).^2,'all');
    r.RmsDiff = sqrt(mean(d.^2,'all'));
    r.MaxDiff = max(abs(d(:)));
    r.Time = t;
    results(iHalo) = r;
    fprintf('%5d %9.2f %11.4g %12.3g %12.3g %8.1f\n',H,r.Overhead,r.Mse,r.RmsDiff,r.MaxDiff,t);
end
fprintf('(RMS of the frames themselves: %.3g)\n',sqrt(mean(u(:,:,eval).^2,'all')));
end
