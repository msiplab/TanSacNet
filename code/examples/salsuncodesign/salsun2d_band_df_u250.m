function [y,info] = salsun2d_band_df_u250(u,params,target,options)
%SALSUN2D_BAND_DF_U250 Streaming SA-LSUN inference on Alveo U250 with the dataflow band kernel
%
%   [y,info] = salsun2d_band_df_u250(u,params) runs the band kernel with the
%   layer-pipelined estimator (proto/band_df.cpp, salsun2d_band_df) on every
%   frame of u (szy x szx x N, single) and returns the reconstructed frames,
%   as salsun2d_band_u250 does for the HLS-generated kernel: the
%   standardization statistics come from the previous frames
%   (salsun2d_band_sequence; options Statistics, StatsRho, InitFast).
%
%   target is 'hw' (default) or 'sw_emu'. BandRows and Halo select the
%   xclbin proto/build/<target>/salsun2d_band_df_<szy>x<szx>_b<BandRows>h<Halo>_cu<ComputeUnits>.xclbin
%   (proto/Makefile). With FramesPerCall=n (up to ComputeUnits) n
%   consecutive frames run at the same time, one per compute unit;
%   otherwise the bands of a frame are split among the units.
%
%   params as from salsun2d_extract_params, 3 residual blocks of width 1.
%
% Requirements: MATLAB R2026b, XRT, the xclbin and salsun2d_band_df_mex
%               (see build_mex_salsun2d)
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
    u (:,:,:) single
    params (1,1) struct
    target {mustBeMember(target,{'hw','sw_emu'})} = 'hw'
    options.BandRows (1,1) double {mustBeInteger,mustBePositive} = 31
    options.Halo (1,1) double {mustBeInteger,mustBeNonnegative} = 7
    options.ComputeUnits (1,1) double {mustBeInteger,mustBePositive} = 3
    options.FramesPerCall (1,1) double {mustBeInteger,mustBePositive} = 1
    options.Statistics {mustBeMember(options.Statistics,{'previous','ema','fir2'})} = 'fir2'
    options.StatsRho (1,1) double = 0.9
    options.InitFast (1,1) logical = true
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here,'proto'));
buildDir = fullfile(here,'proto','build',target);
frameSize = [size(u,1) size(u,2)];
xclbinPath = fullfile(buildDir,sprintf('salsun2d_band_df_%dx%d_b%dh%d_cu%d.xclbin',frameSize, ...
    options.BandRows,options.Halo,options.ComputeUnits));
if ~isfile(xclbinPath)
    error('salsun2d_band_df_u250:noXclbin','%s not found. Build it with proto/Makefile.',xclbinPath)
end
if exist('salsun2d_band_df_mex','file') ~= 3
    error('salsun2d_band_df_u250:noMex','salsun2d_band_df_mex not found. Build it with build_mex_salsun2d.')
end
L = salsun2d_hls_layout();

% Parameters of the kernel: the small ones (proto/band_df.h) and those of
% the estimators (pack_est_df)
prm = single([params.Cvh(:); params.Mask(:); params.V0.MusW(:,1); params.V0.MusU(:,1); ...
    params.V0t.MusW(:,1); params.V0t.MusU(:,1)]);
for s = 1:numel(params.Stages), prm = [prm; single(params.Stages(s).Mus(:,1))]; end %#ok<AGROW>
for s = 1:numel(params.Stages), prm = [prm; single(params.Stages(s).SynMus(:,1))]; end %#ok<AGROW>
[w,b,ln] = pack_est_df(params);
W = w.'; LN = single(ln.');

setenv('XILINX_XRT','/opt/xilinx/xrt')
if strcmp(target,'hw')
    unsetenv('XCL_EMULATION_MODE');
else
    setenv('XCL_EMULATION_MODE',target)
    setenv('EMCONFIG_PATH',buildDir)
end

nBlocks = single(prod(frameSize./L.Stride));
    function [yf,muMeas,sigmaMeas] = frameOnCard(x,mu,sigma)
        % x: one frame, or one per compute unit (mu, sigma: NDec x NEst x n)
        n = size(x,3);
        stats = permute(cat(4,mu,sigma),[1 2 4 3]);        % NDec x NEst x 2 x n
        [yf,sums] = salsun2d_band_df_mex(xclbinPath,x,prm,W,b,LN,stats,options.BandRows,options.Halo);
        sums = reshape(sums,size(stats));
        muMeas = zeros(size(mu),'single'); sigmaMeas = zeros(size(sigma),'single');
        for j = 1:n
            [muMeas(:,:,j),sigmaMeas(:,:,j)] = salsun2d_stats_from_sums(sums(:,:,1,j),sums(:,:,2,j),nBlocks,L.Epsilon);
        end
    end
[y,info] = salsun2d_band_sequence(u,params,@frameOnCard, ...
    Statistics=options.Statistics,StatsRho=options.StatsRho,InitFast=options.InitFast, ...
    FramesPerCall=options.FramesPerCall);
end
