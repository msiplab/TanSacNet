function [y,info] = salsun2d_band_u250(u,params,target,options)
%SALSUN2D_BAND_U250 Streaming SA-LSUN inference on Alveo U250, band by band
%
%   [y,info] = salsun2d_band_u250(u,params) runs the band kernel
%   (salsun2d_band_kernel, design salsun2d_hls_band) on every frame of u
%   (szy x szx x N, single) and returns the reconstructed frames. The
%   kernel processes one frame per call, band by band on chip; the
%   standardization statistics are taken from the previous frames on the
%   host (salsun2d_band_sequence; options Statistics and StatsRho).
%
%   target is 'hw' (default), 'hw_emu' or 'sw_emu'; BandRows (default 31)
%   selects the xclbin build/<target>/salsun2d_hls_band_<szy>x<szx>_b<BandRows>.xclbin.
%   One MATLAB session can use only one XRT target (see salsun2d_u250).
%
%   With Halo=0 the xclbin is that of one band of the whole frame height
%   (Makefile BAND = szy/4, HALO = 0, ..._b<BandRows>h0...). With
%   FramesPerCall=n (up to ComputeUnits) n consecutive frames are run at
%   the same time, one whole frame per compute unit, all with the
%   statistics of the frames before them (salsun2d_band_sequence).
%
%   The statistics of the first frame are measured with the fixed-point
%   model of the kernel (salsun2d_band_fixed_model; about 1.3 s on the
%   CPU), or with InitFast=true by salsun2d_infer_fast (floating point,
%   about 0.1 s): after the first frames the stream runs at the kernel
%   time per frame.
%
% Requirements: MATLAB R2026b, XRT, the xclbin and salsun2d_band_mex
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
    target {mustBeMember(target,{'hw','hw_emu','sw_emu'})} = 'hw'
    options.BandRows (1,1) double {mustBeInteger,mustBePositive} = 31
    % Compute units in the xclbin (Makefile NCU): the bands of a frame are
    % split among them
    options.ComputeUnits (1,1) double {mustBeInteger,mustBePositive} = 1
    options.Statistics {mustBeMember(options.Statistics,{'previous','ema','fir2'})} = 'fir2'
    options.StatsRho (1,1) double = 0.9
    options.InitFast (1,1) logical = false
    options.Halo (1,1) double {mustBeInteger,mustBeNonnegative} = 7
    options.FramesPerCall (1,1) double {mustBeInteger,mustBePositive} = 1
end
here = fileparts(mfilename('fullpath'));
buildDir = fullfile(here,'build',target);
frameSize = [size(u,1) size(u,2)];
[nRes,width] = salsun2d_estimator_size(params);
estSize = '';
if nRes ~= 3 || width ~= 2
    estSize = sprintf('_r%dw%g',nRes,width);     % design with smaller estimators (Makefile RES, WIDTH)
end
if options.ComputeUnits > 1
    estSize = sprintf('%s_cu%d',estSize,options.ComputeUnits);
end
bandSize = sprintf('b%d',options.BandRows);
if options.Halo ~= 7
    bandSize = sprintf('%sh%d',bandSize,options.Halo);   % Makefile HALO
end
xclbinPath = fullfile(buildDir,sprintf('salsun2d_hls_band_%dx%d_%s%s.xclbin',frameSize,bandSize,estSize));
if ~isfile(xclbinPath)
    error('salsun2d_band_u250:noXclbin', ...
        '%s not found. Build it with "make xclbin TARGET=%s DESIGN=salsun2d_hls_band SZY=%d SZX=%d BAND=%d RES=%d WIDTH=%g".', ...
        xclbinPath,target,frameSize,options.BandRows,nRes,width)
end
if exist('salsun2d_band_mex','file') ~= 3
    error('salsun2d_band_u250:noMex','salsun2d_band_mex not found. Build it with build_mex_salsun2d.')
end
L = salsun2d_hls_layout();
w = salsun2d_pack_params(params,RowMajorWeights=true);
salsun2d_check_band_mask(w);

% One target per MATLAB session (see salsun2d_u250)
sessionTarget = getenv('SALSUN2D_U250_TARGET');
if isempty(sessionTarget)
    setenv('SALSUN2D_U250_TARGET',target)
elseif ~strcmp(sessionTarget,target)
    error('salsun2d_band_u250:targetSwitch', ...
        'This MATLAB session already uses target ''%s''. Restart MATLAB to use ''%s''.',sessionTarget,target)
end
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
        [yf,sums] = salsun2d_band_mex(xclbinPath,x,w,stats,options.BandRows);
        sums = reshape(sums,size(stats));
        muMeas = zeros(size(mu),'single'); sigmaMeas = zeros(size(sigma),'single');
        for j = 1:n
            [muMeas(:,:,j),sigmaMeas(:,:,j)] = salsun2d_stats_from_sums(sums(:,:,1,j),sums(:,:,2,j),nBlocks,L.Epsilon);
        end
    end
% The first frame is initialized with the statistics of the fixed-point
% model of the kernel, or of the floating-point network (InitFast)
[paramsQ,q] = salsun2d_band_fixed_model(params);
[y,info] = salsun2d_band_sequence(u,params,@frameOnCard, ...
    Statistics=options.Statistics,StatsRho=options.StatsRho, ...
    InitParams=paramsQ,InitQuantizer=q,InitFast=options.InitFast, ...
    FramesPerCall=options.FramesPerCall);
end
