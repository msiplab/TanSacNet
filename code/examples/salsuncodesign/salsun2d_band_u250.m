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
    options.Statistics {mustBeMember(options.Statistics,{'previous','ema','fir2'})} = 'fir2'
    options.StatsRho (1,1) double = 0.9
end
here = fileparts(mfilename('fullpath'));
buildDir = fullfile(here,'build',target);
frameSize = [size(u,1) size(u,2)];
xclbinPath = fullfile(buildDir,sprintf('salsun2d_hls_band_%dx%d_b%d.xclbin',frameSize,options.BandRows));
if ~isfile(xclbinPath)
    error('salsun2d_band_u250:noXclbin', ...
        '%s not found. Build it with "make xclbin TARGET=%s DESIGN=salsun2d_hls_band SZY=%d SZX=%d BAND=%d".', ...
        xclbinPath,target,frameSize,options.BandRows)
end
if exist('salsun2d_band_mex','file') ~= 3
    error('salsun2d_band_u250:noMex','salsun2d_band_mex not found. Build it with build_mex_salsun2d.')
end
L = salsun2d_hls_layout();
w = salsun2d_pack_params(params);
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
        [yf,sums] = salsun2d_band_mex(xclbinPath,x,w,cat(3,mu,sigma));
        [muMeas,sigmaMeas] = salsun2d_stats_from_sums(sums(:,:,1),sums(:,:,2),nBlocks,L.Epsilon);
    end
[y,info] = salsun2d_band_sequence(u,params,@frameOnCard, ...
    Statistics=options.Statistics,StatsRho=options.StatsRho);
end
