function [y,elapsed] = salsun2d_u250(x,w,target,options)
%SALSUN2D_U250 SA-LSUN 2-D analysis and synthesis on Alveo U250
%
%   y = salsun2d_u250(x,w) passes each frame of x (single, szy x szx x N)
%   through salsun2d_kernel on the Alveo U250 and returns the
%   reconstructed frames. w is the parameter vector from
%   salsun2d_pack_params. The result corresponds to salsun2d_hls(x,w)
%   applied to each frame. The xclbin is chosen by the frame size:
%   build/<target>/<design>_<szy>x<szx>.xclbin (see Makefile).
%
%   y = salsun2d_u250(x,w,target) selects the xclbin built for target,
%   which is 'hw' (default), 'hw_emu' or 'sw_emu'.
%
%   y = salsun2d_u250(...,Design=design) selects the kernel design,
%   'salsun2d_hls_opt' (default) or 'salsun2d_hls'.
%
%   [y,elapsed] = salsun2d_u250(...) also returns the elapsed seconds
%   [host-to-device, kernel, device-to-host] measured in the MEX gateway.
%
%   XRT fixes its mode (real card or emulation) when it is first used in a
%   process, so one MATLAB session can use only one target. Calling with
%   another target raises an error; restart MATLAB to switch. The target
%   in use is kept in the environment variable SALSUN2D_U250_TARGET.
%
% Requirements: MATLAB R2026b, XRT, build/<target>/<design>_<size>.xclbin,
%               salsun2d_u250_mex (see build_mex_salsun2d)
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
    x single {mustBeReal}
    w (:,1) single {mustBeReal}
    target {mustBeMember(target,{'hw','hw_emu','sw_emu'})} = 'hw'
    options.Design {mustBeMember(options.Design,{'salsun2d_hls_opt','salsun2d_hls'})} = 'salsun2d_hls_opt'
end

here = fileparts(mfilename('fullpath'));
buildDir = fullfile(here,'build',target);
frameSize = [size(x,1) size(x,2)];
xclbinPath = fullfile(buildDir,sprintf('%s_%dx%d.xclbin',options.Design,frameSize));
if ~isfile(xclbinPath)
    error('salsun2d_u250:noXclbin', ...
        '%s not found. Build it with "make xclbin TARGET=%s DESIGN=%s SZY=%d SZX=%d".', ...
        xclbinPath,target,options.Design,frameSize)
end
if exist('salsun2d_u250_mex','file') ~= 3
    error('salsun2d_u250:noMex', ...
        'salsun2d_u250_mex not found. Build it with build_mex_salsun2d.')
end
L = salsun2d_hls_layout();
if numel(w) ~= L.NParams
    error('salsun2d_u250:params','w must have %d elements.',L.NParams)
end

% One target per MATLAB session (see above)
sessionTarget = getenv('SALSUN2D_U250_TARGET');
if isempty(sessionTarget)
    setenv('SALSUN2D_U250_TARGET',target)
elseif ~strcmp(sessionTarget,target)
    error('salsun2d_u250:targetSwitch', ...
        ['This MATLAB session already uses target ''%s''. ' ...
        'Restart MATLAB to use target ''%s''.'],sessionTarget,target)
end

setenv('XILINX_XRT','/opt/xilinx/xrt')
if strcmp(target,'hw')
    % XRT treats an empty XCL_EMULATION_MODE as emulation, so remove it
    unsetenv('XCL_EMULATION_MODE');
else
    setenv('XCL_EMULATION_MODE',target)
    setenv('EMCONFIG_PATH',buildDir)
end

[y,elapsed] = salsun2d_u250_mex(xclbinPath,x,w);
end
