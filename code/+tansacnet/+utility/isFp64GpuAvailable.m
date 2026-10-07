function tf = isFp64GpuAvailable()
%ISFP64GPUAVAILABLE Check if a GPU with full-rate FP64 arithmetic is available
%
%   tf = tansacnet.utility.isFp64GpuAvailable() returns true if the
%   selected GPU device is expected to have high double-precision
%   throughput, e.g., NVIDIA P100, V100, A100, A800, H100 and B200.
%
%   The check is a heuristic based on the compute capability: the major
%   version is 6 or later and the minor version is 0 (GP100, GV100,
%   GA100, GH100, GB100, ...). GPUs for consumer and workstation use
%   (e.g., GeForce and RTX series) have reduced FP64 throughput and are
%   treated as unavailable even though they support double precision.
%
%   See also tansacnet.utility.isGpuAvailable
%
% Requirements: MATLAB R2022a
%
% Copyright (c) 2026, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%
tf = tansacnet.utility.isGpuAvailable('double');
if tf
    cc = str2double(split(string(gpuDevice().ComputeCapability),"."));
    tf = cc(1) >= 6 && cc(2) == 0;
end
end
