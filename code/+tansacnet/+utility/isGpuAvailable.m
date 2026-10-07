function tf = isGpuAvailable(datatype)
%ISGPUAVAILABLE Check if a GPU is available for the given data type
%
%   tf = tansacnet.utility.isGpuAvailable() returns true if a supported
%   GPU device is available.
%
%   tf = tansacnet.utility.isGpuAvailable(datatype) also requires the
%   selected GPU device to support double-precision arithmetic when
%   DATATYPE is 'double'.
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
arguments
    datatype {mustBeMember(datatype,{'single','double'})} = 'single'
end

tf = canUseGPU;
if tf && strcmp(datatype,'double')
    tf = gpuDevice().SupportsDouble;
end
end
