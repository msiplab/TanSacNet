%MYBUILD Script for building mex codes
%
% Requirements: MATLAB R2021a
%
% Copyright (c) 2017-2022, Shogo MURAMATSU
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
if exist('./mexcodes','dir') ~= 7
    mkdir('./mexcodes')
end

%% Set path
setpath

%% Build mex codes
if license('checkout','matlab_coder')
    import tansacnet.lsun.mexsrcs.*
    datatypes = { 'single', 'double' };
    for idatatype = 1:length(datatypes)
        for useGpuArray = 0:0 % Skip to use GPU Coder 
            fcn_build_orthmtxgen(datatypes{idatatype},useGpuArray);
            fcn_build_orthmtxgen_diff(datatypes{idatatype},useGpuArray);
        end
    end
end