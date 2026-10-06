function build_mex_salsun2d()
%BUILD_MEX_SALSUN2D Build the MEX gateway salsun2d_u250_mex for Alveo U250
%
% Requirements: MATLAB R2026b, XRT (/opt/xilinx/xrt)
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

xrtRoot = '/opt/xilinx/xrt';
here = fileparts(mfilename('fullpath'));

mex('-R2018a', ...
    ['-I' fullfile(xrtRoot,'include')], ...
    ['-L' fullfile(xrtRoot,'lib')], '-lxrt_coreutil', ...
    'CXXFLAGS=$CXXFLAGS -std=c++17', ...
    ['LDFLAGS=$LDFLAGS -Wl,-rpath,' fullfile(xrtRoot,'lib')], ...
    '-outdir', here, ...
    fullfile(here,'mex','salsun2d_u250_mex.cpp'));
end
