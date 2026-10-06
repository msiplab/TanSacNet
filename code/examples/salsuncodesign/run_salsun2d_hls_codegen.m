function run_salsun2d_hls_codegen(inputSize,designName)
%RUN_SALSUN2D_HLS_CODEGEN Generate Vitis HLS C++ for salsun2d_hls
%
%   run_salsun2d_hls_codegen(inputSize) generates code for images of size
%   inputSize (default [32 32]) into
%   codegen/<szy>x<szx>/salsun2d_hls/hdlsrc.
%
%   run_salsun2d_hls_codegen(inputSize,designName) generates code for
%   designName, 'salsun2d_hls' (default) or 'salsun2d_hls_opt', into
%   codegen/<szy>x<szx>/<designName>/hdlsrc.
%
% Requirements: MATLAB R2026b, HDL Coder, Deep Learning Toolbox
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
    inputSize (1,2) double = [32 32]
    designName {mustBeMember(designName,{'salsun2d_hls','salsun2d_hls_opt'})} = 'salsun2d_hls'
end
here = fileparts(mfilename('fullpath'));
addpath(here,fullfile(here,'..','..'),fullfile(here,'..','salsun'));
assignin('base','hlsInputSize',inputSize);
assignin('base','hlsDesignName',designName);

L = salsun2d_hls_layout();
cfg = coder.config('hls');
cfg.TestBenchName = 'salsun2d_hls_tb';
cfg.GenerateHLSTestBench = false;
cfg.SimulateGeneratedCode = false;
cfg.SynthesizeGeneratedCode = false;
cfg.SynthesisTool = "Xilinx Vitis HLS";
cfg.SynthesisToolChipFamily = 'Virtex UltraScale+';
cfg.SynthesisToolDeviceName = 'xcu250';
cfg.SynthesisToolPackageName = 'figd2104';
cfg.SynthesisToolSpeedValue = '-2L-e';

codegen('-config',cfg,designName,'-args', ...
    {zeros(inputSize,'single'),zeros(L.NParams,1,'single')}, ...
    '-d',fullfile(here,'codegen',sprintf('%dx%d',inputSize)));
end
