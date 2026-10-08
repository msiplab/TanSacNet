function run_salsun2d_hls_codegen(inputSize,designName,options)
%RUN_SALSUN2D_HLS_CODEGEN Generate Vitis HLS C++ for salsun2d_hls
%
%   run_salsun2d_hls_codegen(inputSize) generates code for images of size
%   inputSize (default [32 32]) into
%   codegen/<szy>x<szx>/salsun2d_hls/hdlsrc.
%
%   run_salsun2d_hls_codegen(inputSize,designName) generates code for
%   designName, 'salsun2d_hls' (default), 'salsun2d_hls_opt' or
%   'salsun2d_hls_band', into codegen/<szy>x<szx>/<designName>/hdlsrc
%   (codegen/<szy>x<szx>_b<BandRows>/... for the band design).
%
%   For 'salsun2d_hls_band', inputSize is the frame size and the design
%   is generated for one band of BandRows block rows (default 31) with
%   L.Halo block rows of context above and below: an input of
%   (BandRows + 2*L.Halo)*My x szx. run_salsun2d_hls_codegen(...,
%   BandRows=n) changes the band size.
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
    designName {mustBeMember(designName,{'salsun2d_hls','salsun2d_hls_opt','salsun2d_hls_band'})} = 'salsun2d_hls'
    options.BandRows (1,1) double {mustBeInteger,mustBePositive} = 31
end
here = fileparts(mfilename('fullpath'));
addpath(here,fullfile(here,'..','..'),fullfile(here,'..','salsun'));
assignin('base','hlsInputSize',inputSize);
assignin('base','hlsDesignName',designName);
assignin('base','hlsBandRows',options.BandRows);

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

switch designName
    case 'salsun2d_hls_opt'
        args = {zeros(inputSize,'single'),zeros(L.NParams,1,'single'), ...
            zeros([L.NThetaRows inputSize./L.Stride],'single')};        % angle buffer
    case 'salsun2d_hls_band'
        subSize = [(options.BandRows + 2*L.Halo)*L.Stride(1) inputSize(2)];
        args = {zeros(subSize,'single'),zeros(L.NParams,1,'single'), ...
            zeros(L.NDec,L.NEst,'single'),ones(L.NDec,L.NEst,'single')}; % mu, sigma
    otherwise
        args = {zeros(inputSize,'single'),zeros(L.NParams,1,'single')};
end
outDir = fullfile(here,'codegen',sprintf('%dx%d',inputSize));
if strcmp(designName,'salsun2d_hls_band')
    outDir = [outDir sprintf('_b%d',options.BandRows)];   % one band size per directory
end
codegen('-config',cfg,designName,'-args',args,'-d',outDir);
end
