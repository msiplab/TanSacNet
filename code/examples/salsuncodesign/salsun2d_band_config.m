function c = salsun2d_band_config(newConfig)
%SALSUN2D_BAND_CONFIG Estimator size of the band design
%
%   c = salsun2d_band_config() returns [numResidualBlocks width] of the
%   angle estimators that salsun2d_hls_band is built for (default [3 2],
%   the network of main_salsun2d.m).
%
%   salsun2d_band_config([r w]) sets it for this MATLAB session, e.g.
%   before the code generation of a design with smaller estimators
%   (run_salsun2d_hls_codegen does this from its options);
%   salsun2d_band_config('reset') restores the default.
%
%   The band design reads it at code generation time (coder.const of an
%   extrinsic call), so one generated design has one estimator size.
%
% Requirements: MATLAB R2026b
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
persistent config
if isempty(config)
    config = [3 2];
end
if nargin > 0
    if (ischar(newConfig) || isstring(newConfig)) && strcmp(newConfig,'reset')
        config = [3 2];
    else
        validateattributes(newConfig,{'numeric'},{'numel',2,'positive'})
        config = double(newConfig(:).');
    end
end
c = config;
end
