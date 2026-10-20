function c = salsun2d_band_config(newConfig)
%SALSUN2D_BAND_CONFIG Estimator size of the band design
%
%   c = salsun2d_band_config() returns [numResidualBlocks width halo] of
%   the design that salsun2d_hls_band is built for: the size of the angle
%   estimators (default 3 and 2, the network of main_salsun2d.m) and the
%   halo of a band in block rows (default L.Halo = 7; 0 for a design that
%   processes the whole frame height as one band).
%
%   salsun2d_band_config([r w]) or salsun2d_band_config([r w h]) sets it
%   for this MATLAB session, e.g. before the code generation of a design
%   with smaller estimators (run_salsun2d_hls_codegen does this from its
%   options); salsun2d_band_config('reset') restores the default.
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
default = [3 2 7];
if isempty(config)
    config = default;
end
if nargin > 0
    if (ischar(newConfig) || isstring(newConfig)) && strcmp(newConfig,'reset')
        config = default;
    else
        validateattributes(newConfig,{'numeric'},{'nonnegative'})
        assert(any(numel(newConfig) == [2 3]),'salsun2d_band_config:size', ...
            'The configuration is [r w] or [r w h].')
        config = default;
        config(1:numel(newConfig)) = double(newConfig(:).');
    end
end
c = config;
end
