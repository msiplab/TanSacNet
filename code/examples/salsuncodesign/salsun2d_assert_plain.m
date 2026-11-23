function salsun2d_assert_plain(params,caller)
%SALSUN2D_ASSERT_PLAIN Check that the parameters need no plug-in support
%
%   salsun2d_assert_plain(params,caller) throws an error if params (from
%   salsun2d_extract_params) has the boundary operation 'Termination' or
%   an estimator other than the default one (constant angles, plug-in
%   estimators). caller names the function that implements the transform
%   itself (the fast MEX inference, the band and FPGA designs) and
%   supports only circular boundaries and the default estimator;
%   salsun2d_infer supports all of them.
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
if isfield(params,'BoundaryOperation') && ~strcmp(params.BoundaryOperation,'Circular')
    error('salsun2d:unsupported', ...
        '%s supports the boundary operation ''Circular'' only (use salsun2d_infer).',caller)
end
ests = [{params.V0.Estimator}, arrayfun(@(s) s.Estimator,params.Stages(:).','UniformOutput',false)];
for k = 1:numel(ests)
    e = ests{k};
    if isfield(e,'Function') || (isfield(e,'Type') && ~strcmp(e.Type,'param')) ...
            || (isfield(e,'Padding') && ~strcmp(e.Padding,'circular'))
        error('salsun2d:unsupported', ...
            '%s supports the default estimator with circular neighbors only (use salsun2d_infer).',caller)
    end
end
end
