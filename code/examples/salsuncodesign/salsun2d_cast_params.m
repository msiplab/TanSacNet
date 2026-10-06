function params = salsun2d_cast_params(params,classname)
%SALSUN2D_CAST_PARAMS Cast every numeric array in SA-LSUN parameters
%
%   params = salsun2d_cast_params(params,classname) casts the arrays in
%   params (from salsun2d_extract_params) to classname, e.g. 'double',
%   leaving index-like fields (Stride, Shift, Channels, Neighbor,
%   NumberOfZeroPadAngles) unchanged.
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
arguments
    params struct
    classname {mustBeTextScalar}
end
indexFields = {'Stride','Shift','SynShift','Channels','Neighbor','NumberOfZeroPadAngles'};
for iElem = 1:numel(params)
    names = fieldnames(params);
    for iField = 1:numel(names)
        v = params(iElem).(names{iField});
        if isstruct(v)
            params(iElem).(names{iField}) = salsun2d_cast_params(v,classname);
        elseif isfloat(v) && ~any(strcmp(names{iField},indexFields))
            params(iElem).(names{iField}) = cast(v,classname);
        end
    end
end
end
