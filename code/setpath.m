%SETPATH Path setup for *TanSacNet Package*
%
% Requirements: MATLAB R2022a
%
% Copyright (c) 2022, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%
% LinedIn: https://www.linkedin.com/in/shogo-muramatsu-627b084b
%
isMexCodesAvailable = true;

if  exist('./+tansacnet/','dir') == 7
    envname = 'TANSACNET_ROOT';
    if strcmp(getenv(envname),'')
        setenv(envname,pwd)
    end
    addpath(fullfile(getenv(envname),'.'))
    %
    sdirmexcodes = fullfile(getenv(envname),'mexcodes');
    if isMexCodesAvailable
        addpath(sdirmexcodes)
    elseif strfind(path,sdirmexcodes)  %#ok
        rmpath(sdirmexcodes)
    end
else
    error(['Move to the root directory of TanSacNet Package ' ...
           'before executing setpath.']);
end
