function [nAnglesFull,nZeroPad] = fcn_salsunanglecounts(targetType,nChs,noDcLeakage)
%FCN_SALSUNANGLECOUNTS Numbers of angles a SA-LSUN rotation takes
%
%   [nAnglesFull,nZeroPad] = fcn_salsunanglecounts(targetType,nChs,noDcLeakage)
%   returns the number of angles nAnglesFull of the 'theta' input of the
%   rotation layer of type targetType ('Initial' or 'Intermediate') for
%   nChs channels, and the number nZeroPad of the leading angles that are
%   fixed to zero for no DC leakage. An angle estimator outputs all
%   nAnglesFull angles, the first nZeroPad of which are zero. The counts
%   do not depend on the number of spatial dimensions.
%
%   Initial      : nAnglesFull = (nChs-2)*nChs/4 (combined [anglesW; anglesU]),
%                  nZeroPad = nChs/2-1 with no DC leakage, otherwise 0
%   Intermediate : nAnglesFull = (nChs-2)*nChs/8 (antisymmetric half only),
%                  nZeroPad = 0 (no DC leakage does not apply)
%
% Requirements: MATLAB R2022a
%
% Copyright (c) 2026, Motoyasu SUZUKI and Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/

if mod(nChs,2)~=0
    throw(MException('SaLsunLayer:InvalidNumberOfChannels',...
        '%d : NumberOfChannels must be even.',nChs))
end
if strcmp(targetType,'Initial')
    nAnglesFull = (nChs-2)*nChs/4;
    if noDcLeakage
        nZeroPad = nChs/2-1;
    else
        nZeroPad = 0;
    end
elseif strcmp(targetType,'Intermediate')
    nAnglesFull = (nChs-2)*nChs/8;
    nZeroPad = 0;
else
    error('TargetType should be either of ''Initial'' or ''Intermediate''')
end
end
