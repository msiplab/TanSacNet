function prm = fcn_lsunexpandprm(prm,info)
%FCN_LSUNEXPANDPRM Embed a uniform angle set into the locally-structured one
%
%   prm = fcn_lsunexpandprm(prm,info) replaces every uniform angle vector
%   prm.th{s}{k} of size [nAngles(k) x 1] by the [nAngles(k) x nBlocks]
%   array that repeats it over the block grid, leaving the shrinkage
%   parameters prm.a and prm.b untouched.
%
%   The uniform parameterisation is the subset of the locally-structured one
%   on which every column of every angle array is the same, so the denoiser
%   returned here is *identical* to the one passed in. Training onwards from
%   this point therefore searches a strict superset of the uniform model and
%   can only improve on it, which is what makes the comparison of the two
%   parameterisations meaningful: starting the locally-structured model from
%   scratch instead pits 168*nBlocks angles against 168 on the same budget
%   and measures the optimisation, not the parameterisation.
%
%   See also FCN_LSUNINITPRM, FCN_LSUNSETANGLES.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

arguments
    prm struct
    info struct
end

for s = 1:numel(prm.th)
    for k = 1:numel(prm.th{s})
        th = prm.th{s}{k};
        if size(th,2) == 1
            prm.th{s}{k} = repmat(th,1,info.nBlocks);
        elseif size(th,2) ~= info.nBlocks
            error("tansacnet:lsun:BlockCountMismatch",...
                ['Rotation %d carries %d columns, which is neither 1 ' ...
                 '(uniform) nor the %d blocks of this network.'],...
                k,size(th,2),info.nBlocks);
        end
    end
end
end
