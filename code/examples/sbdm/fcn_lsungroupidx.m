function [idx,nGroups] = fcn_lsungroupidx(info,groupSize)
%FCN_LSUNGROUPIDX Map each LSUN block to a group of neighbouring blocks
%
%   [idx,nGroups] = fcn_lsungroupidx(info,groupSize) returns IDX of size
%   [1 nBlocks]: block b uses the angles of group IDX(b). Groups are
%   rectangles of GROUPSIZE = [gr gc] blocks. [1 1] gives one group per block
%   (fully local LSUN); a group as large as the whole grid gives the uniform
%   (NSCLT) model.
%
%   Block order follows the rotation layers, which reshape the coefficient
%   grid [nrows ncols] column-major: block b = r + (c-1)*nrows.
%   Use square groups if in doubt: they do not depend on that convention.

nr = info.inputSize(1)/info.stride(1);
nc = info.inputSize(2)/info.stride(2);
[R,C] = ndgrid(1:nr,1:nc);
gr = ceil(R/groupSize(1));
gc = ceil(C/groupSize(2));
nGr = ceil(nr/groupSize(1));
idx = reshape(gr + (gc-1)*nGr,1,[]);
nGroups = max(idx);
end
