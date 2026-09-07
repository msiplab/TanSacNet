function [imgs,crops] = fcn_loadkodak(datfolder,idx,cropSize,nCrops,useGpu)
%FCN_LOADKODAK Grayscale Kodak images at full resolution, and crops of them
%
%   [imgs,crops] = fcn_loadkodak(datfolder,idx,cropSize,nCrops,useGpu)
%   returns the Kodak images of index IDX converted to grayscale at their
%   native resolution, and NCROPS random patches of size CROPSIZE taken from
%   each of them.
%
%   Training the score estimator on patches of full-resolution images and
%   evaluating it on whole images is what keeps the training cost independent
%   of the evaluation resolution. It also avoids the resolution mismatch that
%   arises when the training images are downsampled: downsampling removes the
%   high-frequency content that the shrinkage has to learn to keep.
%
%   Pass NCROPS = 0 to skip the patches.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

arguments
    datfolder (1,:) char
    idx (1,:) double
    cropSize (1,2) double = [64 64]
    nCrops (1,1) double = 0
    useGpu (1,1) logical = false
end

imgs = cell(1,numel(idx));
crops = cell(1,numel(idx)*nCrops);
c = 0;
for i = 1:numel(idx)
    x = im2single(rgb2gray(imread(fullfile(datfolder,sprintf('kodim%02d.png',idx(i))))));
    if useGpu, x = gpuArray(x); end
    imgs{i} = x;
    sz = size(x);
    for j = 1:nCrops
        r = randi(sz(1)-cropSize(1)+1);
        s = randi(sz(2)-cropSize(2)+1);
        c = c+1;
        crops{c} = x(r:r+cropSize(1)-1, s:s+cropSize(2)-1);
    end
end
crops = crops(1:c);
end
