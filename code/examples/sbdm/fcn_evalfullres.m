function [psnrs,ssims] = fcn_evalfullres(fden,imgs,sigmas,seedBase)
%FCN_EVALFULLRES PSNR and SSIM of a denoiser on full-resolution images
%
%   [psnrs,ssims] = fcn_evalfullres(fden,imgs,sigmas,seedBase) adds AWGN of
%   each level in SIGMAS to each image of the cell array IMGS and measures the
%   PSNR and the SSIM of FDEN(v,sigma). FDEN takes the image size as its third
%   argument, which lets the caller build an operator of the right size:
%
%     fden = @(v,sigma,sz) ...
%
%   The returned arrays are of size [numel(sigmas) numel(imgs)].
%
%   The noise realisation of image i at every noise level is fixed by
%   SEEDBASE+i on both the host and the device generator, so the comparison
%   between denoisers is on identical data.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

arguments
    fden function_handle
    imgs cell
    sigmas (1,:) single
    seedBase (1,1) double = 7000
end

nS = numel(sigmas); nI = numel(imgs);
psnrs = zeros(nS,nI); ssims = zeros(nS,nI);
for i = 1:nI
    x = imgs{i};
    sz = size(x);
    ref = double(gather(x));
    for k = 1:nS
        rng(seedBase+i)
        if canUseGPU, gpurng(seedBase+i); end
        v = x + sigmas(k)*randn(sz,'single','like',x);
        z = min(max(double(gather(fden(v,sigmas(k),sz))),0),1);
        psnrs(k,i) = psnr(z,ref);
        ssims(k,i) = ssim(z,ref);
    end
end
end
