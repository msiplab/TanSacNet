function [x,v] = fcn_ssloadfull(datfolder,idx,cfg,useGpu)
%FCN_SSLOADFULL Full-size grayscale kodimXX and its noisy observation
%
%   Same as fcn_ssloadimage, but without the centre crop:
%   x : full image (single, [0,1]); v = x + cfg.sigma*n, n ~ N(0,1) drawn
%   on the CPU with rng(cfg.seedBase + idx) (same seed rule as the crops;
%   the noise image is larger, so it is not the crop's noise).
imgs = fcn_loadkodak(datfolder,idx);   % examples/sbdm helper (grayscale, single)
x = imgs{1};
rng(cfg.seedBase + idx,'twister');
v = x + cfg.sigma*randn(size(x),'single');
if useGpu
    x = gpuArray(x); v = gpuArray(v);
end
end
