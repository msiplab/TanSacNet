function [x,v] = fcn_ssloadimage(datfolder,idx,cfg,useGpu)
%FCN_SSLOADIMAGE Centre crop of grayscale kodimXX and its noisy observation
%
%   x : clean crop (single, [0,1]); v = x + sigma*n, n ~ N(0,1) drawn on
%   the CPU with rng(cfg.seedBase + idx), so v does not depend on the GPU.
imgs = fcn_loadkodak(datfolder,idx);   % examples/sbdm helper (grayscale, single)
img = imgs{1};
[h,w] = size(img);
r0 = floor((h - cfg.cropSize(1))/2);
c0 = floor((w - cfg.cropSize(2))/2);
x = img(r0+(1:cfg.cropSize(1)), c0+(1:cfg.cropSize(2)));
rng(cfg.seedBase + idx,'twister');
v = x + cfg.sigma*randn(size(x),'single');
if useGpu
    x = gpuArray(x); v = gpuArray(v);
end
end
