function xh = fcn_convtnrddenoise(prm,v,sigma)
%FCN_CONVTNRDDENOISE Convolutional TNRD denoiser (baseline)
%
%   xh = fcn_convtnrddenoise(prm,v,sigma) evaluates the tied-weight TNRD
%   denoiser of Example 10.2 of Muramatsu (2026), stage by stage,
%
%     x <- x - sigma*lambda_s*W_{a,s}^T*tanh(W_{a,s}*x + b_{a,s}),
%
%   following the noise-level conditional extension of Example 10.5. Pass
%   SIGMA = 1 to obtain the unconditional denoiser of Example 10.2.
%
%   Note that sigma multiplies the residual from outside the nonlinearity,
%   so for a single stage it only rescales a fixed correction direction. In
%   the LSUN realisation of FCN_LSUNDENOISE the noise level instead enters
%   the argument of the nonlinearity, which is exact because the transform
%   is unitary and hence leaves the noise level of the coefficients at
%   sigma.
%
%   V must be a dlarray with format 'SSCB'. SIGMA is a scalar, or an
%   array of size [1 1 1 B] giving one noise level per batch element.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

xh = v;
for s = 1:numel(prm.Wa)
    y = dlconv(xh,prm.Wa{s},prm.ba{s},'Padding','same');
    g = dltranspconv(tanh(y),prm.Wa{s},prm.bs{s},'Cropping','same');
    xh = xh - sigma.*exp(prm.loglam{s}).*g;
end
end
