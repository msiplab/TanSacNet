function xh = fcn_dcdenoise(anet,snet,info,th,ab,v,sigma)
%FCN_DCDENOISE One-stage LSUN denoiser with optional DC shrinkage
%
%   xh = fcn_dcdenoise(anet,snet,info,th,ab,v,sigma)
%   f(v) = D_th S(E_th v), th = cell of angles (shared: nAngles x 1),
%   S = fcn_dcshrinkcoef (AC always; DC if ab has a0).
%   Without ab.a0 this is examples/sbdm/fcn_lsundenoise with S = 1.
%   Works inside dlfeval (forward), v: 'SSCB' dlarray, sigma scalar or
%   [1 1 1 B].
[anet,snet] = fcn_lsunsetangles(anet,snet,info,th);
[ac,dc] = forward(anet,v);
[acS,dcS] = fcn_dcshrinkcoef(ab,ac,dc,sigma);
xh = forward(snet,acS,dcS);
end
