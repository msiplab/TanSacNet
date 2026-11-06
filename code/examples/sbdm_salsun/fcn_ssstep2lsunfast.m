function [prm,hist] = fcn_ssstep2lsunfast(anet,snet,info,prm,v,sigma,nIters,lr)
%FCN_SSSTEP2LSUNFAST Step 2 for uniform LSUN, frozen-transform shortcut
%
%   Same as fcn_ssstep2lsun (learns only prm.a{1}, prm.b{1}; Adam, nIters,
%   lr; SURE with the exact divergence, angles frozen), but the transform
%   is applied only ONCE. With c = E v (E unitary, DC passed through):
%     ||f(v) - v||^2 = ||sigma (kappa/g) tanh(g c_ac / sigma)||^2
%     div f          = N - sum_AC kappa sech^2(g c_ac / sigma)
%   so every iteration costs only element-wise operations on c.
%   This is the same SURE as examples/sbdm/fcn_lsunsure(...,"exact"),
%   only the residual is computed in the coefficient domain.
%   hist(it) = SURE (per pixel).

[anet,~] = fcn_lsunsetangles(anet,snet,info,prm.th{1});
[ac,~] = predict(anet,dlarray(v,'SSCB'));
ac = extractdata(ac);
N = prod(info.inputSize);
ab.a = prm.a{1}; ab.b = prm.b{1};
avgG = []; avgSqG = []; hist = zeros(nIters,1);
for it = 1:nIters
    [grads,loss] = dlfeval(@lossfun,ab,ac,sigma,N);
    [ab,avgG,avgSqG] = adamupdate(ab,grads,avgG,avgSqG,it,lr);
    hist(it) = double(gather(extractdata(loss)));
end
prm.a{1} = ab.a; prm.b{1} = ab.b;
end

function [grads,loss] = lossfun(ab,ac,sigma,N)
loss = fcn_ssfastsure(ab,ac,sigma,N);
grads = dlgradient(loss,ab);
end
