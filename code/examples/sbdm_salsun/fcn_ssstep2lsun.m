function [prm,hist] = fcn_ssstep2lsun(anet,snet,info,prm,v,sigma,nIters,lr)
%FCN_SSSTEP2LSUN Step 2 for uniform LSUN: SURE on v, angles frozen
%
%   Learns only prm.a{1}, prm.b{1} (kappa_p, g_p of the 15 AC channels).
%   SURE with the exact divergence (angles fixed), computed by
%   examples/sbdm/fcn_lsunsure. examples/sbdm/fcn_trainlsunsure is not
%   used because it also updates the angles.
%   hist(it) = SURE (per pixel).

X = dlarray(v,'SSCB');
ab.a = prm.a{1}; ab.b = prm.b{1};
avgG = []; avgSqG = []; hist = zeros(nIters,1);
for it = 1:nIters
    [grads,loss] = dlfeval(@lossfun,anet,snet,info,prm.th,ab,X,sigma);
    [ab,avgG,avgSqG] = adamupdate(ab,grads,avgG,avgSqG,it,lr);
    hist(it) = double(gather(extractdata(loss)));
end
prm.a{1} = ab.a; prm.b{1} = ab.b;
end

function [grads,loss] = lossfun(anet,snet,info,th,ab,X,sigma)
p.th = th; p.a = {ab.a}; p.b = {ab.b};
sure = fcn_lsunsure(anet,snet,info,p,X,sigma,"exact");
loss = mean(sure,'all');
grads = dlgradient(loss,ab);
end
