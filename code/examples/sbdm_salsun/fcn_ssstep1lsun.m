function [prm,hist] = fcn_ssstep1lsun(anet,snet,info,prm,v,acMask,nIters,lr0,lr1)
%FCN_SSSTEP1LSUN Step 1 for uniform LSUN: zonal-coding autoencoder on v only
%
%   Same loss as fcn_ssstep1salsun, ||v||^2 - ||dc||^2 - ||mask.*ac||^2,
%   with the shared angles prm.th{1} (one column = uniform LSUN) of the
%   tansacnet.lsun pair from examples/sbdm/fcn_lsunpair. Only the angles
%   are learned. Adam with cosine decay from lr0 to lr1.
%   hist(it) = loss/||v||^2.

X = dlarray(v,'SSCB');
m = reshape(cast(acMask,'like',v),1,1,[]);
e0 = double(gather(sum(v.^2,'all')));
th = prm.th{1};
avgG = []; avgSqG = []; hist = zeros(nIters,1);
for it = 1:nIters
    [grads,loss] = dlfeval(@lossfun,anet,snet,info,th,X,m);
    lr = lr1 + (lr0 - lr1)*(1 + cos(pi*it/nIters))/2;
    [th,avgG,avgSqG] = adamupdate(th,grads,avgG,avgSqG,it,lr,0.9,0.999,1e-12);
    hist(it) = double(gather(extractdata(loss)))/e0;
end
prm.th{1} = th;
end

function [grads,loss] = lossfun(anet,snet,info,th,X,m)
anet = fcn_lsunsetangles(anet,snet,info,th);
[ac,dc] = forward(anet,X);
loss = sum(X.^2,'all') - sum(dc.^2,'all') - sum((m.*ac).^2,'all');
grads = dlgradient(loss,th);
end
