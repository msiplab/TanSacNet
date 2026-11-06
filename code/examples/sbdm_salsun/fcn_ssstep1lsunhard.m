function [prm,hist,frac] = fcn_ssstep1lsunhard(anet,snet,info,prm,v,thr,nIters,lr0,lr1)
%FCN_SSSTEP1LSUNHARD Step 1 for uniform LSUN with a hard-threshold mask
%
%   Same as fcn_ssstep1lsun (shared angles, Adam, cosine decay lr0 -> lr1),
%   but the zonal mask is replaced by a hard-threshold mask:
%     M = (|c_ac| > thr), computed WITHOUT gradient from the current
%         coefficients and recomputed at every iteration
%     loss = ||v||^2 - ||dc||^2 - ||M .* ac||^2   (DC always kept)
%   i.e. maximise the energy of the AC coefficients above thr
%   (thr = tau*sigma).
%   hist(it) = loss/||v||^2, frac(it) = fraction of AC coefficients with
%   M = 1 (before the update of iteration it).

X = dlarray(v,'SSCB');
e0 = double(gather(sum(v.^2,'all')));
th = prm.th{1};
avgG = []; avgSqG = []; hist = zeros(nIters,1); frac = zeros(nIters,1);
for it = 1:nIters
    [grads,loss,fr] = dlfeval(@lossfun,anet,snet,info,th,X,thr);
    lr = lr1 + (lr0 - lr1)*(1 + cos(pi*it/nIters))/2;
    [th,avgG,avgSqG] = adamupdate(th,grads,avgG,avgSqG,it,lr,0.9,0.999,1e-12);
    hist(it) = double(gather(extractdata(loss)))/e0;
    frac(it) = fr;
end
prm.th{1} = th;
end

function [grads,loss,fr] = lossfun(anet,snet,info,th,X,thr)
anet = fcn_lsunsetangles(anet,snet,info,th);
[ac,dc] = forward(anet,X);
a = extractdata(ac);
M = cast(abs(a) > thr,'like',a);          % no gradient through the mask
loss = sum(X.^2,'all') - sum(dc.^2,'all') - sum((M.*ac).^2,'all');
grads = dlgradient(loss,th);
fr = double(gather(mean(M,'all')));
end
