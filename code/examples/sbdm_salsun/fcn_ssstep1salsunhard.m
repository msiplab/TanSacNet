function [net,hist,frac] = fcn_ssstep1salsunhard(net,v,thr,nIters,lr0,lr1)
%FCN_SSSTEP1SALSUNHARD Step 1 for SA-LSUN with a hard-threshold mask
%
%   Same as fcn_ssstep1salsun (estimator weights, Adam B1 0.9, B2 0.999,
%   eps 1e-12, cosine decay lr0 -> lr1), but with a hard-threshold mask
%   instead of the zonal mask:
%     M = (|c_ac| > thr), computed WITHOUT gradient from the current
%         coefficients and recomputed at every iteration
%     loss = ||v||^2 - ||dc||^2 - ||M .* ac||^2   (DC always kept)
%   net must be nets.train of fcn_ssbuildsalsunsz built with an all-ones
%   AC mask, so that layer Lv1_AcMask outputs all AC coefficients.
%   hist(it) = loss/||v||^2, frac(it) = fraction of AC coefficients with
%   M = 1 (before the update of iteration it).

X = dlarray(v,'SSCB');
e0 = double(gather(sum(v.^2,'all')));
avgG = []; avgSqG = []; hist = zeros(nIters,1); frac = zeros(nIters,1);
for it = 1:nIters
    [grads,loss,fr] = dlfeval(@lossfun,net,X,thr);
    lr = lr1 + (lr0 - lr1)*(1 + cos(pi*it/nIters))/2;
    [net,avgG,avgSqG] = adamupdate(net,grads,avgG,avgSqG,it,lr,0.9,0.999,1e-12);
    hist(it) = double(gather(extractdata(loss)))/e0;
    frac(it) = fr;
end
end

function [grads,loss,fr] = lossfun(net,X,thr)
[ac,dc] = forward(net,X,'Outputs',{'Lv1_AcMask','Lv1_DcOut'});
a = extractdata(ac);
M = cast(abs(a) > thr,'like',a);          % no gradient through the mask
loss = sum(X.^2,'all') - sum(dc.^2,'all') - sum((M.*ac).^2,'all');
grads = dlgradient(loss,net.Learnables);
fr = double(gather(mean(M,'all')));
end
