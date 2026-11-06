function [net,hist] = fcn_ssstep2salsun(net,v,sigma,nIters,lr,epsFd)
%FCN_SSSTEP2SALSUN Step 2 for SA-LSUN: SURE on v, estimator frozen
%
%   Learns only the 30 parameters of layer Lv1_Shrink (kappa_p, g_p).
%   The angle estimator (step 1) is frozen, but the angles are still
%   re-estimated for every input, so SURE uses the Monte-Carlo divergence
%   of the whole map (one new probe per iteration), fcn_sssuresalsun "mc".
%   hist(it) = SURE (per pixel, one-probe estimate).

X = dlarray(v,'SSCB');
idx = find(strcmp(net.Learnables.Layer,'Lv1_Shrink'));
lrn = net.Learnables(idx,:);
avgG = []; avgSqG = []; hist = zeros(nIters,1);
for it = 1:nIters
    b = dlarray(randn(size(v),'like',v),'SSCB');
    [grads,loss] = dlfeval(@lossfun,net,X,sigma,b,epsFd,idx);
    [lrn,avgG,avgSqG] = adamupdate(lrn,grads,avgG,avgSqG,it,lr);
    L = net.Learnables; L.Value(idx) = lrn.Value; net.Learnables = L;
    hist(it) = double(gather(extractdata(loss)));
end
end

function [grads,loss] = lossfun(net,X,sigma,b,epsFd,idx)
loss = fcn_sssuresalsun(net,X,sigma,"mc",b,epsFd);
grads = dlgradient(loss,net.Learnables(idx,:));
end
