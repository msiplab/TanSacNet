function [ab,hist] = fcn_dcstep2(ac,dc,sigma,N,ab,nIters,lr)
%FCN_DCSTEP2 Step 2 (SURE, angles frozen) for N, with optional DC shrinkage
%
%   [ab,hist] = fcn_dcstep2(ac,dc,sigma,N,ab,nIters,lr)
%   Same as fcn_ssstep2lsunfast (Adam, default betas, nIters, lr; exact
%   SURE from the coefficients c = E v computed once), but with
%   fcn_dcsure, so ab may contain the DC parameters a0, b0.
%   ac, dc : numeric coefficients. hist(it) = SURE (per pixel).
avgG = []; avgSqG = []; hist = zeros(nIters,1);
for it = 1:nIters
    [grads,loss] = dlfeval(@lossfun,ab,ac,dc,sigma,N);
    [ab,avgG,avgSqG] = adamupdate(ab,grads,avgG,avgSqG,it,lr);
    hist(it) = double(gather(extractdata(loss)));
end
end

function [grads,loss] = lossfun(ab,ac,dc,sigma,N)
loss = fcn_dcsure(ab,ac,dc,sigma,N);
grads = dlgradient(loss,ab);
end
