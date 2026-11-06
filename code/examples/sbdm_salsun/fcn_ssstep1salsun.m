function [net,hist] = fcn_ssstep1salsun(net,v,nIters,lr0,lr1)
%FCN_SSSTEP1SALSUN Step 1 for SA-LSUN: zonal-coding autoencoder on v only
%
%   Loss ||v||^2 - ||F(v)||^2 (>= 0), F = analyzer + coefficient mask,
%   as modelGradients in examples/salsun/main_salsun2d.mlx. One image,
%   so one iteration = one epoch. Adam (B1 0.9, B2 0.999, eps 1e-12) with
%   cosine decay of the learning rate from lr0 to lr1, as in the example.
%   hist(it) = loss/||v||^2 (fraction of energy NOT kept).

X = dlarray(v,'SSCB');
e0 = double(gather(sum(v.^2,'all')));
avgG = []; avgSqG = []; hist = zeros(nIters,1);
for it = 1:nIters
    [grads,loss] = dlfeval(@lossfun,net,X);
    lr = lr1 + (lr0 - lr1)*(1 + cos(pi*it/nIters))/2;
    [net,avgG,avgSqG] = adamupdate(net,grads,avgG,avgSqG,it,lr,0.9,0.999,1e-12);
    hist(it) = double(gather(extractdata(loss)))/e0;
end
end

function [grads,loss] = lossfun(net,X)
Y = forward(net,X);
loss = sum(X.^2,'all') - sum(Y.^2,'all');
grads = dlgradient(loss,net.Learnables);
end
