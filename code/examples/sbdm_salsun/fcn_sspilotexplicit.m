function [divJ,fu] = fcn_sspilotexplicit(ctx,ab,B)
%FCN_SSPILOTEXPLICIT Explicit MC divergence of the whole pilot chain
%
%   For each probe b_j (B: H x W x 1 x P), runs the full chain
%     v + eps b_j -> N -> p'_j -> angles (estimator) -> E, S, D
%   and returns divJ(j) = b_j'(f(v + eps b_j) - f(v)) / (eps N),
%   one value per probe (no precomputation, no unitarity shortcut).
%   Used to check fcn_sspilotsure. fu = f(v).
fu = chain(ctx,ab,ctx.X);
P = size(B,4); divJ = zeros(P,1);
for j = 1:P
    b = dlarray(B(:,:,:,j),'SSCB');
    fb = chain(ctx,ab,ctx.X + ctx.epsFd*b);
    divJ(j) = double(gather(extractdata(sum(b.*(fb - fu),'all'))))/(ctx.epsFd*numel(b));
end
end

function f = chain(ctx,ab,Z)
p = ctx.Nf(Z);
th = fcn_ssgetthetas(ctx.tnet,p,ctx.names);
net = fcn_sssetconstangles(ctx.cshrink,ctx.names,th,ab.a,ab.b);
f = predict(net,Z);
end
