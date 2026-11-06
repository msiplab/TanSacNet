function [divJ,fu] = fcn_dcsapexplicit(ctx,ab,B)
%FCN_DCSAPEXPLICIT Explicit MC divergence of the whole pilot chain (DC options)
%
%   fcn_sspilotexplicit with the DC shrinkage: for each probe b_j
%   (B: H x W x 1 x P) the full chain
%     v + eps b_j -> pilot -> p'_j -> angles (estimator) -> E, S, D
%   is run, with no precomputation and no unitarity shortcut, and
%   divJ(j) = b_j'(f(v + eps b_j) - f(v)) / (eps N). fu = f(v).
%   Used to check fcn_dcsurepilot.
fu = chain(ctx,ab,ctx.X);
P = size(B,4); divJ = zeros(P,1);
for j = 1:P
    b = dlarray(B(:,:,:,j),'SSCB');
    fb = chain(ctx,ab,ctx.X + ctx.epsFd*b);
    divJ(j) = double(gather(extractdata(sum(b.*(fb - fu),'all'))))/(ctx.epsFd*numel(b));
end
end

function f = chain(ctx,ab,Z)
p = ctx.Pf(Z);
th = fcn_ssgetthetas(ctx.tnet,p,ctx.names);
net = fcn_sssetconstangles(ctx.cshrink,ctx.names,th);
net = fcn_dcsetshrink(net,ab);
f = predict(net,Z);
end
