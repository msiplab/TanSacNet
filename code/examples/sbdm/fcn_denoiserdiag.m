function d = fcn_denoiserdiag(fdenoise,fpair,sz,nProbe,nPower)
%FCN_DENOISERDIAG Structural diagnostics of a denoiser
%
%   d = fcn_denoiserdiag(fdenoise,fpair,sz,nProbe,nPower) measures the
%   properties that the plug-and-play, RED and SBDM frameworks of
%   Chapter 10 of Muramatsu (2026) require of a denoiser. FDENOISE is a
%   handle f(v) taking and returning a dlarray of format 'SSCB'. FPAIR is a
%   handle x -> D*E*x realising the analysis operator followed by its
%   adjoint, or [] to skip the tightness measurement.
%
%   Returned fields:
%
%     d.tightErr  mean of ||D*E*x - x||/||x|| over random probes, i.e. the
%                 departure from Parseval tightness of the dictionary
%     d.jacAsym   mean relative asymmetry of the Jacobian,
%                 |u'*J*w - w'*J*u|/(||J'*u||*||w||), measured exactly by
%                 automatic differentiation. RED requires this to vanish
%     d.jacRho    spectral radius of the Jacobian by power iteration on J',
%                 equal to ||J||_2 when the Jacobian is symmetric.
%                 Non-expansiveness requires this to be at most one
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

arguments
    fdenoise function_handle
    fpair
    sz (1,2) double
    nProbe (1,1) double = 8
    nPower (1,1) double = 30
end

% departure from Parseval tightness
d.tightErr = NaN;
if ~isempty(fpair)
    e = zeros(nProbe,1);
    for i = 1:nProbe
        x = dlarray(randn(sz,'single'),'SSCB');
        y = fpair(x);
        e(i) = sqrt(fcn_scalar(sum((y-x).^2,'all'))/fcn_scalar(sum(x.^2,'all')));
    end
    d.tightErr = mean(e);
end

% Jacobian asymmetry: u'*J*w = (J'*u)'*w and w'*J*u = (J'*w)'*u
v0 = dlarray(rand(sz,'single'),'SSCB');
a = zeros(nProbe,1);
for i = 1:nProbe
    u = dlarray(randn(sz,'single'),'SSCB');
    w = dlarray(randn(sz,'single'),'SSCB');
    Jtu = dlfeval(@vjp,fdenoise,v0,u);
    Jtw = dlfeval(@vjp,fdenoise,v0,w);
    n1 = fcn_scalar(sum(Jtu.*w,'all'));
    n2 = fcn_scalar(sum(Jtw.*u,'all'));
    d1 = fcn_scalar(sum(Jtu.^2,'all'));
    d2 = fcn_scalar(sum(w.^2,'all'));
    a(i) = abs(n1-n2)/sqrt(d1*d2);
end
d.jacAsym = mean(a);

% spectral radius by power iteration on J'
p = dlarray(randn(sz,'single'),'SSCB');
p = p/sqrt(fcn_scalar(sum(p.^2,'all')));
rho = 0;
for i = 1:nPower
    q = dlfeval(@vjp,fdenoise,v0,p);
    rho = sqrt(fcn_scalar(sum(q.^2,'all')));
    p = q/rho;
end
d.jacRho = rho;
end

function g = vjp(fdenoise,v,u)
% vector-Jacobian product J'*u at v
v = dlarray(v,'SSCB');
y = fdenoise(v);
g = dlgradient(sum(u.*y,'all'),v);
end

function s = fcn_scalar(x)
% unformatted double scalar from a (possibly formatted, possibly gpu) dlarray
s = double(gather(extractdata(x)));
end
