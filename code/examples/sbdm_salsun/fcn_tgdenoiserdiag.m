function d = fcn_tgdenoiserdiag(fdenoise,sz,nProbe,nPower,v0)
%FCN_TGDENOISERDIAG fcn_denoiserdiag at a given point v0
%
%   d = fcn_tgdenoiserdiag(fdenoise,sz,nProbe,nPower,v0)
%   COPY of the Jacobian part of examples/sbdm/fcn_denoiserdiag.m (Sensei's,
%   not changed). The only difference: the point v0 where the Jacobian is
%   taken is given by the caller (e.g. a noisy test image), instead of
%   v0 = rand(sz). The probes u, w and the power-iteration start are drawn
%   in the same order and the same way (randn(sz,'single'), CPU stream).
%     d.jacAsym : mean |u'Jw - w'Ju| / (||J'u|| ||w||) over nProbe probes
%     d.jacRho  : spectral radius by power iteration on J' (nPower steps)
%   (Used by main_sbdm_salsun_tnrdguar.m.)

v0 = dlarray(single(gather(v0)),'SSCB');
a = zeros(nProbe,1);
for i = 1:nProbe
    u = dlarray(randn(sz,'single'),'SSCB');
    w = dlarray(randn(sz,'single'),'SSCB');
    Jtu = dlfeval(@vjp,fdenoise,v0,u);
    Jtw = dlfeval(@vjp,fdenoise,v0,w);
    n1 = sc(sum(Jtu.*w,'all'));
    n2 = sc(sum(Jtw.*u,'all'));
    d1 = sc(sum(Jtu.^2,'all'));
    d2 = sc(sum(w.^2,'all'));
    a(i) = abs(n1-n2)/sqrt(d1*d2);
end
d.jacAsym = mean(a);
p = dlarray(randn(sz,'single'),'SSCB');
p = p/sqrt(sc(sum(p.^2,'all')));
rho = 0;
for i = 1:nPower
    q = dlfeval(@vjp,fdenoise,v0,p);
    rho = sqrt(sc(sum(q.^2,'all')));
    p = q/rho;
end
d.jacRho = rho;
end

function g = vjp(fdenoise,v,u)
v = dlarray(v,'SSCB');
y = fdenoise(v);
g = dlgradient(sum(u.*y,'all'),v);
end

function s = sc(x)
s = double(gather(extractdata(x)));
end
