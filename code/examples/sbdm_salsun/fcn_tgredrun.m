function r = fcn_tgredrun(fbank,sgBank,V,X,Hf,Htf,schedule,eta,beta,nIters,doClip)
%FCN_TGREDRUN Score-based RED point estimate (eps = 0) with a model bank
%
%   r = fcn_tgredrun(fbank,sgBank,V,X,Hf,Htf,schedule,eta,beta,nIters,doClip)
%   The iteration of main_sbdm.m (local fcn_redrun, mode 4 = annealed
%   sigma(t), no Langevin term):
%     z <- z - eta*( H'(Hz - V) + beta*(z - f(z; sigma(t))) ),  z <- clip(z,0,1)
%   started at z = V. f(.; sigma(t)) = fbank{k}, the fixed-sigma model whose
%   training sigma sgBank(k) is nearest to sigma(t) in log scale (the same
%   rule for every method). doClip = false drops the clipping (extra run:
%   the clipping can hide a divergence).
%   Recorded per iteration t: r.res(t) = ||z(t) - z(t-1)||_2 (fixed-point
%   residual, z(0) = V), r.psnr(t), r.k(t) (model used). The run stops
%   early if z is not finite or max|z| > 1e6 (r.blowup = true; the rest
%   of the curves is NaN). r.psnrEnd, r.ssimEnd: final estimate (clipped to
%   [0,1] for the measures only when doClip = false).

[~,k] = min(abs(log(schedule(:)) - log(sgBank(:).')),[],2);
z = V;
r.res = nan(nIters,1); r.psnr = nan(nIters,1); r.k = k(1:nIters);
r.blowup = false; r.itBlowup = NaN;
for it = 1:nIters
    zOld = z;
    gData = Htf(Hf(z) - V);
    gReg = z - fbank{k(it)}(z);
    z = z - eta*(gData + beta*gReg);
    if doClip, z = min(max(z,0),1); end
    r.res(it) = norm(z(:) - zOld(:));
    if ~all(isfinite(z(:))) || max(abs(z(:))) > 1e6
        r.blowup = true; r.itBlowup = it;
        break
    end
    r.psnr(it) = psnr(min(max(z,0),1),X);
end
zc = min(max(z,0),1);
if r.blowup
    r.psnrEnd = NaN; r.ssimEnd = NaN;
else
    r.psnrEnd = psnr(zc,X); r.ssimEnd = ssim(zc,X);
end
r.est = single(zc);
end
