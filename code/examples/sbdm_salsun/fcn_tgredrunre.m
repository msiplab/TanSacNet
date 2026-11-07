function r = fcn_tgredrunre(fbank,remake,sgBank,V,X,Hf,Htf,schedule,eta,beta,nIters,doClip,K)
%FCN_TGREDRUNRE fcn_tgredrun + re-estimation of the SA angles every K its
%
%   r = fcn_tgredrunre(fbank,remake,sgBank,V,X,Hf,Htf,schedule,eta,beta,nIters,doClip,K)
%   COPY of fcn_tgredrun.m (the RED iteration of Part 3, not changed) with
%   one change: the denoiser of the current sigma model can be rebuilt with
%   angles estimated from the current iterate z.
%     fbank{k}    : denoiser of model k with the angles estimated from V
%                   (as Part 3).
%     remake(k,z) : denoiser of model k with the angles estimated from z
%                   (then constant until the next re-estimation).
%     K = inf     : no re-estimation; identical to fcn_tgredrun (each model
%                   uses fbank{k}, angles from V).
%     K finite    : before step t the angles are re-estimated from the
%                   current iterate z(t-1) if the sigma model switches at t
%                   (t > 1), or if t - tLast >= K (tLast = the step of the
%                   last estimation). Step 1 uses fbank{k(1)} (z(0) = V, so
%                   this is the estimate from z(0)).
%   Within one step the angles are constant.
%   Extra fields: r.nRe (number of re-estimations), r.itRe (their steps),
%   r.tRe (time spent in remake, s).

[~,k] = min(abs(log(schedule(:)) - log(sgBank(:).')),[],2);
z = V;
r.res = nan(nIters,1); r.psnr = nan(nIters,1); r.k = k(1:nIters);
r.blowup = false; r.itBlowup = NaN;
r.nRe = 0; r.itRe = zeros(1,0); r.tRe = 0;
kCur = 0; tLast = -inf; f = [];
for it = 1:nIters
    if k(it) ~= kCur
        if it == 1 || isinf(K)
            f = fbank{k(it)};
        else
            tr = tic; f = remake(k(it),z); r.tRe = r.tRe + toc(tr);
            r.nRe = r.nRe + 1; r.itRe(end+1) = it;
        end
        kCur = k(it); tLast = it;
    elseif isfinite(K) && it - tLast >= K
        tr = tic; f = remake(k(it),z); r.tRe = r.tRe + toc(tr);
        r.nRe = r.nRe + 1; r.itRe(end+1) = it;
        tLast = it;
    end
    zOld = z;
    gData = Htf(Hf(z) - V);
    gReg = z - f(z);
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
end
