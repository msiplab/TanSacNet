function [g,n] = fcn_tgclipgrad(g,maxNorm)
%FCN_TGCLIPGRAD Global-norm gradient clipping for the TNRD parameter struct
%
%   [g,n] = fcn_tgclipgrad(g,maxNorm)
%   n = sqrt(sum of squares over all fields and cells of g) (before
%   clipping). If n > maxNorm, every gradient is scaled by maxNorm/n, so the
%   global norm becomes maxNorm. maxNorm = inf: no change (n is still
%   returned).
%   (Used by fcn_tgtrainconvtnrdcos.m.)

s = 0;
for f = string(fieldnames(g)).'
    for k = 1:numel(g.(f))
        s = s + double(gather(extractdata(sum(g.(f){k}.^2,'all'))));
    end
end
n = sqrt(s);
if isfinite(maxNorm) && n > maxNorm
    c = single(maxNorm/n);
    for f = string(fieldnames(g)).'
        for k = 1:numel(g.(f))
            g.(f){k} = c*g.(f){k};
        end
    end
end
end
