function k = fcn_sskeptstats(ac,dc,Z,m,thrList)
%FCN_SSKEPTSTATS Kept-energy statistics of a unitary transform on Z
%
%   k = fcn_sskeptstats(ac,dc,Z,m,thrList)
%   ac, dc  : AC / DC coefficients of Z (dlarray or array)
%   m       : 1 x 1 x nAc zonal AC mask (K = 4)
%   thrList : hard thresholds (tau*sigma), one per entry
%   k.zonalAc        : zonal kept AC energy / AC energy
%   k.hardAc(j)      : energy of AC coefficients with |c| > thrList(j) / AC energy
%   k.hardFrac(j)    : fraction of AC coefficients with |c| > thrList(j)
a = double(gather(dex(ac))); d = double(gather(dex(dc))); z = double(gather(dex(Z)));
eAc = sum(z.^2,'all') - sum(d.^2,'all');      % E unitary
k.zonalAc = sum((double(gather(m)).*a).^2,'all')/eAc;
k.hardAc = zeros(1,numel(thrList)); k.hardFrac = k.hardAc;
for j = 1:numel(thrList)
    M = abs(a) > thrList(j);
    k.hardAc(j) = sum(a(M).^2)/eAc;
    k.hardFrac(j) = mean(M,'all');
end
end

function y = dex(y)
if isdlarray(y), y = extractdata(y); end
end
