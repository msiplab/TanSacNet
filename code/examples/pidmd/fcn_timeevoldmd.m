function res = fcn_timeevoldmd(x0,exVals,exVecs,nFrames)
%FCN_TIMEEVOLDMD この関数の概要をここに記述
%   詳細説明をここに記述
arguments
    x0 (:,:) {mustBeNumeric}
    exVals (:,1) {mustBeNumeric}
    exVecs (:,:) {mustBeNumeric}
    nFrames (1,1) {mustBePositive,mustBeInteger}
end

res = zeros(numel(x0),nFrames);
b = exVecs'*x0(:);
precoefs = b;
for iFrame = 1:nFrames-1
    coefs = diag(exVals)*precoefs;
    res(:,iFrame+1) = exVecs*coefs;
    precoefs = coefs;
end
end