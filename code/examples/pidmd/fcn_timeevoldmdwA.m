function res = fcn_timeevoldmdwA(x0,A,nFrames)
%FCN_TIMEEVOLDMD この関数の概要をここに記述
%   詳細説明をここに記述
arguments
    x0 (:,:) {mustBeNumeric}
    A (1,1) 
    nFrames (1,1) {mustBePositive,mustBeInteger}
end

res = zeros(numel(x0),nFrames);
prestate = x0(:);
for iFrame = 1:nFrames-1
    curstate = A(prestate);
    res(:,iFrame+1) = curstate;
    prestate = curstate;
end
end