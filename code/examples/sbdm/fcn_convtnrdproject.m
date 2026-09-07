function prm = fcn_convtnrdproject(prm)
%FCN_CONVTNRDPROJECT Project the TNRD dictionary onto the Stiefel manifold
%
%   prm = fcn_convtnrdproject(prm) replaces every filter bank W_a by the
%   nearest matrix with orthonormal columns, W_a <- U*V^T where U*S*V^T is
%   the thin SVD of W_a. This is the projection used by the "parseval"
%   dictionary of Example 10.2 of Muramatsu (2026) to keep the frame tight.
%
%   The LSUN realisation needs no counterpart: its rotation-angle
%   parameterisation is unitary for every parameter value.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

for s = 1:numel(prm.Wa)
    W = single(extractdata(prm.Wa{s}));
    [fs1,fs2,~,P] = size(W);
    N = fs1*fs2;
    Wm = reshape(permute(W,[4,1,2,3]),P,N);
    [U,~,V] = svd(Wm,'econ');
    prm.Wa{s} = dlarray(permute(reshape(single(U*V'),P,fs1,fs2,1),[2,3,4,1]));
end
end
