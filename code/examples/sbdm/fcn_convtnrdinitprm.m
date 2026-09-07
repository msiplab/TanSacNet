function prm = fcn_convtnrdinitprm(nStages,nFilters,filterSize,mode)
%FCN_CONVTNRDINITPRM Initial parameters of the convolutional TNRD baseline
%
%   prm = fcn_convtnrdinitprm(nStages,nFilters,filterSize,mode) initialises
%   the tied-weight TNRD denoiser of Example 10.2 of Muramatsu (2026),
%
%     g(v) = W_a^T*tanh(W_a*v + b_a) + b_s,
%
%   which is used as the baseline of the LSUN realisation. MODE selects the
%   constraint imposed on the dictionary:
%
%     "tied"     no constraint; Parseval tightness is left to the training
%     "parseval" W_a^T W_a = I is maintained by an SVD projection onto the
%                Stiefel manifold, applied periodically during training
%
%   Unlike the LSUN realisation, neither mode makes the adjoint relation
%   exact throughout training: "tied" does not enforce it at all and
%   "parseval" restores it only at the projection steps.
%
% Copyright (c) 2026, Shogo MURAMATSU, All rights reserved.

arguments
    nStages (1,1) double = 1
    nFilters (1,1) double = 32
    filterSize (1,1) double = 5
    mode (1,1) string = "tied"
end

P = nFilters; fs = filterSize; N = fs*fs;
prm.Wa = cell(1,nStages);
prm.ba = cell(1,nStages);
prm.bs = cell(1,nStages);
prm.loglam = cell(1,nStages);
for s = 1:nStages
    switch mode
        case "tied"
            W0 = randn(fs,fs,1,P,'single')/sqrt(P*N);
        case "parseval"
            D1 = single(dctmtx(fs));
            [U0,~,V0] = svd([kron(D1,D1); randn(P-N,N,'single')],'econ');
            W0 = permute(reshape(single(U0*V0'),P,fs,fs,1),[2,3,4,1]);
        otherwise
            error("mode must be ""tied"" or ""parseval""")
    end
    prm.Wa{s} = dlarray(W0);
    prm.ba{s} = dlarray(zeros(P,1,'single'));
    prm.bs{s} = dlarray(zeros(1,1,'single'));
    % the Parseval-tight initialisation has unit operator gain, so the
    % regularisation parameter starts an order of magnitude smaller
    if mode == "parseval"
        prm.loglam{s} = dlarray(single(log(0.1)));
    else
        prm.loglam{s} = dlarray(single(0));
    end
end
end
