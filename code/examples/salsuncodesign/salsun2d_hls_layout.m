function L = salsun2d_hls_layout()
%SALSUN2D_HLS_LAYOUT Configuration and parameter layout for salsun2d_hls
%
%   L = salsun2d_hls_layout() returns the fixed network configuration of
%   the HLS implementation and the offsets of each parameter in the flat
%   parameter vector built by salsun2d_pack_params.
%
%   Configuration (as in main_salsun2d.m): stride [4 4], overlapping
%   factor [3 3] (stages h1, h2, v1, v2), no DC leakage, neighbor blocks
%   [3 3], 3 residual blocks, width 2.
%
%   All offsets are 0-based; parameter p of length n occupies
%   w(L.p + (1:n)). Matrices are stored column-major as in MATLAB.
%
% Requirements: MATLAB R2026b
%
% Copyright (c) 2026, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%

%% Configuration
L.Stride = [4 4];
L.NDec = 16;                 % channels per block
L.Ps = 8;                    % symmetric channels
L.Pa = 8;                    % antisymmetric channels
L.Neighbor = [3 3];
L.NResBlocks = 3;
L.Width = 2;
L.Epsilon = 1e-8;            % state standardization
L.LnEpsilon = 1e-5;          % LayerNorm in residual blocks
L.NStages = 4;
% Intermediate stages in analysis order: shift [v h] and target half
% (1: difference, 2: sum), then the same for synthesis
L.Shift    = [0 1; 0 -1; 1 0; -1 0];
L.Target   = [1; 2; 1; 2];
L.SynShift = [0 -1; 0 1; -1 0; 1 0];
L.SynTarget = [1; 2; 1; 2];

% Estimators: 1 = initial rotation, 2..5 = intermediate stages
nEst = 1 + L.NStages;
L.NEst = nEst;
L.EstChannelFirst = [2 9 9 9 9];   % input channels first:last (1-based)
L.EstChannelLast = [16 16 16 16 16];
L.EstNCh = L.EstChannelLast - L.EstChannelFirst + 1;          % 15, 8, ...
L.EstNFeat = L.EstNCh*prod(L.Neighbor);                       % 135, 72, ...
L.EstNHidden = L.Width*L.EstNFeat;                            % 270, 144, ...
L.EstNAngles = [49 28 28 28 28];       % predicted angles
L.EstNZeroPad = [7 0 0 0 0];           % leading zero angles (no DC leakage)
L.EstNAnglesTotal = L.EstNAngles + L.EstNZeroPad;             % 56, 28, ...
L.ThetaOffset = [0 cumsum(L.EstNAnglesTotal(1:end-1))];       % rows in theta store
L.NThetaRows = sum(L.EstNAnglesTotal);                        % 168
L.MaxNFeat = max(L.EstNFeat);
L.MaxNHidden = max(L.EstNHidden);

%% Parameter layout
off = 0;
[L.Cvh,off] = take(off,L.NDec*L.NDec);
[L.Mask,off] = take(off,L.NDec);
[L.V0MusW,off] = take(off,L.Ps);
[L.V0MusU,off] = take(off,L.Pa);
[L.V0tMusW,off] = take(off,L.Ps);
[L.V0tMusU,off] = take(off,L.Pa);
L.StageMus = zeros(1,L.NStages);
L.StageSynMus = zeros(1,L.NStages);
for iStage = 1:L.NStages
    [L.StageMus(iStage),off] = take(off,L.Pa);
    [L.StageSynMus(iStage),off] = take(off,L.Pa);
end
L.Gamma = zeros(nEst,L.NResBlocks);
L.Beta = zeros(nEst,L.NResBlocks);
L.W1 = zeros(nEst,L.NResBlocks);
L.B1 = zeros(nEst,L.NResBlocks);
L.W2 = zeros(nEst,L.NResBlocks);
L.B2 = zeros(nEst,L.NResBlocks);
L.Wo = zeros(1,nEst);
L.Bo = zeros(1,nEst);
for iEst = 1:nEst
    nF = L.EstNFeat(iEst);
    nH = L.EstNHidden(iEst);
    for iRes = 1:L.NResBlocks
        [L.Gamma(iEst,iRes),off] = take(off,nF);
        [L.Beta(iEst,iRes),off] = take(off,nF);
        [L.W1(iEst,iRes),off] = take(off,nH*nF);
        [L.B1(iEst,iRes),off] = take(off,nH);
        [L.W2(iEst,iRes),off] = take(off,nF*nH);
        [L.B2(iEst,iRes),off] = take(off,nF);
    end
    [L.Wo(iEst),off] = take(off,L.EstNAngles(iEst)*nF);
    [L.Bo(iEst),off] = take(off,L.EstNAngles(iEst));
end
L.NParams = off;
end

function [offset,next] = take(offset,n)
next = offset + n;
end
