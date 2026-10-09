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
% Parallel lanes of the estimators (salsun2d_hls_opt): the blocks of a
% column are processed by gcd(nRows,NLanes) lanes (15 for 300 x 300
% frames). With 75 lanes the whole-frame build failed in routing.
L.NLanes = 15;
% Band design (salsun2d_hls_band): the estimators process a group of
% gcd(nCols,NColGroup) columns at a time with one lane per block in the
% fully connected layers (30 block rows = 30 lanes for 300 x 300 frames
% with 16-row bands), NLnLanes blocks per cycle in the LayerNorm and
% residual add, and NGeluUnits blocks per cycle in the GELU. With
% NColGroup = 3 (90 lanes) the layers with fewer lanes reach blocks all
% over the die: routing congestion level 7 in SLR0, and the DDR clocks of
% the shell failed timing.
L.NColGroup = 1;
L.NLnLanes = 15;
L.NGeluUnits = 3;
% Fixed-point estimators of the band design, [word fraction] lengths,
% chosen with evaluate_salsun2d_fixed_estimator(Common=true) on the
% trained network (MSE +0.01%): one format for the inputs of the fully
% connected layers (largest magnitude below 32, one bit of headroom), one for
% all weight matrices (largest magnitude 0.84, range +-2), exact
% accumulators (18 x 18-bit products, up to 270 terms), and the sums of
% squares of the LayerNorm. 18 bits match the DSP multipliers (27 x 18)
% and the block RAM widths. L.FcInputs products per lane and cycle.
L.FixSignal = [18 11];
L.FixWeight = [18 16];
L.FixAcc = [48 27];
L.FixSq = [48 22];
L.FcInputs = 4;
% Halo of the band-wise streaming design (salsun2d_hls_band): the
% structural receptive field in block rows, 6 for the analysis (five
% estimators with 3 x 3 neighbors and the two vertical atom extensions)
% plus 1 for the synthesis. With this halo the rows of a band equal
% whole-frame processing with the same statistics.
L.Halo = 7;
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
% Band design: with a coefficient mask that keeps only the first
% antisymmetric channel (Ps+1) after the last stage, only the first row
% of the last rotation U = D*G_K*...*G_1 matters, and the Givens rotations
% G_k on pairs (i,j) with i > 1 do not change that row. Those are all
% but the first Pa-1 rotations (pairs (1,2)..(1,Pa)), so the last
% estimator predicts only its first Pa-1 angles. salsun2d_check_band_mask
% checks the mask.
L.LastStageNAngles = L.Pa - 1;                                % 7 of 28
% Pairs of the Givens rotations in the order of fcn_orthmtxgen
% (1,2),(1,3),...,(1,n),(2,3),...,(n-1,n), for n = Ps = Pa
[L.GivensTop,L.GivensBottom] = givensPairs(L.Pa);
L.EstNAnglesUsed = L.EstNAngles;
L.EstNAnglesUsed(end) = L.LastStageNAngles;
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

function [tops,btms] = givensPairs(n)
tops = zeros(1,n*(n-1)/2);
btms = zeros(1,n*(n-1)/2);
k = 0;
for iTop = 1:n-1
    for iBtm = iTop+1:n
        k = k + 1;
        tops(k) = iTop;
        btms(k) = iBtm;
    end
end
end
