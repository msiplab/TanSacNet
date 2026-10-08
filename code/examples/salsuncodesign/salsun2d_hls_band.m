function [y,sum1,sum2] = salsun2d_hls_band(x,w,mu,sigma,skipRows) %#codegen
%SALSUN2D_HLS_BAND SA-LSUN 2-D analysis and synthesis of one band, for HLS
%
%   [y,sum1,sum2] = salsun2d_hls_band(x,w,mu,sigma,skipRows) processes one band
%   of a frame in the streaming (overlap-save) form of
%   salsun2d_infer_stream:
%
%   x      - the band with its halo, (B + 2*L.Halo)*My x szx single:
%            L.Halo block rows of context above and below the band
%            (circular at the frame boundary), B block rows of the band.
%   w      - packed parameters (salsun2d_pack_params).
%   mu, sigma - standardization statistics per channel and estimator,
%            L.NDec x L.NEst single, from the previous frames (the
%            channel statistics equal the per-feature statistics of the
%            network, since the features are shifted channels).
%   y      - reconstructed rows of the band only, B*My x szx.
%   skipRows - int32, number of leading rows of the band whose blocks
%            are left out of the sums (rows already counted by the
%            previous band when the last band of a frame is moved up to
%            end at the frame boundary); 0 otherwise.
%   sum1, sum2 - L.NDec x L.NEst sums and sums of squares of the
%            estimator input channels over the valid blocks of the band
%            (minus the skipped rows), to be accumulated over the bands
%            of a frame for the exact statistics of the next frame.
%
%   The halo L.Halo = 7 is the structural receptive field (6 block rows
%   for the five estimators and the two vertical atom extensions, plus 1
%   for the synthesis), so the rows of the band are exactly those of
%   whole-frame processing with the same statistics (salsun2d_hls_opt
%   with those statistics), up to rounding of the accumulated sums.
%
%   Compared with salsun2d_hls_opt: no whole-frame buffers, the angles of
%   the band stay on chip (no DDR round trip), the statistics are given
%   instead of measured on the image, and the output is only the band.
%   The computations of each block are the same, in the same order.
%
%   The estimators process gcd(nCols,L.NColGroup) columns at a time with
%   one lane per block (see salsun2d_hls_layout); for 300 x 300 frames,
%   B = 16 gives 30 block rows and 3 x 30 = 90 lanes.
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

coder.extrinsic('salsun2d_hls_layout');
L = coder.const(salsun2d_hls_layout());
[szy,szx] = size(x);
nRows = szy/L.Stride(1);          % block rows of the band with halo
nCols = szx/L.Stride(2);
H = int32(L.Halo);
rowFirst = H + 1;                  % valid rows of the band
rowLast = int32(nRows) - H;
sumFirst = rowFirst + int32(skipRows);   % first row counted in the sums

% Angles of the band with halo, on chip. URAM words are 72 bits wide, so
% two 32-bit values are packed per word (ARRAY_RESHAPE on the innermost
% C++ dimension), which halves the URAM count.
theta = zeros(L.NThetaRows,nRows,nCols,'single');
coder.hdl.literaltext("#pragma HLS ARRAY_RESHAPE variable=theta type=cyclic factor=2 dim=3")
coder.hdl.literaltext("#pragma HLS BIND_STORAGE variable=theta type=ram_2p impl=uram")
sum1 = zeros(L.NDec,L.NEst,'single');
sum2 = zeros(L.NDec,L.NEst,'single');

%% Analysis (Ya and Yb alternate)
coder.hdl.literaltext("#pragma HLS ARRAY_RESHAPE variable=Ya type=cyclic factor=2 dim=3")
coder.hdl.literaltext("#pragma HLS ARRAY_RESHAPE variable=Yb type=cyclic factor=2 dim=3")
coder.hdl.literaltext("#pragma HLS BIND_STORAGE variable=Ya type=ram_2p impl=uram")
coder.hdl.literaltext("#pragma HLS BIND_STORAGE variable=Yb type=ram_2p impl=uram")
Ya = blockDct(x,w,L,nRows,nCols);
[theta,sum1,sum2] = estimateAngles(theta,sum1,sum2,Ya,w,L,coder.ignoreConst(int32(1)), ...
    mu,sigma,nRows,nCols,sumFirst,rowLast);
Yb = rotateInitialOrFinal(Ya,theta,w,L,coder.ignoreConst(false),nRows,nCols);
for iStage = int32(1):int32(L.NStages)
    Ya = atomExtension(Yb,int32(L.Shift(iStage,:)),int32(L.Target(iStage)),nRows,nCols);
    [theta,sum1,sum2] = estimateAngles(theta,sum1,sum2,Ya,w,L,coder.ignoreConst(iStage+1), ...
        mu,sigma,nRows,nCols,sumFirst,rowLast);
    Yb = rotateIntermediate(Ya,theta,w,L,iStage,coder.ignoreConst(false),nRows,nCols);
end

%% Coefficient mask
for c = 1:nCols
    for r = 1:nRows
        for k = 1:L.NDec
            Yb(k,r,c) = Yb(k,r,c)*w(L.Mask+k);
        end
    end
end

%% Synthesis (the band rows need the coefficients one block around them)
for iStage = int32(L.NStages):-1:int32(1)
    Ya = rotateIntermediate(Yb,theta,w,L,iStage,coder.ignoreConst(true),nRows,nCols);
    Yb = atomExtension(Ya,int32(L.SynShift(iStage,:)),int32(L.SynTarget(iStage)),nRows,nCols);
end
Ya = rotateInitialOrFinal(Yb,theta,w,L,coder.ignoreConst(true),nRows,nCols);
y = blockIdct(Ya,w,L,nCols,rowFirst,rowLast);
end

%% Block transforms
function Y = blockDct(x,w,L,nRows,nCols)
% n = iv + (ih-1)*My indexes the pixels of a block in column-major order
coder.inline('never')
My = L.Stride(1);
Mx = L.Stride(2);
Y = coder.nullcopy(zeros(L.NDec,nRows,nCols,'single'));   % all elements are written
for c = 1:nCols
    for r = 1:nRows
        for k = 1:L.NDec
            acc = single(0);
            for ih = 1:Mx
                for iv = 1:My
                    n = iv + (ih-1)*My;
                    acc = acc + w(L.Cvh + k + (n-1)*L.NDec)*x((r-1)*My+iv,(c-1)*Mx+ih);
                end
            end
            Y(k,r,c) = acc;
        end
    end
end
end

function x = blockIdct(Y,w,L,nCols,rowFirst,rowLast)
% Only the block rows rowFirst..rowLast (the band) are reconstructed
coder.inline('never')
My = L.Stride(1);
Mx = L.Stride(2);
nOut = rowLast - rowFirst + 1;
x = zeros(nOut*My,nCols*Mx,'single');
for c = 1:nCols
    for r = rowFirst:rowLast
        ro = r - rowFirst;
        for ih = 1:Mx
            for iv = 1:My
                n = iv + (ih-1)*My;
                acc = single(0);
                for k = 1:L.NDec
                    acc = acc + w(L.Cvh + k + (n-1)*L.NDec)*Y(k,r,c);
                end
                x(ro*My+iv,(c-1)*Mx+ih) = acc;
            end
        end
    end
end
end

%% Atom extension
function Z = atomExtension(Y,shift,target,nRows,nCols)
coder.inline('never')
ps = size(Y,1)/2;
Z = coder.nullcopy(zeros(size(Y),'single'));   % all elements are written
for c = 1:nCols
    for r = 1:nRows
        rs = wrap(int32(r)-shift(1),int32(nRows));
        cs = wrap(int32(c)-shift(2),int32(nCols));
        for j = 1:ps
            if target == 1
                ys = Y(j,r,c) + Y(ps+j,r,c);
                yd = Y(j,rs,cs) - Y(ps+j,rs,cs);
            else
                ys = Y(j,rs,cs) + Y(ps+j,rs,cs);
                yd = Y(j,r,c) - Y(ps+j,r,c);
            end
            Z(j,r,c) = single(0.5)*(ys + yd);
            Z(ps+j,r,c) = single(0.5)*(ys - yd);
        end
    end
end
end

%% Rotations
% The rotations are applied to the coefficients of a block directly as
% the sequence of Givens rotations, without forming the matrices: for
% U = D*G_K*...*G_1 (D the sign flips mus), the analysis computes
% D*(G_K*(...*(G_1*y))) and the synthesis U.'*y = G_1.'*(...*(G_K.'*(D*y))),
% K multiply-add pairs per block instead of the K x n matrix updates of
% the generator and the n x n product. G_k rotates the pair (i,j) of
% L.GivensTop(k), L.GivensBottom(k) as [c -s; s c] (as fcn_orthmtxgen).
% For each column, the cosines and sines of all blocks are computed
% first (one angle per cycle), and the blocks are then rotated in a
% pipeline with one rotation per cycle on average.

function Z = rotateInitialOrFinal(Y,theta,w,L,isFinal,nRows,nCols)
% Initial rotation (W on the symmetric, U on the antisymmetric channels)
% or its transpose (isFinal). The callers pass isFinal with
% coder.ignoreConst, so that the analysis and the synthesis share one
% instance of the function.
coder.inline('never')
nHalf = coder.const(L.EstNAnglesTotal(1)/2);   % angles of W and of U (28)
if isFinal
    offMusW = L.V0tMusW;
    offMusU = L.V0tMusU;
else
    offMusW = L.V0MusW;
    offMusU = L.V0MusU;
end
tops = coder.const(L.GivensTop);
btms = coder.const(L.GivensBottom);
Z = coder.nullcopy(zeros(size(Y),'single'));   % all elements are written
musW = zeros(L.Ps,1,'single');
musU = zeros(L.Pa,1,'single');
for i = 1:L.Ps
    musW(i) = w(offMusW+i);
    musU(i) = w(offMusU+i);
end
csW = zeros(nHalf,nRows,'single');
snW = zeros(nHalf,nRows,'single');
csU = zeros(nHalf,nRows,'single');
snU = zeros(nHalf,nRows,'single');
vs = zeros(L.Ps,1,'single');
va = zeros(L.Pa,1,'single');
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=musW type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=musU type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=vs type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=va type=complete dim=1")
for c = 1:nCols
    for r = 1:nRows
        for a = 1:nHalf
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            tw = theta(a,r,c);
            tu = theta(nHalf+a,r,c);
            csW(a,r) = cos(tw);
            snW(a,r) = sin(tw);
            csU(a,r) = cos(tu);
            snU(a,r) = sin(tu);
        end
    end
    for r = 1:nRows
        coder.hdl.literaltext(coder.const(sprintf('#pragma HLS PIPELINE II=%d',nHalf)))
        for j = 1:L.Ps
            vs(j) = Y(j,r,c);
            va(j) = Y(L.Ps+j,r,c);
        end
        if isFinal
            for j = 1:L.Ps
                vs(j) = musW(j)*vs(j);
                va(j) = musU(j)*va(j);
            end
            for a = nHalf:-1:1
                [vs(tops(a)),vs(btms(a))] = givensT(vs(tops(a)),vs(btms(a)),csW(a,r),snW(a,r));
                [va(tops(a)),va(btms(a))] = givensT(va(tops(a)),va(btms(a)),csU(a,r),snU(a,r));
            end
        else
            for a = 1:nHalf
                [vs(tops(a)),vs(btms(a))] = givens(vs(tops(a)),vs(btms(a)),csW(a,r),snW(a,r));
                [va(tops(a)),va(btms(a))] = givens(va(tops(a)),va(btms(a)),csU(a,r),snU(a,r));
            end
            for j = 1:L.Ps
                vs(j) = musW(j)*vs(j);
                va(j) = musU(j)*va(j);
            end
        end
        for j = 1:L.Ps
            Z(j,r,c) = vs(j);
            Z(L.Ps+j,r,c) = va(j);
        end
    end
end
end

function Z = rotateIntermediate(Y,theta,w,L,iStage,isSynthesis,nRows,nCols)
% Rotation U of the antisymmetric channels of an intermediate stage, or
% its transpose (isSynthesis; passed with coder.ignoreConst by the
% callers, so that one instance serves both). In the last stage only the
% first L.LastStageNAngles rotations are applied: with the mask checked
% by salsun2d_check_band_mask, only channel Ps+1 of the analysis output
% is kept (the other outputs are set to zero, as the mask would), and
% the synthesis input is zero in the other antisymmetric channels, on
% which the remaining rotations act.
coder.inline('never')
iEst = iStage + 1;
offTheta = int32(L.ThetaOffset(iEst));
if isSynthesis
    offMus = int32(L.StageSynMus(iStage));
else
    offMus = int32(L.StageMus(iStage));
end
nAng = coder.const(L.Pa*(L.Pa-1)/2);
isLast = iStage == int32(L.NStages);
if isLast
    nRot = int32(L.LastStageNAngles);
else
    nRot = int32(nAng);
end
tops = coder.const(L.GivensTop);
btms = coder.const(L.GivensBottom);
Z = Y;
mus = zeros(L.Pa,1,'single');
for i = 1:L.Pa
    mus(i) = w(offMus+i);
end
cs = zeros(nAng,nRows,'single');
sn = zeros(nAng,nRows,'single');
va = zeros(L.Pa,1,'single');
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=mus type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=va type=complete dim=1")
for c = 1:nCols
    for r = 1:nRows
        for a = 1:nAng
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            t = theta(offTheta+a,r,c);
            cs(a,r) = cos(t);
            sn(a,r) = sin(t);
        end
    end
    for r = 1:nRows
        coder.hdl.literaltext(coder.const(sprintf('#pragma HLS PIPELINE II=%d',nAng)))
        for j = 1:L.Pa
            va(j) = Y(L.Ps+j,r,c);
        end
        if isSynthesis
            for j = 1:L.Pa
                va(j) = mus(j)*va(j);
            end
            for a = nAng:-1:1
                if a <= nRot
                    [va(tops(a)),va(btms(a))] = givensT(va(tops(a)),va(btms(a)),cs(a,r),sn(a,r));
                end
            end
        else
            for a = 1:nAng
                if a <= nRot
                    [va(tops(a)),va(btms(a))] = givens(va(tops(a)),va(btms(a)),cs(a,r),sn(a,r));
                end
            end
            for j = 1:L.Pa
                va(j) = mus(j)*va(j);
            end
            if isLast
                for j = 2:L.Pa
                    va(j) = single(0);
                end
            end
        end
        for j = 1:L.Pa
            Z(L.Ps+j,r,c) = va(j);
        end
    end
end
end

function [t,b] = givens(t0,b0,c,s)
% [t; b] = [c -s; s c]*[t0; b0]
t = c*t0 - s*b0;
b = s*t0 + c*b0;
end

function [t,b] = givensT(t0,b0,c,s)
% [t; b] = [c -s; s c].'*[t0; b0]
t = c*t0 + s*b0;
b = c*b0 - s*t0;
end

%% Angle estimation
function [theta,sum1,sum2] = estimateAngles(theta,sum1,sum2,Y,w,L,iEst,mu,sigma,nRows,nCols,sumFirst,sumLast)
% A group of nCG = gcd(nCols,L.NColGroup) columns of blocks is processed
% at a time: nB = nCG*nRows blocks, block (r,c0+cc-1) at index
% b = (cc-1)*nRows + r. The fully connected layers have one lane per
% block (nB parallel multiply-adds); the other layers use fewer lanes
% (L.NLnLanes, L.NGeluUnits), as they are a small part of the work.
% The standardization uses the given statistics mu(:,iEst), sigma(:,iEst)
% per channel; the channel sums over the rows sumFirst..sumLast (the
% band without rows already counted) are accumulated for the statistics
% of the next frame.
coder.inline('never')
ch0 = int32(L.EstChannelFirst(iEst));
nCh = int32(L.EstNCh(iEst));
nF = int32(L.EstNFeat(iEst));
nH = int32(L.EstNHidden(iEst));
nA = int32(L.EstNAngles(iEst));          % predicted angles
nAU = int32(L.EstNAnglesUsed(iEst));     % angles computed (7 in the last stage)
nZ = int32(L.EstNZeroPad(iEst));
offTheta = int32(L.ThetaOffset(iEst));
nv = L.Neighbor(1);
nh = L.Neighbor(2);
nCG = coder.const(gcd(nCols,L.NColGroup));
nB = coder.const(nCG*nRows);

% Channel sums over the valid blocks of the band (statistics of the next frame)
for j = 1:nCh
    acc1 = single(0);
    acc2 = single(0);
    for c = 1:nCols
        for r = sumFirst:sumLast
            v = Y(ch0+j-1,r,c);
            acc1 = acc1 + v;
            acc2 = acc2 + v*v;
        end
    end
    sum1(ch0+j-1,iEst) = sum1(ch0+j-1,iEst) + acc1;
    sum2(ch0+j-1,iEst) = sum2(ch0+j-1,iEst) + acc2;
end

% Reciprocals of the scales: the standardization is then a multiplication
invSigma = zeros(L.NDec,1,'single');
for j = 1:nCh
    coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
    invSigma(j) = single(1)/sigma(ch0+j-1,iEst);
end

% All intermediate arrays have MaxNHidden rows, so that one version of
% each layer function serves every call. They are split completely over
% the blocks (dimension 1 in the generated C++): one bank per block.
F = zeros(L.MaxNHidden,nB,'single');
LN = zeros(L.MaxNHidden,nB,'single');
Z1 = zeros(L.MaxNHidden,nB,'single');
A = zeros(L.MaxNHidden,nB,'single');
Z2 = zeros(L.MaxNHidden,nB,'single');
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=F type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=LN type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=Z1 type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=A type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=Z2 type=complete dim=1")
for c0 = 1:nCG:nCols
    % Local state of the blocks of the column group, standardized with
    % the given statistics (one neighbor value per cycle)
    for cc = 1:nCG
        c = c0 + cc - 1;
        for r = 1:nRows
            b = (cc-1)*nRows + r;
            for iv = 1:nv
                for ih = 1:nh
                    for j = 1:nCh
                        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
                        vshift = int32(fix(nv/2)) - int32(iv-1);
                        hshift = int32(fix(nh/2)) - int32(ih-1);
                        rs = wrap(int32(r)-vshift,int32(nRows));
                        cs = wrap(int32(c)-hshift,int32(nCols));
                        iF = ((iv-1)*nh + (ih-1))*nCh + j;
                        F(iF,b) = (Y(ch0+j-1,rs,cs) - mu(ch0+j-1,iEst))*invSigma(j);
                    end
                end
            end
        end
    end

    % Residual blocks: LayerNorm, FC, GELU (tanh), FC, skip
    for iRes = 1:L.NResBlocks
        LN = layerNorm(F,w,int32(L.Gamma(iEst,iRes)),int32(L.Beta(iEst,iRes)),nF,L.LnEpsilon,L);
        Z1 = fullyConnected(LN,w,int32(L.W1(iEst,iRes)),int32(L.B1(iEst,iRes)),nH,nF);
        A = gelu(Z1,nH,L);
        Z2 = fullyConnected(A,w,int32(L.W2(iEst,iRes)),int32(L.B2(iEst,iRes)),nF,nH);
        F = residualAdd(F,Z2,nF,L);
    end

    % Output angles (leading zeros for no DC leakage). In the last stage
    % only the first nAU angles are used; all nA rows are still computed,
    % because the pipeline of fullyConnected needs nOut >= 28 (and the
    % output layer is 3% of the multiply-adds)
    Z1 = fullyConnected(F,w,int32(L.Wo(iEst)),int32(L.Bo(iEst)),nA,nF);
    for cc = 1:nCG
        c = c0 + cc - 1;
        for r = 1:nRows
            b = (cc-1)*nRows + r;
            for a = 1:nZ
                coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
                theta(offTheta+a,r,c) = single(0);
            end
            for a = 1:nAU
                coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
                theta(offTheta+nZ+a,r,c) = Z1(a,b);
            end
        end
    end
end
end

function Z = fullyConnected(X,w,offW,offB,nOut,nIn)
% Z(o,b) = B(o) + sum_i W(o,i)*X(i,b) for all blocks b of the group.
% One weight is read per cycle and applied to all blocks at once (one
% lane per block). The loops over i and o are flattened into one
% pipeline; Z(o,b) is updated again only nOut cycles later, which is
% more than the adder latency for every layer (nOut >= 28; smaller nOut
% would break the pipeline), so the
% dependence through Z is declared false.
coder.inline('never')
% Only rows 1..nOut are written and read, so Z is not zero-filled
Z = coder.nullcopy(zeros(size(X),'single'));
nB = size(X,2);
for o = 1:nOut
    coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
    bv = w(offB+o);
    for b = 1:nB
        Z(o,b) = bv;
    end
end
for i = 1:nIn
    for o = 1:nOut
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        coder.hdl.literaltext("#pragma HLS DEPENDENCE variable=Z type=inter false")
        wv = w(offW + o + (i-1)*nOut);
        for b = 1:nB
            Z(o,b) = Z(o,b) + wv*X(i,b);
        end
    end
end
end

function F = residualAdd(F,Z2,nF,L)
% F(i,b) = F(i,b) + Z2(i,b), L.NLnLanes blocks per cycle
coder.inline('never')
nB = size(F,2);
nL = coder.const(gcd(nB,L.NLnLanes));
nG = nB/nL;
for i = 1:nF
    for g = 1:nG
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        for lane = 1:nL
            b = (g-1)*nL + lane;
            F(i,b) = F(i,b) + Z2(i,b);
        end
    end
end
end

function Z = layerNorm(X,w,offGamma,offBeta,nF,lnEpsilon,L)
% LayerNorm over the features of each block, nL = L.NLnLanes blocks at a
% time. The sums over the features of a block are sequential (the
% pipeline waits for the adder); the normalization multiplies by the
% reciprocal of the standard deviation (one divider, one square root).
coder.inline('never')
nB = size(X,2);
nL = coder.const(gcd(nB,L.NLnLanes));
nG = nB/nL;
Z = coder.nullcopy(zeros(size(X),'single'));   % rows 1..nF are all written
m = zeros(1,nB,'single');
s = zeros(1,nB,'single');
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=m type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=s type=complete dim=1")
for g = 1:nG
    for i = 1:nF
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        for lane = 1:nL
            b = (g-1)*nL + lane;
            m(b) = m(b) + X(i,b);
        end
    end
end
for b = 1:nB
    coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
    m(b) = m(b)/single(nF);
end
for g = 1:nG
    for i = 1:nF
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        for lane = 1:nL
            b = (g-1)*nL + lane;
            d = X(i,b) - m(b);
            s(b) = s(b) + d*d;
        end
    end
end
for b = 1:nB
    coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
    s(b) = single(1)/sqrt(s(b)/single(nF) + single(lnEpsilon));
end
for i = 1:nF
    for g = 1:nG
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        gam = w(offGamma+i);
        bet = w(offBeta+i);
        for lane = 1:nL
            b = (g-1)*nL + lane;
            Z(i,b) = gam*((X(i,b) - m(b))*s(b)) + bet;
        end
    end
end
end

function A = gelu(Z,nH,L)
% GELU (tanh form), nU = L.NGeluUnits blocks per cycle
coder.inline('never')
nB = size(Z,2);
nU = coder.const(gcd(nB,L.NGeluUnits));
nG = nB/nU;
A = coder.nullcopy(zeros(size(Z),'single'));   % rows 1..nH are all written
for h = 1:nH
    for g = 1:nG
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        for u = 1:nU
            b = (g-1)*nU + u;
            v = Z(h,b);
            A(h,b) = single(0.5)*v*(single(1) + ...
                tanh(single(sqrt(2/pi))*(v + single(0.044715)*v*v*v)));
        end
    end
end
end

function i = wrap(i,n)
% 1-based circular index for shifts smaller than n, in integer arithmetic
% (mod on doubles becomes a long fmod in HLS)
if i < 1
    i = i + n;
elseif i > n
    i = i - n;
end
end
