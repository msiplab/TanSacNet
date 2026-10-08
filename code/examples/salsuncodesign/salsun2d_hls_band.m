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
%   The number of lanes nL = gcd(nRows,L.NLanes) must divide the number
%   of block rows of x (band + halo); for 300 x 300 frames, B = 31 gives
%   45 block rows and 15 lanes.
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
Yb = rotateInitialOrFinal(Ya,theta,w,L,false,nRows,nCols);
for iStage = int32(1):int32(L.NStages)
    Ya = atomExtension(Yb,int32(L.Shift(iStage,:)),int32(L.Target(iStage)),nRows,nCols);
    [theta,sum1,sum2] = estimateAngles(theta,sum1,sum2,Ya,w,L,coder.ignoreConst(iStage+1), ...
        mu,sigma,nRows,nCols,sumFirst,rowLast);
    Yb = rotateIntermediate(Ya,theta,w,L,iStage,false,nRows,nCols);
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
    Ya = rotateIntermediate(Yb,theta,w,L,iStage,true,nRows,nCols);
    Yb = atomExtension(Ya,int32(L.SynShift(iStage,:)),int32(L.SynTarget(iStage)),nRows,nCols);
end
Ya = rotateInitialOrFinal(Yb,theta,w,L,true,nRows,nCols);
y = blockIdct(Ya,w,L,nCols,rowFirst,rowLast);
end

%% Block transforms
function Y = blockDct(x,w,L,nRows,nCols)
% n = iv + (ih-1)*My indexes the pixels of a block in column-major order
coder.inline('never')
My = L.Stride(1);
Mx = L.Stride(2);
Y = zeros(L.NDec,nRows,nCols,'single');
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
Z = zeros(size(Y),'single');
for c = 1:nCols
    for r = 1:nRows
        rs = wrap(r-shift(1),nRows);
        cs = wrap(c-shift(2),nCols);
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
function Z = rotateInitialOrFinal(Y,theta,w,L,isFinal,nRows,nCols)
coder.inline('never')
nHalf = L.EstNAnglesTotal(1)/2;
if isFinal
    offMusW = L.V0tMusW;
    offMusU = L.V0tMusU;
else
    offMusW = L.V0MusW;
    offMusU = L.V0MusU;
end
Z = zeros(size(Y),'single');
angles = zeros(nHalf,1,'single');
for c = 1:nCols
    for r = 1:nRows
        for a = 1:nHalf
            angles(a) = theta(a,r,c);
        end
        W = orthMatrix(angles,w,offMusW,L.Ps);
        for a = 1:nHalf
            angles(a) = theta(nHalf+a,r,c);
        end
        U = orthMatrix(angles,w,offMusU,L.Pa);
        for i = 1:L.Ps
            accS = single(0);
            accA = single(0);
            for j = 1:L.Ps
                if isFinal
                    accS = accS + W(j,i)*Y(j,r,c);
                    accA = accA + U(j,i)*Y(L.Ps+j,r,c);
                else
                    accS = accS + W(i,j)*Y(j,r,c);
                    accA = accA + U(i,j)*Y(L.Ps+j,r,c);
                end
            end
            Z(i,r,c) = accS;
            Z(L.Ps+i,r,c) = accA;
        end
    end
end
end

function Z = rotateIntermediate(Y,theta,w,L,iStage,isSynthesis,nRows,nCols)
coder.inline('never')
iEst = iStage + 1;
nAngles = int32(L.EstNAnglesTotal(iEst));
offTheta = int32(L.ThetaOffset(iEst));
if isSynthesis
    offMus = int32(L.StageSynMus(iStage));
else
    offMus = int32(L.StageMus(iStage));
end
Z = Y;
angles = zeros(L.Pa*(L.Pa-1)/2,1,'single');
for c = 1:nCols
    for r = 1:nRows
        for a = 1:nAngles
            angles(a) = theta(offTheta+a,r,c);
        end
        U = orthMatrix(angles,w,offMus,L.Pa);
        for i = 1:L.Pa
            acc = single(0);
            for j = 1:L.Pa
                if isSynthesis
                    acc = acc + U(j,i)*Y(L.Ps+j,r,c);
                else
                    acc = acc + U(i,j)*Y(L.Ps+j,r,c);
                end
            end
            Z(L.Ps+i,r,c) = acc;
        end
    end
end
end

function M = orthMatrix(angles,w,offMus,n)
% Product of Givens rotations followed by sign flips (as fcn_orthmtxgen).
% The cosines and sines are computed first in a pipelined loop. Not
% inlined: one generator serves all rotations, and the name M used by
% the pragma below is kept.
coder.inline('never')
cs = zeros(numel(angles),1,'single');
sn = zeros(numel(angles),1,'single');
for a = 1:numel(angles)
    coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
    cs(a) = cos(angles(a));
    sn(a) = sin(angles(a));
end
M = zeros(n,n,'single');
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=M type=complete dim=0")
for i = 1:n
    M(i,i) = single(1);
end
iAng = int32(1);
for iTop = 1:n-1
    for iBtm = iTop+1:n
        for j = 1:n
            coder.hdl.literaltext("#pragma HLS UNROLL")
            vt = M(iTop,j);
            vb = M(iBtm,j);
            u = sn(iAng)*(vt + vb);
            M(iTop,j) = (cs(iAng) + sn(iAng))*vt - u;
            M(iBtm,j) = (cs(iAng) - sn(iAng))*vb + u;
        end
        iAng = iAng + 1;
    end
end
for i = 1:n
    for j = 1:n
        M(i,j) = w(offMus+i)*M(i,j);
    end
end
end

%% Angle estimator (control path)
function [theta,sum1,sum2] = estimateAngles(theta,sum1,sum2,Y,w,L,iEst,mu,sigma,nRows,nCols,sumFirst,sumLast)
% One column of blocks (nRows blocks, index b) is processed at a time.
% The standardization uses the given statistics mu(:,iEst), sigma(:,iEst)
% per channel; the channel sums over the rows sumFirst..sumLast (the
% band without rows already counted) are accumulated for the statistics
% of the next frame.
coder.inline('never')
ch0 = int32(L.EstChannelFirst(iEst));
nCh = int32(L.EstNCh(iEst));
nF = int32(L.EstNFeat(iEst));
nH = int32(L.EstNHidden(iEst));
nA = int32(L.EstNAngles(iEst));
nZ = int32(L.EstNZeroPad(iEst));
offTheta = int32(L.ThetaOffset(iEst));
nv = L.Neighbor(1);
nh = L.Neighbor(2);

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

% All intermediate arrays have MaxNHidden rows, so that one version of
% each layer function serves every call. They are split cyclically over
% the blocks of the column (dimension 1 in the generated C++) into nL
% banks, one per lane: block b is in bank mod(b-1,nL).
nL = coder.const(gcd(nRows,L.NLanes));
F = zeros(L.MaxNHidden,nRows,'single');
LN = zeros(L.MaxNHidden,nRows,'single');
Z1 = zeros(L.MaxNHidden,nRows,'single');
A = zeros(L.MaxNHidden,nRows,'single');
Z2 = zeros(L.MaxNHidden,nRows,'single');
Th = zeros(L.MaxNHidden,nRows,'single');
coder.hdl.literaltext(coder.const(partitionPragma('F',nL)))
coder.hdl.literaltext(coder.const(partitionPragma('LN',nL)))
coder.hdl.literaltext(coder.const(partitionPragma('Z1',nL)))
coder.hdl.literaltext(coder.const(partitionPragma('A',nL)))
coder.hdl.literaltext(coder.const(partitionPragma('Z2',nL)))
coder.hdl.literaltext(coder.const(partitionPragma('Th',nL)))
for c = 1:nCols
    % Local state of the blocks in column c, standardized with the given statistics
    for r = 1:nRows
        iF = int32(0);
        for vshift = fix(nv/2):-1:-fix(nv/2)
            for hshift = fix(nh/2):-1:-fix(nh/2)
                rs = wrap(r-vshift,nRows);
                cs = wrap(c-hshift,nCols);
                for j = 1:nCh
                    F(iF+j,r) = (Y(ch0+j-1,rs,cs) - mu(ch0+j-1,iEst))/sigma(ch0+j-1,iEst);
                end
                iF = iF + nCh;
            end
        end
    end

    % Residual blocks: LayerNorm, FC, GELU (tanh), FC, skip
    for iRes = 1:L.NResBlocks
        LN = layerNorm(F,w,int32(L.Gamma(iEst,iRes)),int32(L.Beta(iEst,iRes)),nF,L.LnEpsilon,nL);
        Z1 = fullyConnected(LN,w,int32(L.W1(iEst,iRes)),int32(L.B1(iEst,iRes)),nH,nF,nL);
        A = gelu(Z1,nH);
        Z2 = fullyConnected(A,w,int32(L.W2(iEst,iRes)),int32(L.B2(iEst,iRes)),nF,nH,nL);
        for r = 1:nRows
            for i = 1:nF
                F(i,r) = F(i,r) + Z2(i,r);
            end
        end
    end

    % Output angles (leading zeros for no DC leakage)
    Th = fullyConnected(F,w,int32(L.Wo(iEst)),int32(L.Bo(iEst)),nA,nF,nL);
    for r = 1:nRows
        for a = 1:nZ
            theta(offTheta+a,r,c) = single(0);
        end
        for a = 1:nA
            theta(offTheta+nZ+a,r,c) = Th(a,r);
        end
    end
end
end

function Z = fullyConnected(X,w,offW,offB,nOut,nIn,nL)
% Z(o,b) = B(o) + sum_i W(o,i)*X(i,b) for all blocks b of a column.
% Each weight W(o,i) is read once and applied to the blocks in groups of
% nL lanes, one group per cycle. Z(o,b) is updated again only at the next
% i, so the adder latency does not limit the pipeline.
coder.inline('never')
% Only rows 1..nOut are written and read, so Z is not zero-filled
% (a zero fill would cost as many cycles as the layer itself)
Z = coder.nullcopy(zeros(size(X),'single'));
nB = size(X,2);
nG = nB/nL;
for o = 1:nOut
    for b = 1:nB
        Z(o,b) = w(offB+o);
    end
end
for i = 1:nIn
    for o = 1:nOut
        wv = w(offW + o + (i-1)*nOut);
        for g = 1:nG
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            coder.hdl.literaltext("#pragma HLS DEPENDENCE variable=Z type=inter false")
            for lane = 1:nL
                b = (g-1)*nL + lane;
                Z(o,b) = Z(o,b) + wv*X(i,b);
            end
        end
    end
end
end

function Z = layerNorm(X,w,offGamma,offBeta,nF,lnEpsilon,nL)
% LayerNorm over the features of each block. For each group of nL blocks
% the sums run over i with the nL blocks in parallel, keeping the order of
% summation (so the result is bit-identical to salsun2d_hls). Each sum is
% updated once per iteration of i, so the pipeline waits for the adder.
% The normalization uses one divider, pipelined over the blocks.
coder.inline('never')
nB = size(X,2);
nG = nB/nL;
Z = coder.nullcopy(zeros(size(X),'single'));   % rows 1..nF are all written
m = zeros(1,nB,'single');
s = zeros(1,nB,'single');
coder.hdl.literaltext(coder.const(sprintf( ...
    '#pragma HLS ARRAY_PARTITION variable=m type=cyclic factor=%d dim=1',nL)))
coder.hdl.literaltext(coder.const(sprintf( ...
    '#pragma HLS ARRAY_PARTITION variable=s type=cyclic factor=%d dim=1',nL)))
for g = 1:nG
    for i = 1:nF
        coder.hdl.literaltext("#pragma HLS PIPELINE")
        for lane = 1:nL
            b = (g-1)*nL + lane;
            m(b) = m(b) + X(i,b);
        end
    end
end
for b = 1:nB
    m(b) = m(b)/single(nF);
end
for g = 1:nG
    for i = 1:nF
        coder.hdl.literaltext("#pragma HLS PIPELINE")
        for lane = 1:nL
            b = (g-1)*nL + lane;
            d = X(i,b) - m(b);
            s(b) = s(b) + d*d;
        end
    end
end
for b = 1:nB
    s(b) = sqrt(s(b)/single(nF) + single(lnEpsilon));
end
for b = 1:nB
    for i = 1:nF
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        Z(i,b) = w(offGamma+i)*((X(i,b) - m(b))/s(b)) + w(offBeta+i);
    end
end
end

function A = gelu(Z,nH)
coder.inline('never')
nB = size(Z,2);
A = coder.nullcopy(zeros(size(Z),'single'));   % rows 1..nH are all written
for h = 1:nH
    for b = 1:nB
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        v = Z(h,b);
        A(h,b) = single(0.5)*v*(single(1) + ...
            tanh(single(sqrt(2/pi))*(v + single(0.044715)*v*v*v)));
    end
end
end

function i = wrap(i,n)
% 1-based circular index
i = mod(i-1,n) + 1;
end

function str = partitionPragma(name,nL)
% Cyclic split over the blocks of a column (dimension 1 in C++)
str = sprintf('#pragma HLS ARRAY_PARTITION variable=%s type=cyclic factor=%d dim=1',name,nL);
end
