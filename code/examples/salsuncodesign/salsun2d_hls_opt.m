function [y,theta] = salsun2d_hls_opt(x,w,theta) %#codegen
%SALSUN2D_HLS_OPT SA-LSUN 2-D analysis and synthesis optimized for HLS
%
%   [y,theta] = salsun2d_hls_opt(x,w,theta) computes the same y as
%   salsun2d_hls(x,w), restructured for the FPGA. theta is a work buffer
%   of size L.NThetaRows x szy/My x szx/Mx (L = salsun2d_hls_layout)
%   that receives the estimated angles; its input values are not used.
%
%   - The estimated angles (3.8 MB for 300 x 300 frames) are written by
%     the analysis side and read once by the synthesis side. They are
%     kept in theta, which the kernel places in DDR: on chip they filled
%     a whole SLR of URAM, and the paths to them limited routing and
%     timing. Using the same name for input and output makes the
%     generated code update theta in place.
%
%   - The angle estimators process one column of blocks (nRows blocks)
%     at a time. The fully connected layers read one weight per cycle and
%     apply it to all blocks of the column in parallel.
%   - The estimator and the fully connected layer are not inlined, so one
%     piece of hardware serves all five estimators and all their layers.
%   - The coefficients alternate between two buffers instead of a copy
%     per processing step.
%
%   The HLS pragmas are written here with coder.hdl.literaltext, so that
%   the generated C++ needs no manual edits.
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

% The layout is evaluated in MATLAB at code generation and folded into
% constants
coder.extrinsic('salsun2d_hls_layout');
L = coder.const(salsun2d_hls_layout());
[szy,szx] = size(x);
nRows = szy/L.Stride(1);
nCols = szx/L.Stride(2);


%% Analysis (Ya and Yb alternate)
Ya = blockDct(x,w,L,nRows,nCols);
% coder.ignoreConst keeps one estimator for all five calls
theta = estimateAngles(theta,Ya,w,L,coder.ignoreConst(int32(1)),nRows,nCols);
Yb = rotateInitialOrFinal(Ya,theta,w,L,false,nRows,nCols);
for iStage = int32(1):int32(L.NStages)
    Ya = atomExtension(Yb,int32(L.Shift(iStage,:)),int32(L.Target(iStage)),nRows,nCols);
    theta = estimateAngles(theta,Ya,w,L,coder.ignoreConst(iStage+1),nRows,nCols);
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

%% Synthesis
for iStage = int32(L.NStages):-1:int32(1)
    Ya = rotateIntermediate(Yb,theta,w,L,iStage,true,nRows,nCols);
    Yb = atomExtension(Ya,int32(L.SynShift(iStage,:)),int32(L.SynTarget(iStage)),nRows,nCols);
end
Ya = rotateInitialOrFinal(Yb,theta,w,L,true,nRows,nCols);
y = blockIdct(Ya,w,L,nRows,nCols);
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

function x = blockIdct(Y,w,L,nRows,nCols)
coder.inline('never')
My = L.Stride(1);
Mx = L.Stride(2);
x = zeros(nRows*My,nCols*Mx,'single');
for c = 1:nCols
    for r = 1:nRows
        for ih = 1:Mx
            for iv = 1:My
                n = iv + (ih-1)*My;
                acc = single(0);
                for k = 1:L.NDec
                    acc = acc + w(L.Cvh + k + (n-1)*L.NDec)*Y(k,r,c);
                end
                x((r-1)*My+iv,(c-1)*Mx+ih) = acc;
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
function theta = estimateAngles(theta,Y,w,L,iEst,nRows,nCols)
% One column of blocks (nRows blocks, index b) is processed at a time
coder.inline('never')
% Sizes and offsets are int32 so that indexing uses integer arithmetic
nBlks = nRows*nCols;
ch0 = int32(L.EstChannelFirst(iEst));
nCh = int32(L.EstNCh(iEst));
nF = int32(L.EstNFeat(iEst));
nH = int32(L.EstNHidden(iEst));
nA = int32(L.EstNAngles(iEst));
nZ = int32(L.EstNZeroPad(iEst));
offTheta = int32(L.ThetaOffset(iEst));
nv = L.Neighbor(1);
nh = L.Neighbor(2);

% Standardization statistics per channel (equal to those per feature)
mu = zeros(L.NDec,1,'single');
sigma = zeros(L.NDec,1,'single');
for j = 1:nCh
    acc = single(0);
    for c = 1:nCols
        for r = 1:nRows
            acc = acc + Y(ch0+j-1,r,c);
        end
    end
    m = acc/single(nBlks);
    acc = single(0);
    for c = 1:nCols
        for r = 1:nRows
            d = Y(ch0+j-1,r,c) - m;
            acc = acc + d*d;
        end
    end
    mu(j) = m;
    sigma(j) = sqrt(acc/single(nBlks-1)) + single(L.Epsilon);
end

% All intermediate arrays have MaxNHidden rows, so that one version of
% each layer function serves every call, and are partitioned over the
% blocks of the column (dimension 1 in the generated C++)
F = zeros(L.MaxNHidden,nRows,'single');
LN = zeros(L.MaxNHidden,nRows,'single');
Z1 = zeros(L.MaxNHidden,nRows,'single');
A = zeros(L.MaxNHidden,nRows,'single');
Z2 = zeros(L.MaxNHidden,nRows,'single');
Th = zeros(L.MaxNHidden,nRows,'single');
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=F type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=LN type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=Z1 type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=A type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=Z2 type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=Th type=complete dim=1")
for c = 1:nCols
    % Local state of the blocks in column c
    for r = 1:nRows
        iF = int32(0);
        for vshift = fix(nv/2):-1:-fix(nv/2)
            for hshift = fix(nh/2):-1:-fix(nh/2)
                rs = wrap(r-vshift,nRows);
                cs = wrap(c-hshift,nCols);
                for j = 1:nCh
                    F(iF+j,r) = (Y(ch0+j-1,rs,cs) - mu(j))/sigma(j);
                end
                iF = iF + nCh;
            end
        end
    end

    % Residual blocks: LayerNorm, FC, GELU (tanh), FC, skip
    for iRes = 1:L.NResBlocks
        LN = layerNorm(F,w,int32(L.Gamma(iEst,iRes)),int32(L.Beta(iEst,iRes)),nF,L.LnEpsilon);
        Z1 = fullyConnected(LN,w,int32(L.W1(iEst,iRes)),int32(L.B1(iEst,iRes)),nH,nF);
        A = gelu(Z1,nH);
        Z2 = fullyConnected(A,w,int32(L.W2(iEst,iRes)),int32(L.B2(iEst,iRes)),nF,nH);
        for r = 1:nRows
            for i = 1:nF
                F(i,r) = F(i,r) + Z2(i,r);
            end
        end
    end

    % Output angles (leading zeros for no DC leakage)
    Th = fullyConnected(F,w,int32(L.Wo(iEst)),int32(L.Bo(iEst)),nA,nF);
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

function Z = fullyConnected(X,w,offW,offB,nOut,nIn)
% Z(o,b) = B(o) + sum_i W(o,i)*X(i,b) for all blocks b of a column.
% One weight W(o,i) per cycle (consecutive o), applied to every b in
% parallel. Z(o,b) is updated again only nOut cycles later, so the adder
% latency does not limit the pipeline.
coder.inline('never')
% Only rows 1..nOut are written and read, so Z is not zero-filled
% (a zero fill would cost as many cycles as the layer itself)
Z = coder.nullcopy(zeros(size(X),'single'));
nB = size(X,2);
for o = 1:nOut
    for b = 1:nB
        Z(o,b) = w(offB+o);
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

function Z = layerNorm(X,w,offGamma,offBeta,nF,lnEpsilon)
% LayerNorm over the features of each block. The sums run over i in the
% outer loop for all blocks b in parallel, keeping the order of summation
% (so the result is bit-identical to salsun2d_hls). Each sum is updated
% once per iteration of i, so the pipeline waits for the adder; with the
% blocks in parallel this costs about nF times the adder latency in total.
% The normalization uses one divider, pipelined over the blocks.
coder.inline('never')
nB = size(X,2);
Z = coder.nullcopy(zeros(size(X),'single'));   % rows 1..nF are all written
m = zeros(1,nB,'single');
s = zeros(1,nB,'single');
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=m type=complete dim=0")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=s type=complete dim=0")
for i = 1:nF
    coder.hdl.literaltext("#pragma HLS PIPELINE")
    for b = 1:nB
        m(b) = m(b) + X(i,b);
    end
end
for b = 1:nB
    m(b) = m(b)/single(nF);
end
for i = 1:nF
    coder.hdl.literaltext("#pragma HLS PIPELINE")
    for b = 1:nB
        d = X(i,b) - m(b);
        s(b) = s(b) + d*d;
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
