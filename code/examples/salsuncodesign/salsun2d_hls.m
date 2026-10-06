function y = salsun2d_hls(x,w) %#codegen
%SALSUN2D_HLS SA-LSUN 2-D analysis and synthesis written for HLS
%
%   y = salsun2d_hls(x,w) computes the same result as
%   salsun2d_infer(x,params) for the configuration in salsun2d_hls_layout,
%   where w = salsun2d_pack_params(params).
%
%   This version is written for HDL Coder (Vitis HLS): fixed
%   configuration, explicit loops, all parameters in the single vector w,
%   and no array functions such as circshift or matrix products. The
%   image size is taken from x and becomes a constant at code generation.
%
%   Two observations keep the control path cheap:
%   - The features of an estimator are circular shifts of the same
%     channels, so their mean and variance over the image equal those of
%     the channels. The standardization statistics are computed once per
%     channel instead of once per feature.
%   - Each estimator reads the neighbors of a block before rotation, so
%     the rotated coefficients go to a second buffer.
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

% Estimated angles of all estimators, reused by the synthesis side
theta = zeros(L.NThetaRows,nRows,nCols,'single');

%% Analysis
Y = blockDct(x,w,L,nRows,nCols);
theta = estimateAngles(theta,Y,w,L,1,nRows,nCols);
Y = rotateInitialOrFinal(Y,theta,w,L,false,nRows,nCols);
for iStage = 1:L.NStages
    Y = atomExtension(Y,L.Shift(iStage,:),L.Target(iStage),nRows,nCols);
    theta = estimateAngles(theta,Y,w,L,iStage+1,nRows,nCols);
    Y = rotateIntermediate(Y,theta,w,L,iStage,false,nRows,nCols);
end

%% Coefficient mask
for c = 1:nCols
    for r = 1:nRows
        for k = 1:L.NDec
            Y(k,r,c) = Y(k,r,c)*w(L.Mask+k);
        end
    end
end

%% Synthesis
for iStage = L.NStages:-1:1
    Y = rotateIntermediate(Y,theta,w,L,iStage,true,nRows,nCols);
    Y = atomExtension(Y,L.SynShift(iStage,:),L.SynTarget(iStage),nRows,nCols);
end
Y = rotateInitialOrFinal(Y,theta,w,L,true,nRows,nCols);
y = blockIdct(Y,w,L,nRows,nCols);
end

%% Block transforms
function Y = blockDct(x,w,L,nRows,nCols)
% Y(:,r,c) = Cvh * vec(block (r,c) of x), column-major within the block
My = L.Stride(1);
Y = zeros(L.NDec,nRows,nCols,'single');
for c = 1:nCols
    for r = 1:nRows
        for k = 1:L.NDec
            acc = single(0);
            for n = 1:L.NDec
                iv = mod(n-1,My) + 1;
                ih = floor((n-1)/My) + 1;
                acc = acc + w(L.Cvh + k + (n-1)*L.NDec)*x((r-1)*My+iv,(c-1)*L.Stride(2)+ih);
            end
            Y(k,r,c) = acc;
        end
    end
end
end

function x = blockIdct(Y,w,L,nRows,nCols)
% vec(block (r,c) of x) = Cvh.' * Y(:,r,c)
My = L.Stride(1);
x = zeros(nRows*My,nCols*L.Stride(2),'single');
for c = 1:nCols
    for r = 1:nRows
        for n = 1:L.NDec
            acc = single(0);
            for k = 1:L.NDec
                acc = acc + w(L.Cvh + k + (n-1)*L.NDec)*Y(k,r,c);
            end
            iv = mod(n-1,My) + 1;
            ih = floor((n-1)/My) + 1;
            x((r-1)*My+iv,(c-1)*L.Stride(2)+ih) = acc;
        end
    end
end
end

%% Atom extension
function Z = atomExtension(Y,shift,target,nRows,nCols)
% Butterfly, shift of the difference (target 1) or sum (target 2) half
% by one block with circular boundary, butterfly and scaling by 1/2
ps = size(Y,1)/2;
Z = zeros(size(Y),'single');
for c = 1:nCols
    for r = 1:nRows
        rs = wrap(r-shift(1),nRows);   % source block of the shifted half
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
% Initial rotation: [W0 0; 0 U0]*y. Final rotation: [W0.' 0; 0 U0.']*y.
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
% Rotate the antisymmetric half by U (analysis) or U.' (synthesis)
iEst = iStage + 1;
nAngles = L.EstNAnglesTotal(iEst);
if isSynthesis
    offMus = L.StageSynMus(iStage);
else
    offMus = L.StageMus(iStage);
end
Z = Y;
angles = zeros(nAngles,1,'single');
for c = 1:nCols
    for r = 1:nRows
        for a = 1:nAngles
            angles(a) = theta(L.ThetaOffset(iEst)+a,r,c);
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
% Product of Givens rotations followed by sign flips (as fcn_orthmtxgen)
M = zeros(n,n,'single');
for i = 1:n
    M(i,i) = single(1);
end
iAng = 1;
for iTop = 1:n-1
    for iBtm = iTop+1:n
        cs = cos(angles(iAng));
        sn = sin(angles(iAng));
        for j = 1:n
            vt = M(iTop,j);
            vb = M(iBtm,j);
            u = sn*(vt + vb);
            M(iTop,j) = (cs + sn)*vt - u;
            M(iBtm,j) = (cs - sn)*vb + u;
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
nBlks = nRows*nCols;
ch0 = L.EstChannelFirst(iEst);
nCh = L.EstNCh(iEst);
nF = L.EstNFeat(iEst);
nH = L.EstNHidden(iEst);
nA = L.EstNAngles(iEst);
nZ = L.EstNZeroPad(iEst);
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

f = zeros(L.MaxNFeat,1,'single');
ln = zeros(L.MaxNFeat,1,'single');
act = zeros(L.MaxNHidden,1,'single');
for c = 1:nCols
    for r = 1:nRows
        % Local state: standardized channels of the neighbor blocks
        iF = 0;
        for vshift = fix(nv/2):-1:-fix(nv/2)
            for hshift = fix(nh/2):-1:-fix(nh/2)
                rs = wrap(r-vshift,nRows);
                cs = wrap(c-hshift,nCols);
                for j = 1:nCh
                    f(iF+j) = (Y(ch0+j-1,rs,cs) - mu(j))/sigma(j);
                end
                iF = iF + nCh;
            end
        end

        % Residual blocks: LayerNorm, FC, GELU (tanh), FC, skip
        for iRes = 1:L.NResBlocks
            acc = single(0);
            for i = 1:nF
                acc = acc + f(i);
            end
            m = acc/single(nF);
            acc = single(0);
            for i = 1:nF
                d = f(i) - m;
                acc = acc + d*d;
            end
            s = sqrt(acc/single(nF) + single(L.LnEpsilon));
            for i = 1:nF
                ln(i) = w(L.Gamma(iEst,iRes)+i)*((f(i) - m)/s) + w(L.Beta(iEst,iRes)+i);
            end
            for h = 1:nH
                acc = w(L.B1(iEst,iRes)+h);
                for i = 1:nF
                    acc = acc + w(L.W1(iEst,iRes) + h + (i-1)*nH)*ln(i);
                end
                act(h) = single(0.5)*acc*(single(1) + ...
                    tanh(single(sqrt(2/pi))*(acc + single(0.044715)*acc*acc*acc)));
            end
            for i = 1:nF
                acc = w(L.B2(iEst,iRes)+i);
                for h = 1:nH
                    acc = acc + w(L.W2(iEst,iRes) + i + (h-1)*nF)*act(h);
                end
                f(i) = f(i) + acc;
            end
        end

        % Output angles (leading zeros for no DC leakage)
        for a = 1:nZ
            theta(L.ThetaOffset(iEst)+a,r,c) = single(0);
        end
        for a = 1:nA
            acc = w(L.Bo(iEst)+a);
            for i = 1:nF
                acc = acc + w(L.Wo(iEst) + a + (i-1)*nA)*f(i);
            end
            theta(L.ThetaOffset(iEst)+nZ+a,r,c) = acc;
        end
    end
end
end

function i = wrap(i,n)
% 1-based circular index
i = mod(i-1,n) + 1;
end
