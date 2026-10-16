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

coder.extrinsic('salsun2d_hls_layout','salsun2d_band_config');
% Estimator size of this design (salsun2d_band_config, read at code
% generation time)
C = coder.const(salsun2d_band_config());
L = coder.const(salsun2d_hls_layout('NumResidualBlocks',C(1),'Width',C(2),'AlignedWeights',true));
[szy,szx] = size(x);
nRows = szy/L.Stride(1);          % block rows of the band with halo
nCols = szx/L.Stride(2);
H = int32(L.Halo);
rowFirst = H + 1;                  % valid rows of the band
rowLast = int32(nRows) - H;
sumFirst = rowFirst + int32(skipRows);   % first row counted in the sums

% Angles of the band with halo, on chip: the used angles only
% (L.ThetaStoreOffset), as 18-bit fractions of pi (L.FixAngle). URAM
% words are 72 bits wide, so four values are packed per word
% (ARRAY_RESHAPE on the innermost C++ dimension). The store was the
% largest memory of the kernel (46 URAM with single-precision angles and
% the zero rows) and its multiplexers congested the routing.
T = fixedTypes(L);
% (every stored angle is written by its estimator before it is read)
theta = coder.nullcopy(zeros(L.NThetaStoreRows,nRows,nCols,'like',T.Ang));
coder.hdl.literaltext("#pragma HLS ARRAY_RESHAPE variable=theta type=cyclic factor=4 dim=3")
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
% Two channels (one word of Yb) per cycle: a store of single channels
% would read, modify and write the word of the previous one (II = 11)
mask = zeros(L.NDec,1,'single');
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=mask type=complete dim=1")
for k = 1:L.NDec
    mask(k) = w(L.Mask+k);
end
for c = 1:nCols
    for r = 1:nRows
        for k2 = 1:L.NDec/2
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            for q = 1:2
                k = 2*(k2-1) + q;
                Yb(k,r,c) = Yb(k,r,c)*mask(k);
            end
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
% n = iv + (ih-1)*My indexes the pixels of a block in column-major order.
% The DCT matrix is held in registers; for each column of blocks the
% pixels are read first (one per cycle), then one coefficient of a block
% is computed per cycle (16 products). Reading the 16 matrix entries of a
% coefficient from w limited the loop to II = 8.
coder.inline('never')
My = L.Stride(1);
Mx = L.Stride(2);
nPix = My*Mx;
cvh = zeros(L.NDec,nPix,'single');
px = zeros(nPix,nRows,'single');               % one bank per pixel
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=cvh type=complete dim=0")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=px type=complete dim=2")
for n = 1:nPix
    for k = 1:L.NDec
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        cvh(k,n) = w(L.Cvh + k + (n-1)*L.NDec);
    end
end
Y = coder.nullcopy(zeros(L.NDec,nRows,nCols,'single'));   % all elements are written
for c = 1:nCols
    for r = 1:nRows
        for ih = 1:Mx
            for iv = 1:My
                coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
                px(iv+(ih-1)*My,r) = x((r-1)*My+iv,(c-1)*Mx+ih);
            end
        end
    end
    for r = 1:nRows
        for k = 1:L.NDec
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            acc = single(0);
            for n = 1:nPix
                acc = acc + cvh(k,n)*px(n,r);
            end
            Y(k,r,c) = acc;
        end
    end
end
end

function x = blockIdct(Y,w,L,nCols,rowFirst,rowLast)
% Only the block rows rowFirst..rowLast (the band) are reconstructed. As
% blockDct: the matrix in registers, the coefficients of a column of
% blocks read first (two per cycle, one word of Y), then one pixel per
% cycle (16 products)
coder.inline('never')
My = L.Stride(1);
Mx = L.Stride(2);
nPix = My*Mx;
nOut = rowLast - rowFirst + 1;
cvh = zeros(L.NDec,nPix,'single');
yc = zeros(L.NDec,size(Y,2),'single');      % one bank per channel
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=cvh type=complete dim=0")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=yc type=complete dim=2")
for n = 1:nPix
    for k = 1:L.NDec
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        cvh(k,n) = w(L.Cvh + k + (n-1)*L.NDec);
    end
end
x = zeros(nOut*My,nCols*Mx,'single');
for c = 1:nCols
    for r = rowFirst:rowLast
        for k2 = 1:L.NDec/2
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            for q = 1:2
                k = 2*(k2-1) + q;
                yc(k,r) = Y(k,r,c);
            end
        end
    end
    for r = rowFirst:rowLast
        ro = r - rowFirst;
        for ih = 1:Mx
            for iv = 1:My
                coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
                n = iv + (ih-1)*My;
                acc = single(0);
                for k = 1:L.NDec
                    acc = acc + cvh(k,n)*yc(k,r);
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
% D*(G_K*(...*(G_1*y))) and the synthesis U.'*y = G_1.'*(...*(G_K.'*(D*y))).
% G_k rotates the pair (i,j) of L.GivensTop(k), L.GivensBottom(k) as
% [c -s; s c] (as fcn_orthmtxgen).
% For each column of blocks, the cosines and sines of all blocks are
% computed first (one angle per cycle), then the rotations are applied
% rotation by rotation, each to the nRows blocks of the column in turn
% (outer loop over the rotations, inner pipelined loop over the blocks):
% one rotation unit at one rotation per cycle, and a block is rotated
% again only nRows cycles later, so the chain of dependent rotations does
% not stall the pipeline. (An earlier form unrolled the 28 rotations of a
% block in a pipeline of II = 28; its wiring was the main routing
% congestion of the kernel.)

function Z = rotateInitialOrFinal(Y,theta,w,L,isFinal,nRows,nCols)
% Initial rotation (W on the symmetric, U on the antisymmetric channels)
% or its transpose (isFinal). W and U are applied in the same loop (two
% rotation units). The callers pass isFinal with coder.ignoreConst, so
% that the analysis and the synthesis share one instance of the function.
coder.inline('never')
nHalf = coder.const(L.EstNAnglesTotal(1)/2);   % angles of W and of U (28)
nZ = coder.const(L.EstNZeroPad(1));            % leading zero angles of W (not stored)
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
vs = zeros(L.Ps,nRows,'single');
va = zeros(L.Pa,nRows,'single');
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=musW type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=musU type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=vs type=complete dim=2")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=va type=complete dim=2")
for c = 1:nCols
    for r = 1:nRows
        for a = 1:nHalf
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            if a > nZ
                tw = angleOf(theta(a-nZ,r,c));
            else
                tw = single(0);
            end
            tu = angleOf(theta(nHalf-nZ+a,r,c));
            csW(a,r) = cos(tw);
            snW(a,r) = sin(tw);
            csU(a,r) = cos(tu);
            snU(a,r) = sin(tu);
        end
    end
    for r = 1:nRows
        for j = 1:L.Ps
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            if isFinal
                vs(j,r) = musW(j)*Y(j,r,c);
                va(j,r) = musU(j)*Y(L.Ps+j,r,c);
            else
                vs(j,r) = Y(j,r,c);
                va(j,r) = Y(L.Ps+j,r,c);
            end
        end
    end
    for k = 1:nHalf
        for r = 1:nRows
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            coder.hdl.literaltext("#pragma HLS DEPENDENCE variable=vs type=inter false")
            coder.hdl.literaltext("#pragma HLS DEPENDENCE variable=va type=inter false")
            if isFinal
                a = nHalf + 1 - k;     % transposes in reverse order
            else
                a = k;
            end
            t = tops(a);
            b = btms(a);
            if isFinal
                [vs(t,r),vs(b,r)] = givensT(vs(t,r),vs(b,r),csW(a,r),snW(a,r));
                [va(t,r),va(b,r)] = givensT(va(t,r),va(b,r),csU(a,r),snU(a,r));
            else
                [vs(t,r),vs(b,r)] = givens(vs(t,r),vs(b,r),csW(a,r),snW(a,r));
                [va(t,r),va(b,r)] = givens(va(t,r),va(b,r),csU(a,r),snU(a,r));
            end
        end
    end
    for r = 1:nRows
        for j = 1:L.Ps
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            if isFinal
                Z(j,r,c) = vs(j,r);
                Z(L.Ps+j,r,c) = va(j,r);
            else
                Z(j,r,c) = musW(j)*vs(j,r);
                Z(L.Ps+j,r,c) = musU(j)*va(j,r);
            end
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
offTheta = int32(L.ThetaStoreOffset(iEst));
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
va = zeros(L.Pa,nRows,'single');
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=mus type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=va type=complete dim=2")
for c = 1:nCols
    for r = 1:nRows
        for a = int32(1):nRot          % the stored angles (Pa-1 in the last stage)
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            th = angleOf(theta(offTheta+a,r,c));
            cs(a,r) = cos(th);
            sn(a,r) = sin(th);
        end
    end
    for r = 1:nRows
        for j = 1:L.Pa
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            if isSynthesis
                va(j,r) = mus(j)*Y(L.Ps+j,r,c);
            else
                va(j,r) = Y(L.Ps+j,r,c);
            end
        end
    end
    for k = int32(1):nRot
        for r = 1:nRows
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            coder.hdl.literaltext("#pragma HLS DEPENDENCE variable=va type=inter false")
            if isSynthesis
                a = nRot + 1 - k;      % transposes in reverse order
            else
                a = k;
            end
            t = tops(a);
            b = btms(a);
            if isSynthesis
                [va(t,r),va(b,r)] = givensT(va(t,r),va(b,r),cs(a,r),sn(a,r));
            else
                [va(t,r),va(b,r)] = givens(va(t,r),va(b,r),cs(a,r),sn(a,r));
            end
        end
    end
    for r = 1:nRows
        for j = 1:L.Pa
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            if isSynthesis
                Z(L.Ps+j,r,c) = va(j,r);
            elseif isLast && j > 1
                Z(L.Ps+j,r,c) = single(0);
            else
                Z(L.Ps+j,r,c) = mus(j)*va(j,r);
            end
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

%% Angle estimation (fixed point)
% The estimators compute in fixed point: the inputs of the fully connected
% layers (standardized features and residual sums F, LayerNorm outputs LN,
% GELU outputs A) are L.FixSignal = [word fraction] lengths, the weights
% L.FixWeight, the accumulators L.FixAcc (exact sums of the products), as
% chosen with evaluate_salsun2d_fixed_estimator (Common=true). The
% LayerNorm scaling, the GELU and the standardization are computed in
% single precision from the exact sums and rounded to the signal format;
% the angles are converted to single precision. In MATLAB the fixed-point
% values are emulated in double precision with the same rounding
% (salsun2d_round_fixed), so that the MATLAB execution of this design is
% fast and matches the generated code; in code generation they are fi
% objects (ap_fixed in the generated C++).

function [theta,sum1,sum2] = estimateAngles(theta,sum1,sum2,Y,w,L,iEst,mu,sigma,nRows,nCols,sumFirst,sumLast)
% A group of nCG = gcd(nCols,L.NColGroup) columns of blocks is processed
% at a time: nB = nCG*nRows blocks, block (r,c0+cc-1) at index
% b = (cc-1)*nRows + r. The fully connected layers have one lane per
% block, each with L.FcInputs multipliers; the other layers use fewer
% lanes (L.NLnLanes, L.NGeluUnits), as they are a small part of the work.
% The standardization uses the given statistics mu(:,iEst), sigma(:,iEst)
% per channel; the channel sums over the rows sumFirst..sumLast (the
% band without rows already counted) are accumulated for the statistics
% of the next frame.
coder.inline('never')
T = fixedTypes(L);
ch0 = int32(L.EstChannelFirst(iEst));
nCh = int32(L.EstNCh(iEst));
nF = int32(L.EstNFeat(iEst));
nH = int32(L.EstNHidden(iEst));
nA = int32(L.EstNAngles(iEst));          % predicted angles
nAU = int32(L.EstNAnglesUsed(iEst));     % angles computed (7 in the last stage)
offWord = int32(L.ThetaStoreOffset(iEst)/4);   % first word of the angle store
nWords = idivide(nAU + int32(3),int32(4));
ang = zeros(4,1,'like',T.Ang);
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=ang type=complete dim=0")
nv = L.Neighbor(1);
nh = L.Neighbor(2);
nCG = coder.const(gcd(nCols,L.NColGroup));
nB = coder.const(nCG*nRows);

% Channel sums over the valid blocks of the band (statistics of the next
% frame), one value per cycle with the channels innermost: a sum is
% updated again nCh >= 8 cycles later, so the additions do not stall the
% pipeline (with one channel at a time, II = 8). The order of the terms
% of each sum is unchanged.
acc1 = zeros(L.NDec,1,'single');
acc2 = zeros(L.NDec,1,'single');
for c = 1:nCols
    for r = sumFirst:sumLast
        for j = 1:nCh
            coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
            coder.hdl.literaltext("#pragma HLS DEPENDENCE variable=acc1 type=inter direction=RAW dependent=true distance=8")
            coder.hdl.literaltext("#pragma HLS DEPENDENCE variable=acc2 type=inter direction=RAW dependent=true distance=8")
            v = Y(ch0+j-1,r,c);
            acc1(j) = acc1(j) + v;
            acc2(j) = acc2(j) + v*v;
        end
    end
end
for j = 1:nCh
    sum1(ch0+j-1,iEst) = sum1(ch0+j-1,iEst) + acc1(j);
    sum2(ch0+j-1,iEst) = sum2(ch0+j-1,iEst) + acc2(j);
end

% Reciprocals of the scales: the standardization is then a multiplication
invSigma = zeros(L.NDec,1,'single');
for j = 1:nCh
    coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
    invSigma(j) = single(1)/sigma(ch0+j-1,iEst);
end

% All intermediate arrays have L.MaxNRows rows (features or hidden units),
% so that one version of
% each layer function serves every call. They are split completely over
% the blocks (dimension 1 in the generated C++): one bank per block.
F = zeros(L.MaxNRows,nB,'like',T.X);
LN = zeros(L.MaxNRows,nB,'like',T.X);
Z1 = zeros(L.MaxNRows,nB,'like',T.Acc);
A = zeros(L.MaxNRows,nB,'like',T.X);
Z2 = zeros(L.MaxNRows,nB,'like',T.Acc);
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
                        F(iF,b) = toX((Y(ch0+j-1,rs,cs) - mu(ch0+j-1,iEst))*invSigma(j),T,L);
                    end
                end
            end
        end
    end

    % Residual blocks: LayerNorm, FC, GELU (tanh), FC, skip
    for iRes = 1:L.NResBlocks
        LN = layerNorm(F,w,int32(L.Gamma(iEst,iRes)),int32(L.Beta(iEst,iRes)),nF,L.LnEpsilon,L);
        Z1 = fullyConnected(LN,w,int32(L.W1(iEst,iRes)),int32(L.B1(iEst,iRes)),nH,nF,L);
        A = gelu(Z1,nH,L);
        Z2 = fullyConnected(A,w,int32(L.W2(iEst,iRes)),int32(L.B2(iEst,iRes)),nF,nH,L);
        F = residualAdd(F,Z2,nF,L);
    end

    % Output angles (leading zeros for no DC leakage). In the last stage
    % only the first nAU angles are used; all nA rows are still computed,
    % because the pipeline of fullyConnected needs nOut >= 28 (and the
    % output layer is 3% of the multiply-adds)
    Z1 = fullyConnected(F,w,int32(L.Wo(iEst)),int32(L.Bo(iEst)),nA,nF,L);
    for cc = 1:nCG
        c = c0 + cc - 1;
        for r = 1:nRows
            b = (cc-1)*nRows + r;
            % Four angles per word: collected in registers and written
            % as a whole word (a store of single angles would read,
            % modify and write the word in one cycle)
            for g = 1:nWords
                for p = 1:4
                    coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
                    a = 4*(g-1) + p;
                    if a <= nAU
                        ang(p) = toAngle(Z1(a,b),T,L);
                    else
                        ang(p) = cast(0,'like',T.Ang);
                    end
                    if p == 4
                        for q = 1:4
                            theta(4*(offWord+g-1)+q,r,c) = ang(q);
                        end
                    end
                end
            end
        end
    end
end
end

function Z = fullyConnected(X,w,offW,offB,nOut,nIn,L)
% Z(o,b) = B(o) + sum_i W(o,i)*X(i,b) for all blocks b of the group, in
% fixed point with exact accumulation. The weights are stored row by row
% (salsun2d_pack_params(...,RowMajorWeights=true)), W(o,i) at
% offW + (o-1)*nInPad + i, the rows padded with zeros to nInPad, a
% multiple of P = L.FcInputs, and offW a multiple of P
% (salsun2d_hls_layout(...,AlignedWeights=true)): the P weights
% W(o,i0+1..i0+P) of one cycle are one word of w (the reshaping of w in
% the kernel). For each group of P inputs, the inputs of every block are
% held in registers, and each cycle adds P products to Z(o,b) for all
% blocks at once (one lane per block). Z(o,b) is updated again only nOut
% cycles later, more than the adder latency (nOut >= 28), so the
% dependence through Z is declared false. Inputs beyond nIn are zero.
coder.inline('never')
T = fixedTypes(L);
P = coder.const(L.FcInputs);
nB = size(X,2);
% Only rows 1..nOut are written and read, so Z is not zero-filled
Z = coder.nullcopy(zeros(size(X),'like',T.Acc));
xr = zeros(P,nB,'like',T.X);
wv = zeros(P,1,'like',T.W);
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=xr type=complete dim=0")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=wv type=complete dim=0")
for o = 1:nOut
    coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
    bv = toAcc(w(offB+o),T,L);
    for b = 1:nB
        Z(o,b) = bv;
    end
end
nGi = idivide(int32(nIn) + int32(P) - 1,int32(P));
nWordsRow = nGi;                    % words per row of the weights
for gi = 1:nGi
    i0 = (gi-1)*int32(P);
    for p = 1:P
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        for b = 1:nB
            if i0 + p <= nIn
                xr(p,b) = X(i0+p,b);
            else
                xr(p,b) = toX(0,T,L);
            end
        end
    end
    for o = 1:nOut
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        coder.hdl.literaltext("#pragma HLS DEPENDENCE variable=Z type=inter false")
        % word m of w (P values): the index P*m + p tells HLS the lane
        m = idivide(offW,int32(P)) + (int32(o)-1)*nWordsRow + int32(gi) - 1;
        for p = 1:P
            wv(p) = toW(w(int32(P)*m + int32(p)),T,L);
        end
        for b = 1:nB
            acc = Z(o,b);
            for p = 1:P
                acc = toAccWrap(acc + wv(p)*xr(p,b),T);
            end
            Z(o,b) = acc;
        end
    end
end
end

function F = residualAdd(F,Z2,nF,L)
% F(i,b) = F(i,b) + Z2(i,b) rounded to the signal format, L.NLnLanes
% blocks per cycle
coder.inline('never')
T = fixedTypes(L);
nB = size(F,2);
nL = coder.const(gcd(nB,L.NLnLanes));
nG = nB/nL;
for i = 1:nF
    for g = 1:nG
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        for lane = 1:nL
            b = (g-1)*nL + lane;
            F(i,b) = toX(F(i,b) + Z2(i,b),T,L);
        end
    end
end
end

function Z = layerNorm(X,w,offGamma,offBeta,nF,lnEpsilon,L)
% LayerNorm over the features of each block, nL = L.NLnLanes blocks at a
% time. The sums of the values and of their squares are exact fixed-point
% sums (one feature per cycle); the mean and the reciprocal standard
% deviation are computed from them in single precision, and the
% normalized values are rounded to the signal format.
coder.inline('never')
T = fixedTypes(L);
nB = size(X,2);
nL = coder.const(gcd(nB,L.NLnLanes));
nG = nB/nL;
Z = coder.nullcopy(zeros(size(X),'like',T.X));   % rows 1..nF are all written
s1 = zeros(1,nB,'like',T.Acc);
s2 = zeros(1,nB,'like',T.Sq);
m = zeros(1,nB,'single');
rs = zeros(1,nB,'single');
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=s1 type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=s2 type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=m type=complete dim=1")
coder.hdl.literaltext("#pragma HLS ARRAY_PARTITION variable=rs type=complete dim=1")
for g = 1:nG
    for i = 1:nF
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        for lane = 1:nL
            b = (g-1)*nL + lane;
            v = X(i,b);
            s1(b) = toAccWrap(s1(b) + v,T);
            s2(b) = toSqWrap(s2(b) + v*v,T);
        end
    end
end
for b = 1:nB
    coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
    mb = single(s1(b))/single(nF);
    vb = single(s2(b))/single(nF) - mb*mb;
    vb = max(vb,single(0));
    m(b) = mb;
    rs(b) = single(1)/sqrt(vb + single(lnEpsilon));
end
for i = 1:nF
    for g = 1:nG
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        gam = w(offGamma+i);
        bet = w(offBeta+i);
        for lane = 1:nL
            b = (g-1)*nL + lane;
            Z(i,b) = toX(gam*((single(X(i,b)) - m(b))*rs(b)) + bet,T,L);
        end
    end
end
end

function A = gelu(Z,nH,L)
% GELU (tanh form) in single precision, rounded to the signal format,
% nU = L.NGeluUnits blocks per cycle
coder.inline('never')
T = fixedTypes(L);
nB = size(Z,2);
nU = coder.const(gcd(nB,L.NGeluUnits));
nG = nB/nU;
A = coder.nullcopy(zeros(size(Z),'like',T.X));   % rows 1..nH are all written
for h = 1:nH
    for g = 1:nG
        coder.hdl.literaltext("#pragma HLS PIPELINE II=1")
        for u = 1:nU
            b = (g-1)*nU + u;
            v = single(Z(h,b));
            A(h,b) = toX(single(0.5)*v*(single(1) + ...
                tanh(single(sqrt(2/pi))*(v + single(0.044715)*v*v*v))),T,L);
        end
    end
end
end

%% Fixed-point types and casts
function T = fixedTypes(L)
% Prototypes of the fixed-point values: double in MATLAB (emulation, the
% casts below round), fi objects in code generation. All fi values share
% one fimath (full-precision products and sums, truncation and wrapping on
% assignment, which adds no logic); the casts to the signal, weight and
% accumulator formats round to nearest (ties upward) and saturate
% explicitly. The accumulations are exact and never wrap with these
% accumulator lengths.
if coder.target('MATLAB')
    T.X = double([]);
    T.W = double([]);
    T.Acc = double([]);
    T.Sq = double([]);
    T.Ang = double([]);
else
    Fw = fimath('RoundingMethod','Floor','OverflowAction','Wrap', ...
        'ProductMode','FullPrecision','SumMode','FullPrecision');
    T.X = fi([],1,L.FixSignal(1),L.FixSignal(2),Fw);
    T.W = fi([],1,L.FixWeight(1),L.FixWeight(2),Fw);
    T.Acc = fi([],1,L.FixAcc(1),L.FixAcc(2),Fw);
    T.Sq = fi([],1,L.FixSq(1),L.FixSq(2),Fw);
    T.Ang = fi([],1,L.FixAngle(1),L.FixAngle(2),Fw);   % stored angles, theta/pi
    T.AngIn = fi([],1,27,21,Fw);              % angle before the scaling (toAngle)
    T.InvPi = fi(1/pi,0,18,18,Fw);
end
end

function y = toAngle(v,T,L)
% Angle (radians, accumulator format) into the angle store: theta/pi
% rounded to nearest and wrapped into [-1,1). The angle is first rounded
% (and saturated) to 27 bits, [27 21] (+-32 rad), and multiplied by 1/pi
% in [18 18], one DSP multiplier (27 x 18)
if coder.target('MATLAB')
    y = salsun2d_round_angle(double(v),L.FixAngle(1))/pi;
else
    Fa = fimath('RoundingMethod','Nearest','OverflowAction','Wrap', ...
        'ProductMode','FullPrecision','SumMode','FullPrecision');
    a = roundTo(v,T.AngIn);
    y = cast(fi(a*T.InvPi,numerictype(T.Ang),Fa),'like',T.Ang);
end
end

function t = angleOf(v)
% Angle in radians (single) of a stored value theta/pi
t = single(pi)*single(v);
end

function y = roundTo(v,p)
% Round to nearest and saturate into the type of the prototype p (fi)
Fr = fimath('RoundingMethod','Nearest','OverflowAction','Saturate', ...
    'ProductMode','FullPrecision','SumMode','FullPrecision');
y = cast(fi(v,numerictype(p),Fr),'like',p);
end

function y = toX(v,T,L)
if coder.target('MATLAB')
    y = salsun2d_round_fixed(double(v),L.FixSignal(1),L.FixSignal(2));
else
    y = roundTo(v,T.X);
end
end

function y = toW(v,T,L)
if coder.target('MATLAB')
    y = salsun2d_round_fixed(double(v),L.FixWeight(1),L.FixWeight(2));
else
    y = roundTo(v,T.W);
end
end

function y = toAcc(v,T,L)
% Rounding cast into the accumulator format (biases)
if coder.target('MATLAB')
    y = salsun2d_round_fixed(double(v),L.FixAcc(1),L.FixAcc(2));
else
    y = roundTo(v,T.Acc);
end
end

function y = toAccWrap(v,T)
% Exact sums into the accumulator format (no rounding needed)
if coder.target('MATLAB')
    y = v;
else
    y = cast(v,'like',T.Acc);
end
end

function y = toSqWrap(v,T)
if coder.target('MATLAB')
    y = v;
else
    y = cast(v,'like',T.Sq);
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
