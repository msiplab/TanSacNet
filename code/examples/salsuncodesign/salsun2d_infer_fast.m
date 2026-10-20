function [y,sum1,sum2,muUsed,sigmaUsed] = salsun2d_infer_fast(x,params,mu,sigma) %#codegen
%SALSUN2D_INFER_FAST SA-LSUN 2-D inference written for CPUs and GPUs
%
%   [y,sum1,sum2] = salsun2d_infer_fast(x,params,mu,sigma) computes what
%   the FPGA band kernel computes for one frame x, with the standardization
%   statistics given per channel and estimator (mu, sigma: NDec x NEst,
%   e.g. from the previous frames as in salsun2d_band_sequence), and
%   returns the reconstruction y and the channel sums and sums of squares
%   of the estimator inputs (NDec x NEst) for the statistics of the next
%   frame. It is the fair counterpart of the FPGA kernel for a CPU or a
%   GPU: the whole frame at once (no band halo), every step an operation
%   on all blocks (matrix products for the fully connected layers, the
%   Givens rotations applied to the coefficient rows of all blocks at
%   once), single precision, and the same exact shortcut of the last stage
%   as the kernel (its first Pa-1 angles only; see salsun2d_hls_layout).
%   It runs in MATLAB (multithreaded), on gpuArray inputs, and through
%   MATLAB Coder (MEX, matrix products by BLAS; give the stage targets as
%   numbers, 1 for 'Difference' and 0 for 'Sum', so that the stages form
%   a homogeneous structure array).
%
%   With empty mu and sigma, each estimator standardizes with the
%   statistics of its inputs over the frame itself (as salsun2d_infer
%   without given statistics), e.g. to initialize a stream; [y,sum1,sum2,
%   mu,sigma] = salsun2d_infer_fast(x,params,[],[]) also returns those
%   statistics.
%
%   params as from salsun2d_extract_params (any estimator size), single.
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
dec = params.Stride;
nDec = size(params.Cvh,1);          % a constant size for code generation
ps = nDec/2;
[szy,szx] = size(x);
nRows = szy/dec(1);
nCols = szx/dec(2);
nB = nRows*nCols;
nEst = 1 + numel(params.Stages);
sum1 = zeros(nDec,nEst,'like',x);
sum2 = zeros(nDec,nEst,'like',x);
measure = isempty(mu);
muUsed = zeros(nDec,nEst,'like',x);
sigmaUsed = ones(nDec,nEst,'like',x);
if ~measure
    muUsed(:) = mu;
    sigmaUsed(:) = sigma;
end

%% Analysis
blocks = permute(reshape(x,dec(1),nRows,dec(2),nCols),[1 3 2 4]);
Y = params.Cvh*reshape(blocks,nDec,nB);                     % nDec x nB
[theta,s1,s2,ch,m,sg] = estimateAngles(Y,params.V0.Estimator,muUsed(:,1),sigmaUsed(:,1),measure,nRows,nCols,0);
sum1(ch,1) = s1; sum2(ch,1) = s2; muUsed(ch,1) = m; sigmaUsed(ch,1) = sg;
nHalf = size(theta,1)/2;
Y(1:ps,:) = givensRows(Y(1:ps,:),theta(1:nHalf,:),ps,nHalf,false);
Y(ps+1:end,:) = givensRows(Y(ps+1:end,:),theta(nHalf+1:end,:),ps,nHalf,false);
Y(1:ps,:) = params.V0.MusW(:,1).*Y(1:ps,:);
Y(ps+1:end,:) = params.V0.MusU(:,1).*Y(ps+1:end,:);
nStages = numel(params.Stages);
nAng = ps*(ps-1)/2;
thetas = zeros(nAng,nB,nStages,'like',x);
for iStage = 1:nStages
    st = params.Stages(iStage);
    Y = atomExtension(Y,st.Shift,st.Target,nRows,nCols);
    nUsed = nAng;
    if iStage == nStages
        nUsed = ps - 1;             % last stage: only the rotations of channel ps+1 matter
    end
    [th,s1,s2,ch,m,sg] = estimateAngles(Y,st.Estimator,muUsed(:,iStage+1),sigmaUsed(:,iStage+1),measure,nRows,nCols,nUsed);
    sum1(ch,iStage+1) = s1; sum2(ch,iStage+1) = s2; muUsed(ch,iStage+1) = m; sigmaUsed(ch,iStage+1) = sg;
    thetas(1:nUsed,:,iStage) = th(1:nUsed,:);
    Ya = givensRows(Y(ps+1:end,:),thetas(:,:,iStage),ps,nUsed,false);
    Y(ps+1:end,:) = st.Mus(:,1).*Ya;
end

%% Coefficient mask (the last stage keeps channel ps+1 of the antisymmetric half)
Y = params.Mask(:).*Y;

%% Synthesis
for iStage = nStages:-1:1
    st = params.Stages(iStage);
    nUsed = nAng;
    if iStage == nStages
        nUsed = ps - 1;
    end
    Ya = st.SynMus(:,1).*Y(ps+1:end,:);
    Y(ps+1:end,:) = givensRows(Ya,thetas(:,:,iStage),ps,nUsed,true);
    Y = atomExtension(Y,st.SynShift,st.SynTarget,nRows,nCols);
end
Y(1:ps,:) = givensRows(params.V0t.MusW(:,1).*Y(1:ps,:),theta(1:nHalf,:),ps,nHalf,true);
Y(ps+1:end,:) = givensRows(params.V0t.MusU(:,1).*Y(ps+1:end,:),theta(nHalf+1:end,:),ps,nHalf,true);
blocks = reshape(params.Cvh.'*Y,dec(1),dec(2),nRows,nCols);
y = reshape(ipermute(blocks,[1 3 2 4]),szy,szx);
end

function V = givensRows(V,theta,n,nRot,transposed)
% Apply the Givens rotations G_1..G_nRot (pairs (1,2),(1,3),...,(n-1,n))
% to the rows of V (n x nB, one column per block), with the angles of
% each block in theta(k,:): V = G_nRot*...*G_1*V, or the transposes in
% reverse order
tops = zeros(1,n*(n-1)/2);
btms = zeros(1,n*(n-1)/2);
k = 0;
for i = 1:n-1
    for j = i+1:n
        k = k + 1;
        tops(k) = i;
        btms(k) = j;
    end
end
for kk = 1:nRot
    if transposed
        k = nRot + 1 - kk;           % transposes in reverse order
    else
        k = kk;
    end
    c = cos(theta(k,:));
    s = sin(theta(k,:));
    vt = V(tops(k),:);
    vb = V(btms(k),:);
    if transposed
        V(tops(k),:) = c.*vt + s.*vb;
        V(btms(k),:) = c.*vb - s.*vt;
    else
        V(tops(k),:) = c.*vt - s.*vb;
        V(btms(k),:) = s.*vt + c.*vb;
    end
end
end

function Y = atomExtension(Y,shift,target,nRows,nCols)
nDec = size(Y,1);
ps = nDec/2;
Ys = Y(1:ps,:) + Y(ps+1:end,:);
Ya = Y(1:ps,:) - Y(ps+1:end,:);
% target: 'Difference' / 'Sum', or 1 / 0 (numeric, for code generation)
if ischar(target) || isstring(target)
    isDifference = strcmp(target,'Difference');
else
    isDifference = target ~= 0;
end
if isDifference
    Ya = reshape(circshift(reshape(Ya,ps,nRows,nCols),[0 shift]),ps,nRows*nCols);
else
    Ys = reshape(circshift(reshape(Ys,ps,nRows,nCols),[0 shift]),ps,nRows*nCols);
end
Z = zeros(size(Y),'like',Y);       % same size as Y (fixed rows for code generation)
Z(1:ps,:) = 0.5*(Ys+Ya);
Z(ps+1:end,:) = 0.5*(Ys-Ya);
Y = Z;
end

function [theta,s1,s2,ch,muCh,sgCh] = estimateAngles(Y,est,mu,sigma,measure,nRows,nCols,nUsed)
% Angles of all blocks; nUsed > 0 computes only the first nUsed predicted
% angles (after the zero-padded ones). With measure, the channels are
% standardized with their own mean and standard deviation over the frame
% (those of the features, which are circular shifts of the channels).
ch = est.Channels;
nCh = numel(ch);
nv = est.Neighbor(1);
nh = est.Neighbor(2);
Xc = Y(ch,:);
s1 = sum(Xc,2);
s2 = sum(Xc.^2,2);
if measure
    nBlks = size(Xc,2);
    muCh = s1/nBlks;
    sgCh = sqrt(sum((Xc - muCh).^2,2)/(nBlks-1)) + cast(est.Epsilon,'like',Xc);
else
    muCh = mu(ch);
    sgCh = sigma(ch);
end
Xs = (Xc - muCh)./sgCh;                         % standardized channels
X3 = reshape(Xs,nCh,nRows,nCols);
F = zeros(nCh*nv*nh,nRows*nCols,'like',Y);
iF = 0;
for vshift = fix(nv/2):-1:-fix(nv/2)
    for hshift = fix(nh/2):-1:-fix(nh/2)
        F(iF+(1:nCh),:) = reshape(circshift(X3,[0 vshift hshift]),nCh,nRows*nCols);
        iF = iF + nCh;
    end
end
if isfield(est,'Wp') && ~isempty(est.Wp)
    F = est.Wp*F + est.Bp;
end
for iRes = 1:numel(est.ResBlocks)
    r = est.ResBlocks(iRes);
    m = mean(F,1);
    v = mean((F-m).^2,1);
    ln = r.Gamma.*((F-m)./sqrt(v+1e-5)) + r.Beta;
    z1 = r.W1*ln + r.B1;
    a = 0.5*z1.*(1+tanh(sqrt(2/pi)*(z1+0.044715*z1.^3)));
    F = F + r.W2*a + r.B2;
end
nZ = est.NumberOfZeroPadAngles;
if nUsed > 0
    th = est.Wo(1:nUsed,:)*F + est.Bo(1:nUsed);
else
    th = est.Wo*F + est.Bo;
end
theta = [zeros(nZ,size(F,2),'like',F); th];
end
