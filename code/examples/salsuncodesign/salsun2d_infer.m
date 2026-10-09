function [y,coefs,thetas,stageInputs,stats] = salsun2d_infer(x,params,options)
%SALSUN2D_INFER Reference SA-LSUN 2-D analysis and synthesis
%
%   y = salsun2d_infer(x,params) passes the image x (single, szy x szx)
%   through the SA-LSUN analyzer, the coefficient mask and the synthesizer,
%   and returns the reconstructed image y. params comes from
%   salsun2d_extract_params. The result matches predict of the dlnetwork
%   (Mode 'Whole', ThetaMode 'Reuse') that params was extracted from.
%
%   [y,coefs] = salsun2d_infer(...) also returns the masked analysis
%   coefficients (prod(Stride) x szy/My x szx/Mx).
%
%   [y,coefs,thetas] = salsun2d_infer(...) also returns the estimated
%   rotation angles: thetas{1} for the initial rotation and thetas{k+1}
%   for the k-th intermediate stage, each nAngles x (number of blocks).
%
%   [y,coefs,thetas,stageInputs] = salsun2d_infer(...) also returns the
%   inputs of the angle estimators (the coefficients before each
%   rotation), stageInputs{k} for estimator k, each prod(Stride) x nRows
%   x nCols, for diagnostics such as the standardization statistics.
%
%   [y,coefs,thetas,stageInputs,stats] = salsun2d_infer(...) also returns
%   the standardization statistics measured on this image, a struct array
%   with stats(k).Mu and stats(k).Sigma (per feature of estimator k).
%
%   salsun2d_infer(x,params,Statistics=stats) standardizes the estimator
%   inputs with the given statistics instead of those of the image, for
%   example the statistics of the previous frame of a sequence (see
%   salsun2d_infer_sequence). stats must have the form returned above.
%   The measured statistics of the image are still returned.
%
%   salsun2d_infer(x,params,Quantizer=q) applies q(v,tag) to the
%   intermediate values, for word-length studies (see
%   evaluate_salsun2d_wordlength). tag is 'coefs' (block coefficients
%   after the DCT, every rotation and every atom extension), 'rotation'
%   (the rotation matrices), 'features' (standardized estimator inputs
%   and residual sums), 'ln' (LayerNorm outputs), 'z1', 'act' (the
%   hidden layer before and after GELU), 'angles'. The parameters are
%   quantized separately with salsun2d_quantize_params.
%
%   salsun2d_infer(...,EstimatorTags=true) appends the estimator index to
%   the tags of the estimator signals ('features:k', 'ln:k', 'z1:k',
%   'act:k', 'angles:k', k = 1 for the initial rotation and k = s+1 for
%   stage s), for fixed-point formats per estimator (see
%   salsun2d_calibrate_formats and salsun2d_static_quantizer).
%
%   Only plain arithmetic is used (no Deep Learning Toolbox), as a
%   reference for the HLS implementation. The synthesizer reuses the
%   rotation angles estimated by the analyzer. The computation follows the
%   class of x, so casting x and params to double gives a high-precision
%   reference.
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
arguments
    x (:,:) {mustBeFloat}
    params (1,1) struct
    options.Statistics = []
    options.Quantizer = @(v,tag) v
    options.EstimatorTags (1,1) logical = false
end
q = options.Quantizer;
if options.EstimatorTags
    qEst = @(k) @(v,tag) q(v,sprintf('%s:%d',tag,k));
else
    qEst = @(k) q;
end
nEst = 1 + numel(params.Stages);
if ~isempty(options.Statistics) && numel(options.Statistics) ~= nEst
    error('salsun2d_infer:statistics','Statistics must have %d elements.',nEst)
end
stats = repmat(struct('Mu',[],'Sigma',[]),nEst,1);

dec = params.Stride;
nDec = prod(dec);
[szy,szx] = size(x);
nRows = szy/dec(1);
nCols = szx/dec(2);

% Analysis: block DCT and initial rotation
Y = q(blockDct(x,params.Cvh,dec),'coefs');
stageInputs = cell(nEst,1);
stageInputs{1} = Y;
[theta0,stats(1)] = estimateAngles(Y,params.V0.Estimator,givenStats(options.Statistics,1),qEst(1));
Y = q(rotateInitial(Y,theta0,params.V0.MusW,params.V0.MusU,q),'coefs');

% Analysis: intermediate stages
nStages = numel(params.Stages);
thetaStages = cell(nStages,1);
for iStage = 1:nStages
    stage = params.Stages(iStage);
    Y = q(atomExtension(Y,stage.Shift,stage.Target),'coefs');
    stageInputs{iStage+1} = Y;
    [thetaStages{iStage},stats(iStage+1)] = estimateAngles(Y,stage.Estimator, ...
        givenStats(options.Statistics,iStage+1),qEst(iStage+1));
    Y = q(rotateIntermediate(Y,thetaStages{iStage},stage.Mus,false,q),'coefs');
end

% Coefficient mask
Y = Y.*params.Mask(:);
coefs = reshape(Y,nDec,nRows,nCols);

% Synthesis: intermediate stages in reverse order, reusing the angles
for iStage = nStages:-1:1
    stage = params.Stages(iStage);
    Y = q(rotateIntermediate(Y,thetaStages{iStage},stage.SynMus,true,q),'coefs');
    Y = q(atomExtension(Y,stage.SynShift,stage.SynTarget),'coefs');
end

% Synthesis: final rotation and block IDCT
Y = q(rotateFinal(Y,theta0,params.V0t.MusW,params.V0t.MusU,q),'coefs');
y = blockIdct(Y,params.Cvh,dec);
thetas = [{theta0}; thetaStages];
end

%% Block transforms
function Y = blockDct(x,Cvh,dec)
% Y(:,r,c) = Cvh * (block (r,c) of x, column-major)
[szy,szx] = size(x);
nRows = szy/dec(1);
nCols = szx/dec(2);
blocks = permute(reshape(x,dec(1),nRows,dec(2),nCols),[1 3 2 4]);
Y = reshape(Cvh*reshape(blocks,prod(dec),[]),prod(dec),nRows,nCols);
end

function x = blockIdct(Y,Cvh,dec)
[nDec,nRows,nCols] = size(Y);
blocks = reshape(Cvh.'*reshape(Y,nDec,[]),dec(1),dec(2),nRows,nCols);
x = reshape(ipermute(blocks,[1 3 2 4]),dec(1)*nRows,dec(2)*nCols);
end

%% Atom extension
function Y = atomExtension(Y,shift,target)
% Butterfly, circular shift of the sum or difference half, butterfly
nDec = size(Y,1);
ps = nDec/2;
Ys = Y(1:ps,:,:) + Y(ps+1:end,:,:);
Ya = Y(1:ps,:,:) - Y(ps+1:end,:,:);
if strcmp(target,'Difference')
    Ya = circshift(Ya,[0 shift]);
else
    Ys = circshift(Ys,[0 shift]);
end
Y = 0.5*cat(1,Ys+Ya,Ys-Ya);
end

%% Rotations
function Y = rotateInitial(Y,theta,musW,musU,q)
[nDec,nRows,nCols] = size(Y);
ps = nDec/2;
nAngles = size(theta,1);
Y = reshape(Y,nDec,[]);
for iBlk = 1:nRows*nCols
    W = orthMatrix(theta(1:nAngles/2,iBlk),musColumn(musW,iBlk),q);
    U = orthMatrix(theta(nAngles/2+1:end,iBlk),musColumn(musU,iBlk),q);
    Y(1:ps,iBlk) = W*Y(1:ps,iBlk);
    Y(ps+1:end,iBlk) = U*Y(ps+1:end,iBlk);
end
Y = reshape(Y,nDec,nRows,nCols);
end

function Y = rotateFinal(Y,theta,musW,musU,q)
[nDec,nRows,nCols] = size(Y);
ps = nDec/2;
nAngles = size(theta,1);
Y = reshape(Y,nDec,[]);
for iBlk = 1:nRows*nCols
    W = orthMatrix(theta(1:nAngles/2,iBlk),musColumn(musW,iBlk),q);
    U = orthMatrix(theta(nAngles/2+1:end,iBlk),musColumn(musU,iBlk),q);
    Y(1:ps,iBlk) = W.'*Y(1:ps,iBlk);
    Y(ps+1:end,iBlk) = U.'*Y(ps+1:end,iBlk);
end
Y = reshape(Y,nDec,nRows,nCols);
end

function Y = rotateIntermediate(Y,theta,mus,isSynthesis,q)
% Rotate the antisymmetric half; the transpose is used for synthesis
[nDec,nRows,nCols] = size(Y);
ps = nDec/2;
Y = reshape(Y,nDec,[]);
for iBlk = 1:nRows*nCols
    U = orthMatrix(theta(:,iBlk),musColumn(mus,iBlk),q);
    if isSynthesis
        U = U.';
    end
    Y(ps+1:end,iBlk) = U*Y(ps+1:end,iBlk);
end
Y = reshape(Y,nDec,nRows,nCols);
end

function mu = musColumn(mus,iBlk)
if size(mus,2) == 1
    mu = mus;
else
    mu = mus(:,iBlk);
end
end

function M = orthMatrix(angles,mus,q)
% Product of Givens rotations followed by sign flips (as fcn_orthmtxgen)
n = (1+sqrt(1+8*numel(angles)))/2;
M = eye(n,'like',angles);
iAng = 1;
for iTop = 1:n-1
    vt = M(iTop,:);
    for iBtm = iTop+1:n
        c = cos(angles(iAng));
        s = sin(angles(iAng));
        vb = M(iBtm,:);
        u = s.*(vt+vb);
        vt = (c+s).*vt-u;
        M(iBtm,:) = (c-s).*vb+u;
        iAng = iAng + 1;
    end
    M(iTop,:) = vt;
end
M = q(mus(:).*M,'rotation');
end

function s = givenStats(statistics,k)
% Element k of the given statistics, or [] for the statistics of the image
if isempty(statistics)
    s = [];
else
    s = statistics(k);
end
end

%% Angle estimator (control path)
function [theta,measured] = estimateAngles(Y,est,given,q)
[~,nRows,nCols] = size(Y);
nBlks = nRows*nCols;

% Local state extraction: neighbor blocks with circular boundary
Xc = Y(est.Channels,:,:);
nv = est.Neighbor(1);
nh = est.Neighbor(2);
F = zeros(numel(est.Channels)*nv*nh,nRows,nCols,'like',Y);
iF = 0;
nCh = numel(est.Channels);
for vshift = fix(nv/2):-1:-fix(nv/2)
    for hshift = fix(nh/2):-1:-fix(nh/2)
        F(iF+(1:nCh),:,:) = circshift(Xc,[0 vshift hshift]);
        iF = iF + nCh;
    end
end
F = reshape(F,[],nBlks);

% State standardization over all blocks of the image (or with the given
% statistics, e.g. of the previous frame)
mu = mean(F,2);
v = sum((F-mu).^2,2)/(nBlks-1);
measured = struct('Mu',mu,'Sigma',sqrt(v)+est.Epsilon);
if isempty(given)
    F = (F-measured.Mu)./measured.Sigma;
else
    F = (F-given.Mu)./given.Sigma;
end
F = q(F,'features');

% Residual estimator blocks: LayerNorm, FC, GELU (tanh), FC, skip
for iRes = 1:numel(est.ResBlocks)
    r = est.ResBlocks(iRes);
    mu = mean(F,1);
    v = mean((F-mu).^2,1);
    ln = q(r.Gamma.*((F-mu)./sqrt(v+1e-5)) + r.Beta,'ln');
    z1 = q(r.W1*ln + r.B1,'z1');
    a = q(0.5.*z1.*(1+tanh(sqrt(2/pi).*(z1+0.044715.*z1.^3))),'act');
    F = q(F + r.W2*a + r.B2,'features');
end

% Output: angles, with zero-padded leading angles for no-DC-leakage
theta = q([zeros(est.NumberOfZeroPadAngles,nBlks,'like',F); est.Wo*F + est.Bo],'angles');
end
