function [y,coefs] = salsun2d_infer_klt(x,params,options)
%SALSUN2D_INFER_KLT SA-LSUN 2-D with local KLT rotations instead of the estimators
%
%   y = salsun2d_infer_klt(x,params) runs the analysis, the coefficient
%   mask and the synthesis of salsun2d_infer with the same block DCT, atom
%   extensions and mask, but each rotation is the local KLT (local block
%   PCA) of the coefficients it rotates instead of the rotation of the
%   learned angle estimator: for every block, the second-moment matrix of
%   the rotated channels over the neighboring blocks is diagonalized and
%   its eigenvectors, in decreasing order of the eigenvalues, are the rows
%   of the rotation. No parameter is learned. The initial rotation keeps
%   the DC channel (no DC leakage) and rotates channels 2..Ps and
%   Ps+1..2Ps; each intermediate stage rotates channels Ps+1..2Ps. The
%   synthesis uses the transposes, so without the mask the reconstruction
%   is perfect.
%
%   Options:
%     Neighbor      - neighborhood in blocks, [3 3] (default) as the
%                     estimators, or larger
%     ExcludeCenter - leave the block itself out of its neighborhood
%                     (default false)
%     Ridge         - relative regularization of the second-moment
%                     matrix (default 1e-6 of its trace)
%
%   This is a study of a parametric replacement of the angle estimators;
%   in hardware the rotations of the local KLT would come from Jacobi
%   iterations, which produce Givens rotations directly.
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
    options.Neighbor (1,2) double {mustBeInteger,mustBePositive} = [3 3]
    options.ExcludeCenter (1,1) logical = false
    options.Ridge (1,1) double {mustBeNonnegative} = 1e-6
end
dec = params.Stride;
nDec = prod(dec);
ps = nDec/2;
x = double(x);
Cvh = double(params.Cvh);

Y = blockDct(x,Cvh,dec);
[~,nRows,nCols] = size(Y);
nBlks = nRows*nCols;
Y = reshape(Y,nDec,nRows,nCols);

% Initial rotation: W on channels 2..ps (DC kept), U on ps+1..nDec
W0 = localKlt(Y(2:ps,:,:),options);
U0 = localKlt(Y(ps+1:end,:,:),options);
Y = reshape(Y,nDec,nBlks);
Y(2:ps,:) = rotate(W0,Y(2:ps,:),false);
Y(ps+1:end,:) = rotate(U0,Y(ps+1:end,:),false);
Y = reshape(Y,nDec,nRows,nCols);

% Intermediate stages
nStages = numel(params.Stages);
Us = cell(nStages,1);
for iStage = 1:nStages
    stage = params.Stages(iStage);
    Y = atomExtension(Y,stage.Shift,stage.Target);
    Us{iStage} = localKlt(Y(ps+1:end,:,:),options);
    Y = reshape(Y,nDec,nBlks);
    Y(ps+1:end,:) = rotate(Us{iStage},Y(ps+1:end,:),false);
    Y = reshape(Y,nDec,nRows,nCols);
end

% Coefficient mask
Y = Y.*double(params.Mask(:));
coefs = Y;

% Synthesis with the transposes
for iStage = nStages:-1:1
    stage = params.Stages(iStage);
    Y = reshape(Y,nDec,nBlks);
    Y(ps+1:end,:) = rotate(Us{iStage},Y(ps+1:end,:),true);
    Y = reshape(Y,nDec,nRows,nCols);
    Y = atomExtension(Y,stage.SynShift,stage.SynTarget);
end
Y = reshape(Y,nDec,nBlks);
Y(2:ps,:) = rotate(W0,Y(2:ps,:),true);
Y(ps+1:end,:) = rotate(U0,Y(ps+1:end,:),true);
y = cast(blockIdct(reshape(Y,nDec,nRows,nCols),Cvh,dec),'like',x);
end

function Q = localKlt(Yc,options)
% Q(:,:,b): rows are the eigenvectors of the second-moment matrix of the
% channels Yc over the neighborhood of block b, in decreasing order
[n,nRows,nCols] = size(Yc);
nv = options.Neighbor(1);
nh = options.Neighbor(2);
samples = zeros(n,0,nRows*nCols);
for vshift = fix(nv/2):-1:-fix(nv/2)
    for hshift = fix(nh/2):-1:-fix(nh/2)
        if options.ExcludeCenter && vshift == 0 && hshift == 0
            continue
        end
        s = reshape(circshift(Yc,[0 vshift hshift]),n,1,[]);
        samples = cat(2,samples,s);
    end
end
R = pagemtimes(samples,'none',samples,'transpose');
tr = sum(R.*eye(n),[1 2]);
R = R + (options.Ridge*tr/n + eps).*eye(n);
[V,D] = pageeig(R);
Q = zeros(n,n,size(R,3));
for b = 1:size(R,3)
    [~,order] = sort(real(diag(D(:,:,b))),'descend');
    Q(:,:,b) = real(V(:,order,b)).';
end
end

function Y = rotate(Q,Y,transposed)
% Y(:,b) = Q(:,:,b)*Y(:,b), or Q(:,:,b).'*Y(:,b)
n = size(Y,1);
y3 = reshape(Y,n,1,[]);
if transposed
    y3 = pagemtimes(Q,'transpose',y3,'none');
else
    y3 = pagemtimes(Q,y3);
end
Y = reshape(y3,n,[]);
end

function Y = blockDct(x,Cvh,dec)
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

function Y = atomExtension(Y,shift,target)
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
