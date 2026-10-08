function [y,coefs,stats] = salsun2d_infer_stream(x,params,options)
%SALSUN2D_INFER_STREAM SA-LSUN inference in bands of block rows (overlap-save)
%
%   [y,coefs,stats] = salsun2d_infer_stream(x,params,Statistics=stats0)
%   computes the same quantities as salsun2d_infer(x,params,
%   Statistics=stats0), but processes the image in bands of BandRows
%   block rows, each extended by Halo block rows above and below
%   (circular, as the network's own boundary handling). Only the rows of
%   the band itself are kept: overlap-save for the analysis and, since
%   the synthesis reaches one block beyond the band, overlap-save for the
%   synthesis as well (the same result as overlap-add of the band
%   contributions). This is the reference for a streaming FPGA
%   implementation that keeps a few block rows on chip and never needs
%   the whole frame.
%
%   The standardization statistics must be given (Statistics, the struct
%   array of salsun2d_infer), typically from the previous frames; the
%   statistics of the whole image are not available while streaming.
%   The returned stats are measured on this image from the valid blocks
%   of all bands (per channel, expanded to the features, which equals the
%   per-feature statistics of salsun2d_infer up to rounding), for use
%   with the next frame.
%
%   Options:
%     Halo     - block rows of context on each side of a band (default 3).
%                The structural receptive field needs 6 for the analysis
%                plus 1 for the synthesis; with trained parameters the
%                influence decays to below 1e-3 within 2 to 3 blocks.
%     BandRows - block rows per band (default 5).
%
%   If Statistics is empty, the statistics of the whole image are
%   measured first with salsun2d_infer (two passes; not streamable).
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
    options.Halo (1,1) double {mustBeInteger,mustBePositive} = 3
    options.BandRows (1,1) double {mustBeInteger,mustBePositive} = 5
end
My = params.Stride(1);
[szy,szx] = size(x);
nRows = szy/My;
nDec = prod(params.Stride);
ests = [params.V0.Estimator, params.Stages.Estimator];
nEst = numel(ests);
H = options.Halo;
B = options.BandRows;

given = options.Statistics;
if isempty(given)
    [~,~,~,~,given] = salsun2d_infer(x,params);
end

y = zeros(szy,szx,'like',x);
coefs = zeros(nDec,nRows,szx/params.Stride(2),'like',x);
% Accumulators of the channel statistics over the valid blocks
sum1 = cell(nEst,1); sum2 = cell(nEst,1); nValid = 0;
for k = 1:nEst
    sum1{k} = zeros(numel(ests(k).Channels),1);
    sum2{k} = zeros(numel(ests(k).Channels),1);
end

for r0 = 1:B:nRows
    r1 = min(r0+B-1,nRows);
    % Block rows of the band with halo, circular
    blockRows = mod((r0-H:r1+H)-1,nRows) + 1;
    pixelRows = reshape((blockRows-1)*My + (1:My)',1,[]);
    xb = x(pixelRows,:);
    [yb,cb,~,stageInputs] = salsun2d_infer(xb,params,Statistics=given);
    validBlocks = H + (1:r1-r0+1);              % rows of the band inside xb
    validPixels = (validBlocks(1)-1)*My + (1:numel(validBlocks)*My);
    y((r0-1)*My+1:r1*My,:) = yb(validPixels,:);
    coefs(:,r0:r1,:) = cb(:,validBlocks,:);
    for k = 1:nEst
        Yk = double(reshape(stageInputs{k}(ests(k).Channels,validBlocks,:),numel(ests(k).Channels),[]));
        sum1{k} = sum1{k} + sum(Yk,2);
        sum2{k} = sum2{k} + sum(Yk.^2,2);
    end
    nValid = nValid + numel(validBlocks)*size(cb,3);
end

% Channel statistics -> feature statistics (features are circular shifts
% of the channels, so their statistics over the image are the same)
stats = repmat(struct('Mu',[],'Sigma',[]),nEst,1);
for k = 1:nEst
    mu = sum1{k}/nValid;
    v = (sum2{k} - nValid*mu.^2)/(nValid-1);
    nNeighbors = prod(ests(k).Neighbor);
    stats(k).Mu = cast(repmat(mu,nNeighbors,1),'like',x);
    stats(k).Sigma = cast(repmat(sqrt(max(v,0)),nNeighbors,1),'like',x) + ests(k).Epsilon;
end
end
