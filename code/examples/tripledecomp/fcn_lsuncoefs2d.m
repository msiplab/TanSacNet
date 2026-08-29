function C = fcn_lsuncoefs2d(net,data,info,miniBatchSize)
%FCN_LSUNCOEFS2D LSUN coefficients of a snapshot sequence
%
%   C = FCN_LSUNCOEFS2D(net,data,info) analyses the snapshot sequence
%   data (ny x nx x nT) with the trained LSUN analysis network net and
%   returns the untruncated coefficients
%
%       C : nBlocksY x nBlocksX x M x nT
%
%   with the DC channel first, followed by the M-1 AC channels, so that
%   info.coefMask indexes the channel dimension directly.  The truncation
%   Gamma is deliberately not applied here: the discarded channels are
%   needed to measure how much energy the retained ones capture.
%
%   See also FCN_LSUNTRAIN2D, FCN_ENERGYCONC.

arguments
    net dlnetwork
    data (:,:,:) double
    info struct
    miniBatchSize (1,1) double {mustBePositive,mustBeInteger} = 10
end

szExt = info.szExt;
[nrows,ncols,nFrames] = size(data);
assert(isequal([nrows ncols],info.szOrg), ...
    'data has size %s but the network was trained on %s.', ...
    mat2str([nrows ncols]),mat2str(info.szOrg))
dataExt = padarray(data,szExt-[nrows ncols],0,'post');

nBlocks = szExt./info.stride;
C = zeros(nBlocks(1),nBlocks(2),info.nChsTotal,nFrames);

for iStart = 1:miniBatchSize:nFrames
    idx = iStart:min(iStart+miniBatchSize-1,nFrames);
    X = reshape(dataExt(:,:,idx),szExt(1),szExt(2),1,numel(idx));
    if info.useGPU
        X = gpuArray(X);
    end
    dlX = dlarray(X,'SSCB');
    [o1,o2] = predict(net,dlX);
    if size(o1,3) == 1
        Cb = cat(3,o1,o2);
    else
        Cb = cat(3,o2,o1);
    end
    C(:,:,:,idx) = gather(extractdata(Cb));
end
end
