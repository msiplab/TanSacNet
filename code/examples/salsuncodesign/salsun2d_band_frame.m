function [y,mu,sigma,bands] = salsun2d_band_frame(x,w,mu,sigma,bandRows,bandFcn)
%SALSUN2D_BAND_FRAME Process a frame band by band with the HLS band design
%
%   [y,muNext,sigmaNext] = salsun2d_band_frame(x,w,mu,sigma,bandRows)
%   reconstructs the frame x (single) with salsun2d_hls_band applied to
%   bands of bandRows block rows, each with L.Halo block rows of circular
%   context, using the channel statistics mu, sigma (L.NDec x L.NEst) and
%   returns the statistics measured on this frame (for the next frame).
%   This is the host-side logic of the streaming kernel, in MATLAB.
%
%   The bands start at 1, 1+bandRows, ... and the last band is moved up
%   so that it ends at the last block row (its rows that were already
%   produced are overwritten with the same values), because the band
%   design is generated for one band size.
%
%   bandFcn (optional) replaces salsun2d_hls_band, e.g. by the MEX
%   gateway of the kernel; it must have the same interface.
%
%   bands (optional output) lists the first block row of every band.
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
    x (:,:) single
    w (:,1) single
    mu (:,:) single
    sigma (:,:) single
    bandRows (1,1) double {mustBeInteger,mustBePositive}
    bandFcn = @salsun2d_hls_band
end
L = salsun2d_hls_layout();
My = L.Stride(1);
[szy,szx] = size(x);
nRows = szy/My;
H = L.Halo;
if bandRows > nRows
    error('salsun2d_band_frame:bandRows','bandRows (%d) exceeds the block rows (%d).',bandRows,nRows)
end

bands = 1:bandRows:nRows;
if bands(end) + bandRows - 1 > nRows
    bands(end) = nRows - bandRows + 1;      % move the last band up
end

y = zeros(szy,szx,'single');
sum1 = zeros(L.NDec,L.NEst,'single');
sum2 = zeros(L.NDec,L.NEst,'single');
for r0 = bands
    blockRows = mod((r0-H:r0+bandRows-1+H)-1,nRows) + 1;
    pixelRows = reshape((blockRows-1)*My + (1:My)',1,[]);
    [yb,s1,s2] = bandFcn(x(pixelRows,:),w,mu,sigma);
    y((r0-1)*My+1:(r0+bandRows-1)*My,:) = yb;
    % The moved last band repeats rows already counted: scale its share
    if r0 == bands(end) && numel(bands) > 1 && bands(end) < bands(end-1) + bandRows
        newRows = bands(end) + bandRows - (bands(end-1) + bandRows);
        share = single(newRows/bandRows);
        % the sums of the repeated rows are not separable; approximate the
        % frame sums by the share of new rows (exact when the rows are
        % statistically alike; the next frame tolerates this error)
        s1 = s1*share;
        s2 = s2*share;
    end
    sum1 = sum1 + s1;
    sum2 = sum2 + s2;
end
nBlocks = single(nRows*(szx/L.Stride(2)));
[mu,sigma] = salsun2d_stats_from_sums(sum1,sum2,nBlocks,L.Epsilon);
end
