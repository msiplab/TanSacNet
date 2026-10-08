function [y,info] = salsun2d_infer_sequence(u,params,options)
%SALSUN2D_INFER_SEQUENCE SA-LSUN inference of a frame sequence, streaming style
%
%   y = salsun2d_infer_sequence(u,params) processes the frames u
%   (szy x szx x N) in order with salsun2d_infer and returns the
%   reconstructed frames y. With the default options every frame is
%   processed independently, exactly as salsun2d_infer(u(:,:,t),params).
%
%   The options reproduce what a streaming FPGA implementation can do
%   causally, without waiting for the whole frame or the whole sequence:
%
%   BaseField  - 'none' (default), 'batch' or 'iir': base field
%                separation by salsun2d_base_field. The network processes
%                the fluctuation and the base is added back to the output.
%   Rho, Scope - options of salsun2d_base_field (default 0.9, 'dc').
%   Statistics - how the estimator inputs are standardized:
%                'image'    (default) statistics of the current frame
%                           (the original layer; needs the whole frame)
%                'previous' statistics measured on the previous frame
%                'ema'      exponential moving average of the measured
%                           statistics of the previous frames,
%                           s_t = StatsRho*s_(t-1) + (1-StatsRho)*m_(t-1)
%                'fir2'     mean of the statistics of the two previous
%                           frames (a zero at the Nyquist frequency, which
%                           cancels frame-to-frame alternation)
%                The first frame (and the second for 'fir2') falls back
%                to the statistics of the image.
%   StatsRho   - forgetting factor for 'ema' (default 0.9).
%   Halo       - 0 (default): every frame is processed as a whole.
%                Otherwise the frames are processed in bands of BandRows
%                block rows with Halo block rows of context
%                (salsun2d_infer_stream). The first frame, and the second
%                for 'fir2', are still processed as a whole to obtain
%                initial statistics; Statistics='image' is not available
%                with Halo > 0.
%   BandRows   - block rows per band for Halo > 0 (default 5).
%
%   [y,info] = salsun2d_infer_sequence(...) also returns info with fields
%   BaseField (b), Fluctuation (u - b), Statistics (measured statistics
%   of every frame, cell array) and Coefs (masked coefficients, cell).
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
    u (:,:,:) {mustBeFloat}
    params (1,1) struct
    options.BaseField {mustBeMember(options.BaseField,{'none','batch','iir'})} = 'none'
    options.Rho (1,1) double = 0.9
    options.Scope {mustBeMember(options.Scope,{'dc','full'})} = 'dc'
    options.Statistics {mustBeMember(options.Statistics,{'image','previous','ema','fir2'})} = 'image'
    options.StatsRho (1,1) double {mustBeGreaterThanOrEqual(options.StatsRho,0),mustBeLessThan(options.StatsRho,1)} = 0.9
    options.Halo (1,1) double {mustBeInteger,mustBeNonnegative} = 0
    options.BandRows (1,1) double {mustBeInteger,mustBePositive} = 5
end
if options.Halo > 0 && strcmp(options.Statistics,'image')
    error('salsun2d_infer_sequence:imageStatistics', ...
        'Statistics=''image'' needs the whole frame; use ''previous'', ''ema'' or ''fir2'' with Halo > 0.')
end

[uf,b] = salsun2d_base_field(u,Method=options.BaseField,Rho=options.Rho, ...
    Scope=options.Scope,Stride=params.Stride);

nFrames = size(u,3);
y = zeros(size(u),'like',u);
measured = cell(nFrames,1);
coefs = cell(nFrames,1);
smoothed = [];
for t = 1:nFrames
    switch options.Statistics
        case 'image'
            given = [];
        case 'previous'
            given = previousOrEmpty(measured,t-1);
        case 'ema'
            given = smoothed;
        case 'fir2'
            if t > 2
                given = combineStats(measured{t-1},measured{t-2},0.5,0.5);
            else
                given = previousOrEmpty(measured,t-1);
            end
    end
    if options.Halo > 0 && ~isempty(given)
        [y(:,:,t),coefs{t},measured{t}] = salsun2d_infer_stream(uf(:,:,t),params, ...
            Statistics=given,Halo=options.Halo,BandRows=options.BandRows);
    else
        [y(:,:,t),coefs{t},~,~,measured{t}] = salsun2d_infer(uf(:,:,t),params,Statistics=given);
    end
    y(:,:,t) = y(:,:,t) + b(:,:,t);
    if strcmp(options.Statistics,'ema')
        if isempty(smoothed)
            smoothed = measured{t};
        else
            smoothed = combineStats(smoothed,measured{t},options.StatsRho,1-options.StatsRho);
        end
    end
end

info.BaseField = b;
info.Fluctuation = uf;
info.Statistics = measured;
info.Coefs = coefs;
end

function s = previousOrEmpty(measured,t)
if t >= 1
    s = measured{t};
else
    s = [];
end
end

function s = combineStats(s1,s2,w1,w2)
% Elementwise w1*s1 + w2*s2 of the per-estimator statistics
s = s1;
for k = 1:numel(s1)
    s(k).Mu = w1*s1(k).Mu + w2*s2(k).Mu;
    s(k).Sigma = w1*s1(k).Sigma + w2*s2(k).Sigma;
end
end
