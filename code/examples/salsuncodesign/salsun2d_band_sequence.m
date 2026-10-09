function [y,info] = salsun2d_band_sequence(u,params,frameFcn,options)
%SALSUN2D_BAND_SEQUENCE Streaming inference of a sequence with a band-wise frame function
%
%   [y,info] = salsun2d_band_sequence(u,params,frameFcn) processes the
%   frames u (szy x szx x N, single) one by one with
%   [yt,muMeas,sigmaMeas] = frameFcn(u(:,:,t),mu,sigma), where mu and
%   sigma (L.NDec x L.NEst) are the channel statistics used to
%   standardize the estimator inputs and muMeas, sigmaMeas are those
%   measured on the frame. frameFcn is salsun2d_band_frame with the HLS
%   band design (MATLAB), or the MEX gateway of the band kernel on the
%   card (salsun2d_band_u250).
%
%   The statistics of frame t come from the previous frames, as in
%   salsun2d_infer_sequence: Statistics = 'previous' | 'ema' | 'fir2'
%   (default), StatsRho for 'ema' (default 0.9). The first frame (and
%   the second for 'fir2') uses statistics measured by the reference
%   salsun2d_infer on the frame itself, which is the initialization of
%   the stream (a whole-frame pass on the host).
%
%   info has the fields Mu and Sigma (cells of the statistics used per
%   frame) and MuMeasured, SigmaMeasured.
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
    u (:,:,:) single
    params (1,1) struct
    frameFcn (1,1) function_handle
    options.Statistics {mustBeMember(options.Statistics,{'previous','ema','fir2'})} = 'fir2'
    options.StatsRho (1,1) double {mustBeGreaterThanOrEqual(options.StatsRho,0),mustBeLessThan(options.StatsRho,1)} = 0.9
    % Reference used for the statistics of the first frame: the
    % quantizer and the parameters of the fixed-point model of the kernel
    % (salsun2d_band_fixed_model), or the floating-point network
    options.InitParams = []
    options.InitQuantizer = @(v,tag) v
end
L = salsun2d_hls_layout();
if isempty(options.InitParams)
    options.InitParams = params;
end
init = @(x) initStats(x,options.InitParams,options.InitQuantizer);
nFrames = size(u,3);
y = zeros(size(u),'single');
muUsed = cell(nFrames,1); sigmaUsed = cell(nFrames,1);
muMeas = cell(nFrames,1); sigmaMeas = cell(nFrames,1);
smoothedMu = []; smoothedSigma = [];
rho = single(options.StatsRho);
for t = 1:nFrames
    switch options.Statistics
        case 'previous'
            [mu,sigma] = previousOrInit(muMeas,sigmaMeas,t-1,u(:,:,t),init,L);
        case 'ema'
            if isempty(smoothedMu)
                [mu,sigma] = previousOrInit(muMeas,sigmaMeas,0,u(:,:,t),init,L);
            else
                mu = smoothedMu; sigma = smoothedSigma;
            end
        case 'fir2'
            if t > 2
                mu = 0.5*(muMeas{t-1} + muMeas{t-2});
                sigma = 0.5*(sigmaMeas{t-1} + sigmaMeas{t-2});
            else
                [mu,sigma] = previousOrInit(muMeas,sigmaMeas,t-1,u(:,:,t),init,L);
            end
    end
    [y(:,:,t),muMeas{t},sigmaMeas{t}] = frameFcn(u(:,:,t),mu,sigma);
    muUsed{t} = mu; sigmaUsed{t} = sigma;
    if strcmp(options.Statistics,'ema')
        if isempty(smoothedMu)
            smoothedMu = muMeas{t}; smoothedSigma = sigmaMeas{t};
        else
            smoothedMu = rho*smoothedMu + (1-rho)*muMeas{t};
            smoothedSigma = rho*smoothedSigma + (1-rho)*sigmaMeas{t};
        end
    end
end
info.Mu = muUsed; info.Sigma = sigmaUsed;
info.MuMeasured = muMeas; info.SigmaMeasured = sigmaMeas;
end

function [mu,sigma] = previousOrInit(muMeas,sigmaMeas,t,x,init,L)
% Statistics of frame t, or, for the first frame, those of the frame
% itself measured by the reference (initialization of the stream)
if t >= 1
    mu = muMeas{t}; sigma = sigmaMeas{t};
else
    [stats,ests] = init(x);
    mu = zeros(L.NDec,L.NEst,'single');
    sigma = ones(L.NDec,L.NEst,'single');
    for k = 1:L.NEst
        ch = ests(k).Channels;
        mu(ch,k) = stats(k).Mu(1:numel(ch));
        sigma(ch,k) = stats(k).Sigma(1:numel(ch));
    end
end
end

function [stats,ests] = initStats(x,params,q)
[~,~,~,~,stats] = salsun2d_infer(x,params,Quantizer=q,EstimatorTags=true);
ests = [params.V0.Estimator, params.Stages.Estimator];
end
