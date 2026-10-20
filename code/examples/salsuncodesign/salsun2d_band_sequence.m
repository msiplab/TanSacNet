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
%   the stream (a whole-frame pass on the host). InitFast=true measures
%   them with the vectorized salsun2d_infer_fast instead (floating point,
%   about ten times faster on the CPU).
%
%   FramesPerCall=n (default 1) passes n consecutive frames at a time to
%   frameFcn (u(:,:,t:t+n-1), mu and sigma L.NDec x L.NEst x n; e.g. one
%   frame per compute unit of the card), all with the statistics of the
%   frames before t: frame t+k uses the statistics measured up to frame
%   t-1, as the rule above gives them for frame t (for 'fir2', the frames
%   t-1 and t-2). With one frame more of delay for the later frames of a
%   call, the MSE grows by less than 1% on the wave simulations.
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
    options.InitFast (1,1) logical = false
    options.FramesPerCall (1,1) double {mustBeInteger,mustBePositive} = 1
end
L = salsun2d_hls_layout();
if isempty(options.InitParams)
    options.InitParams = params;
end
if options.InitFast
    init = @(x) initStatsFast(x,params);
else
    init = @(x) initStats(x,options.InitParams,options.InitQuantizer,L);
end
nFrames = size(u,3);
y = zeros(size(u),'single');
muUsed = cell(nFrames,1); sigmaUsed = cell(nFrames,1);
muMeas = cell(nFrames,1); sigmaMeas = cell(nFrames,1);
smoothedMu = []; smoothedSigma = [];
rho = single(options.StatsRho);
for t0 = 1:options.FramesPerCall:nFrames
    idx = t0:min(t0+options.FramesPerCall-1,nFrames);
    last = t0 - 1;                     % frames measured before this call
    mu = zeros(L.NDec,L.NEst,numel(idx),'single');
    sigma = ones(L.NDec,L.NEst,numel(idx),'single');
    for k = 1:numel(idx)
        t = idx(k);
        switch options.Statistics
            case 'previous'
                [mu(:,:,k),sigma(:,:,k)] = previousOrInit(muMeas,sigmaMeas,last,u(:,:,t),init);
            case 'ema'
                if isempty(smoothedMu)
                    [mu(:,:,k),sigma(:,:,k)] = previousOrInit(muMeas,sigmaMeas,0,u(:,:,t),init);
                else
                    mu(:,:,k) = smoothedMu; sigma(:,:,k) = smoothedSigma;
                end
            case 'fir2'
                if last >= 2
                    mu(:,:,k) = 0.5*(muMeas{last} + muMeas{last-1});
                    sigma(:,:,k) = 0.5*(sigmaMeas{last} + sigmaMeas{last-1});
                else
                    [mu(:,:,k),sigma(:,:,k)] = previousOrInit(muMeas,sigmaMeas,last,u(:,:,t),init);
                end
        end
    end
    [y(:,:,idx),mM,sM] = frameFcn(u(:,:,idx),mu,sigma);
    for k = 1:numel(idx)
        t = idx(k);
        muMeas{t} = mM(:,:,k); sigmaMeas{t} = sM(:,:,k);
        muUsed{t} = mu(:,:,k); sigmaUsed{t} = sigma(:,:,k);
        if strcmp(options.Statistics,'ema')
            if isempty(smoothedMu)
                smoothedMu = muMeas{t}; smoothedSigma = sigmaMeas{t};
            else
                smoothedMu = rho*smoothedMu + (1-rho)*muMeas{t};
                smoothedSigma = rho*smoothedSigma + (1-rho)*sigmaMeas{t};
            end
        end
    end
end
info.Mu = muUsed; info.Sigma = sigmaUsed;
info.MuMeasured = muMeas; info.SigmaMeasured = sigmaMeas;
end

function [mu,sigma] = previousOrInit(muMeas,sigmaMeas,t,x,init)
% Statistics of frame t, or, for the first frame, those of the frame
% itself measured by the reference (initialization of the stream)
if t >= 1
    mu = muMeas{t}; sigma = sigmaMeas{t};
else
    [mu,sigma] = init(x);
end
end

function [mu,sigma] = initStats(x,params,q,L)
[~,~,~,~,stats] = salsun2d_infer(x,params,Quantizer=q,EstimatorTags=true);
ests = [params.V0.Estimator, params.Stages.Estimator];
mu = zeros(L.NDec,L.NEst,'single');
sigma = ones(L.NDec,L.NEst,'single');
for k = 1:L.NEst
    ch = ests(k).Channels;
    mu(ch,k) = stats(k).Mu(1:numel(ch));
    sigma(ch,k) = stats(k).Sigma(1:numel(ch));
end
end

function [mu,sigma] = initStatsFast(x,params)
[~,~,~,mu,sigma] = salsun2d_infer_fast(x,params,[],[]);
end
