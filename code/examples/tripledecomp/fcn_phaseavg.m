function [muPhase,phaseIdx,phase] = fcn_phaseavg(Y,nBins,probe)
%FCN_PHASEAVG Phase identification and phase-averaged base-point field
%
%   [muPhase,phaseIdx,phase] = FCN_PHASEAVG(Y,nBins) identifies the
%   instantaneous phase of the statistically periodic snapshot matrix
%   Y (space x time) and returns the phase-averaged base-point field
%   muPhase (space x nBins), the bin index phaseIdx (1 x nT) of every
%   snapshot and the instantaneous phase phase (1 x nT) in [-pi,pi).
%
%   FCN_PHASEAVG(Y,nBins,probe) uses the given sectional signal probe
%   (1 x nT), for instance a lift coefficient or the value of the field at
%   a probe point, instead of the default.  The default probe is the
%   leading principal component of the mean-removed snapshots, which for a
%   shedding wake is a sinusoid at the shedding frequency.
%
%   The phase is the argument of the analytic signal of the mean-removed
%   probe, obtained by a Hilbert transform.  Snapshots are then binned
%   uniformly in phase and averaged within each bin, so that muPhase(:,k)
%   estimates the ensemble average of the field over the k-th phase
%   interval.  Note that muPhase is the base-point field, that is the sum
%   of the time mean and the coherent component of the triple
%   decomposition, not the coherent component alone.
%
%   See also FCN_DIVFREEPROJ, MAIN_BASEFIELD_COMPARE.

arguments
    Y (:,:) double
    nBins (1,1) double {mustBePositive,mustBeInteger}
    probe (1,:) double = []
end

nT = size(Y,2);

%% Probe signal
if isempty(probe)
    Yc = Y - mean(Y,2);
    % Leading principal component without forming the spatial covariance
    [~,~,V] = svds(Yc,1);
    probe = V(:,1).';
else
    assert(numel(probe) == nT, ...
        'probe must have one sample per snapshot (%d).',nT)
end

%% Instantaneous phase from the analytic signal
z = fcn_analytic_(probe(:) - mean(probe));
phase = angle(z).';

%% Uniform phase bins
edgesShift = (phase + pi)/(2*pi);        % in [0,1)
phaseIdx = floor(edgesShift*nBins) + 1;
phaseIdx = min(max(phaseIdx,1),nBins);

%% Phase average
muPhase = zeros(size(Y,1),nBins);
count = zeros(1,nBins);
for k = 1:nBins
    sel = (phaseIdx == k);
    count(k) = nnz(sel);
    if count(k) > 0
        muPhase(:,k) = mean(Y(:,sel),2);
    end
end

% Empty bins fall back to the global mean so that the field stays defined
% over the whole cycle.
if any(count == 0)
    warning('tripledecomp:emptyPhaseBin', ...
        '%d of %d phase bins are empty; falling back to the time mean there.', ...
        nnz(count==0),nBins)
    muPhase(:,count==0) = repmat(mean(Y,2),1,nnz(count==0));
end
end

%% Analytic signal (local implementation, no toolbox dependency)
function z = fcn_analytic_(x)
n = numel(x);
X = fft(x);
h = zeros(n,1);
if mod(n,2) == 0
    h(1) = 1;
    h(n/2+1) = 1;
    h(2:n/2) = 2;
else
    h(1) = 1;
    h(2:(n+1)/2) = 2;
end
z = ifft(X.*h);
end
