%MEASURE_STATISTICS_STABILITY Frame-to-frame stability of the standardization statistics
%
% The first layer of every angle estimator standardizes its input channels
% with the mean and standard deviation over all blocks of the image. This
% script measures how much these statistics vary from frame to frame on
% the wave equation data, for the raw frames u and for the fluctuation
% u' = u - mean_t(u) (base-point separation), at the inputs of all five
% estimators. The data path of an untrained network (zero rotation
% angles) is used to produce the inputs of estimators 2 to 5, since no
% trained network is available yet; the block DCT (estimator 1) does not
% depend on the network.
%
% If the statistics are stable, the standardization of frame t can use
% the statistics of frame t-1 (or fixed statistics) with little error,
% which removes the whole-frame dependency and allows tiled or streamed
% processing.
%
% Requirements: MATLAB R2026b, Deep Learning Toolbox
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

here = fileparts(mfilename('fullpath'));
addpath(here,fullfile(here,'..','..'),fullfile(here,'..','salsun'));

[u,t] = salsun2d_wave_data();
[szy,szx,nFrames] = size(u);
fprintf('Wave data: %d x %d, %d frames, t = %.2f .. %.2f\n',szy,szx,nFrames,t(1),t(end));

% Untrained network (zero angles): only its data path is used
import tansacnet.salsun.*
lg = fcn_createsalsunlgraph2d([],'InputSize',[szy szx],'Stride',[4 4], ...
    'OverlappingFactor',[3 3],'NumberOfVanishingMoments',true, ...
    'NumberOfNeighborBlocks',[3 3],'NumberOfResidualBlocks',3,'Width',2, ...
    'Mode','Whole','ThetaMode','Reuse','Device','cpu');
params = salsun2d_cast_params(salsun2d_extract_params(dlnetwork(lg)),'double');
ests = [params.V0.Estimator, params.Stages.Estimator];
nEst = numel(ests);

ubar = mean(u,3);
cases = {'raw u', u; 'u'' = u - mean_t(u)', u - ubar};
for iCase = 1:size(cases,1)
    data = cases{iCase,2};
    mu = cell(nEst,1);
    sigma = cell(nEst,1);
    for k = 1:nEst
        mu{k} = zeros(numel(ests(k).Channels),nFrames);
        sigma{k} = zeros(numel(ests(k).Channels),nFrames);
    end
    for f = 1:nFrames
        [~,~,~,stageInputs] = salsun2d_infer(data(:,:,f),params);
        for k = 1:nEst
            Y = reshape(stageInputs{k}(ests(k).Channels,:,:),numel(ests(k).Channels),[]);
            mu{k}(:,f) = mean(Y,2);
            sigma{k}(:,f) = std(Y,0,2);
        end
    end

    fprintf('\n== %s\n',cases{iCase,1});
    fprintf('   %-12s %8s %8s | %22s | %22s\n','estimator','sig min','sig max', ...
        'prev frame: dsig/sig','fixed stats: dsig/sig');
    fprintf('   %-12s %8s %8s | %10s %10s | %10s %10s\n','','','','max','median','max','median');
    for k = 1:nEst
        m = mu{k}; sg = sigma{k};
        errSigma = abs(sg(:,2:end) - sg(:,1:end-1))./sg(:,2:end);
        errSigmaFixed = abs(sg - mean(sg,2))./sg;
        errMu = abs(m(:,2:end) - m(:,1:end-1))./sg(:,2:end);
        fprintf('   %-12s %8.3g %8.3g | %10.3g %10.3g | %10.3g %10.3g   (|d mu|/sig max %.2g)\n', ...
            sprintf('estimator %d',k),min(sg(:)),max(sg(:)), ...
            max(errSigma(:)),median(errSigma(:)),max(errSigmaFixed(:)),median(errSigmaFixed(:)),max(errMu(:)));
    end
end

fprintf('\nEnergy: ||ubar||^2 / mean_t ||u||^2 = %.3g\n',sum(ubar(:).^2)/mean(sum(sum(u.^2,1),2)));
