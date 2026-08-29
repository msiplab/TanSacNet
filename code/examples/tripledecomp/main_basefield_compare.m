%MAIN_BASEFIELD_COMPARE Effect of the base-point field estimator on the LSUN
%coefficients
%
% DORMANT.  This script needs data with a genuine random fluctuation u''.
% On the laminar Re = 100 cylinder wake it compares the coherent component
% against pure binning error, see MAIN_LIMITCYCLE_RESIDUAL, so the
% comparison does not test what it is meant to test, and the measure itself
% is degenerate there: the block DC channel, which the no-DC-leakage
% structure makes unlearnable, carries about 99 per cent of the coefficient
% energy in both conditions.  Keep the script for the day a turbulent wake
% is available; do not quote its numbers on this data.
%
% The experiment varies one thing only, the estimator of the base-point
% field mu:
%
%   (a) the plain time average,      mu = mean_t y(t)
%   (b) the phase-averaged field,    mu = <y>(phi(t))
%
% The LSUN that is trained on the residual r(t) = y(t) - mu is configured
% identically in both conditions, down to the random seed, so any
% difference in the retained energy is attributable to the base point.
%
% Run
%   cd code/examples/tripledecomp
%   matlab -licmode onlinelicensing -batch "setup; main_basefield_compare"
%
% Outputs
%   results/values_basefield.json
%   results/figures/fig_basefield_energyconc.pdf
%
% See also FCN_PHASEAVG, FCN_LSUNTRAIN2D, FCN_ENERGYCONC.

%% Configuration
rng(0)
cfg = struct( ...
    'nBins',16, ...
    'K',4, ...
    'stride',[4 4], ...
    'overlap',[3 3], ...
    'trainSubsample',4, ...
    'maxEpochs',4);

% The backward pass through the shift-variant layers costs about ten times
% the forward pass and there is no GPU here, so the basis field is trained
% on every trainSubsample-th snapshot while the coefficients are measured
% on all of them.  Both conditions get exactly the same budget and seed.
lsunopts = struct( ...
    'stride',cfg.stride, ...
    'ovlpFactor',cfg.overlap, ...
    'nCoefs',cfg.K, ...
    'maxEpochs',cfg.maxEpochs, ...
    'miniBatchSize',10, ...
    'initialLearnRate',1e-3, ...
    'noDcLeakage',true, ...
    'seed',0);

thisdir = fileparts(mfilename('fullpath'));

%% Data
disp("Loading cylinder wake snapshots ...")
[Y,gridSize] = fcn_loadcylinder();
nT = size(Y,2);
fprintf('  %d snapshots on a %d x %d grid\n',nT,gridSize(1),gridSize(2));

%% Base-point fields
timeMean = mean(Y,2);
[muPhase,phaseIdx] = fcn_phaseavg(Y,cfg.nBins);

resid = struct();
resid.timeAvg  = Y - timeMean;
resid.phaseAvg = Y - muPhase(:,phaseIdx);

%% Train one LSUN per condition and measure the coefficients
conditions = ["timeAvg" "phaseAvg"];
ec = struct();
dcNorm = struct();
ecBlocks = struct();
for cond = conditions
    fprintf('LSUN training on the %s residual ...\n',cond);
    R = reshape(resid.(cond),gridSize(1),gridSize(2),nT);
    [net,info] = fcn_lsuntrain2d(R(:,:,1:cfg.trainSubsample:end),lsunopts);
    C = fcn_lsuncoefs2d(net,R,info);
    [ec.(cond),ecBlocks.(cond),dcNorm.(cond)] = fcn_energyconc(C,info.coefMask);
    fprintf('  energy concentration = %.4f, DC energy fraction = %.4g\n', ...
        ec.(cond),dcNorm.(cond));
end

%% Report
fprintf('\n  base-point field   energy conc.   residual DC norm\n');
fprintf('  time average       %12.4f   %16.4g\n',ec.timeAvg,dcNorm.timeAvg);
fprintf('  phase average      %12.4f   %16.4g\n\n',ec.phaseAvg,dcNorm.phaseAvg);

%% Export the numbers
metrics = struct( ...
    'energyConcTimeAvg',ec.timeAvg, ...
    'energyConcPhaseAvg',ec.phaseAvg, ...
    'residualDcNormTimeAvg',dcNorm.timeAvg, ...
    'residualDcNormPhaseAvg',dcNorm.phaseAvg, ...
    'numBins',cfg.nBins, ...
    'numChans',cfg.K);
fcn_exportvalues("basefield",cfg,metrics);

%% Figure: distribution of the block-wise energy concentration
fig = figure('Visible','off','Units','centimeters','Position',[0 0 8.6 6.4]);
ax = axes(fig); %#ok<LAXES>
hold(ax,'on')
edges = linspace(0,1,41);
vTime  = ecBlocks.timeAvg(~isnan(ecBlocks.timeAvg));
vPhase = ecBlocks.phaseAvg(~isnan(ecBlocks.phaseAvg));
histogram(ax,vTime,edges,'Normalization','probability', ...
    'DisplayName','time average','FaceAlpha',0.55)
histogram(ax,vPhase,edges,'Normalization','probability', ...
    'DisplayName','phase average','FaceAlpha',0.55)
xline(ax,mean(vTime),'--','HandleVisibility','off')
xline(ax,mean(vPhase),'-','HandleVisibility','off')
hold(ax,'off')
xlabel(ax,'block-wise energy concentration')
ylabel(ax,'fraction of blocks')
legend(ax,'Location','northwest','Box','off')
box(ax,'off')
figpath = fullfile(thisdir,'results','figures','fig_basefield_energyconc.pdf');
exportgraphics(fig,figpath,'ContentType','vector')
close(fig)
fprintf('Wrote %s\n',figpath);
