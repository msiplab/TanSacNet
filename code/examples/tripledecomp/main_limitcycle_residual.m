%MAIN_LIMITCYCLE_RESIDUAL Characterise the flow as a clean limit cycle
%
% The tangent-space reading of LSUN-C assumes that the coherent component
% traces a smooth low-dimensional limit cycle and that what the phase
% average leaves behind is discretisation error rather than a random
% fluctuation.  This script checks that assumption on the data actually
% used, before any network is trained.
%
% If a genuine fluctuation u'' were present, the residual of the
% phase-averaged base-point field would stop decreasing once the phase bins
% resolve the cycle.  Instead the residual keeps falling like the square of
% the bin width, which identifies it as within-bin phase variation.  The
% flow is therefore deterministic and periodic to the accuracy of the data,
% which is the premise of MAIN_COEFFDYN_UNITCIRCLE and, equally, the reason
% why this data cannot support any claim about the random fluctuation.
%
% Run
%   cd code/examples/tripledecomp
%   matlab -licmode onlinelicensing -batch "setup; main_limitcycle_residual"
%
% Outputs
%   results/values_limitcycle.json
%   results/figures/fig_limitcycle_residual.pdf
%
% See also FCN_PHASEAVG, MAIN_COEFFDYN_UNITCIRCLE.

%% Configuration
rng(0)
cfg = struct( ...
    'nBins',16, ...                       % operating point of the other scripts
    'binSweep',[4 8 16 32 64]);

thisdir = fileparts(mfilename('fullpath'));

%% Data
disp("Loading cylinder wake snapshots ...")
[Y,gridSize] = fcn_loadcylinder();
nT = size(Y,2);
fprintf('  %d snapshots on a %d x %d grid\n',nT,gridSize(1),gridSize(2));

timeMean = mean(Y,2);
Efluc = sum((Y-timeMean).^2,'all');

%% Residual of the phase-averaged base-point field against the bin count
sweep = cfg.binSweep;
ratio = zeros(size(sweep));
fprintf('\n  nBins   residual / fluctuation energy\n');
for k = 1:numel(sweep)
    [mu,idx] = fcn_phaseavg(Y,sweep(k));
    R = Y - mu(:,idx);
    ratio(k) = sum(R.^2,'all')/Efluc;
    fprintf('  %5d   %22.5f\n',sweep(k),ratio(k));
end

% A residual made of within-bin phase variation decays like the square of
% the bin width; a genuine random fluctuation would level off.
p = polyfit(log(sweep(:)),log(ratio(:)),1);
decayExponent = p(1);
residualAtOperatingPoint = ratio(sweep == cfg.nBins);
fprintf('\n  fitted decay exponent = %.3f (quadratic decay would give -2)\n',decayExponent);

%% Dimensionality of the fluctuation
s = svd(Y-timeMean,'econ');
cumEnergy = cumsum(s.^2)/sum(s.^2);
modesNinetyNine = find(cumEnergy >= 0.99,1);
modesFourNines = find(cumEnergy >= 0.9999,1);
fprintf('  POD modes for 99%% of the fluctuation energy   : %d\n',modesNinetyNine);
fprintf('  POD modes for 99.99%% of the fluctuation energy: %d\n\n',modesFourNines);

%% Export the numbers
metrics = struct( ...
    'residualRatio',residualAtOperatingPoint, ...
    'decayExponent',decayExponent, ...
    'podModesNinetyNine',modesNinetyNine, ...
    'podModesFourNines',modesFourNines, ...
    'numBins',cfg.nBins);
fcn_exportvalues("limitcycle",cfg,metrics);

%% Figure: the residual decays with the bin width, it does not level off
fig = figure('Visible','off','Units','centimeters','Position',[0 0 8.6 5.4]);
ax = axes(fig); %#ok<LAXES>
loglog(ax,sweep,ratio,'o-','MarkerSize',5,'DisplayName','measured residual')
hold(ax,'on')
ref = ratio(1)*(sweep/sweep(1)).^(-2);
loglog(ax,sweep,ref,'--','Color',[0.5 0.5 0.5], ...
    'DisplayName','quadratic decay')
hold(ax,'off')
grid(ax,'on')
xlim(ax,[min(sweep)/1.3 max(sweep)*1.3])
xticks(ax,sweep)
xlabel(ax,'number of phase bins')
ylabel(ax,'residual / fluctuation energy')
legend(ax,'Location','southwest','Box','off')
box(ax,'off')
figpath = fullfile(thisdir,'results','figures','fig_limitcycle_residual.pdf');
exportgraphics(fig,figpath,'ContentType','vector')
close(fig)
fprintf('Wrote %s\n',figpath);
