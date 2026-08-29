%MAIN_COEFFDYN_UNITCIRCLE Claim 2: the LSUN coefficients of the coherent
%component follow a unitary linear model
%
% The coherent component of the triple decomposition traces a limit cycle,
% which is neutrally stable.  Expressed in the tangent-space coordinates of
% an LSUN, the map that advances one phase bin to the next should therefore
% be an isometry.  The test is not that the constrained identification
% produces eigenvalues on the unit circle, which it does by construction,
% but that the unconstrained one does so as well.
%
% Run
%   cd code/examples/tripledecomp
%   matlab -licmode onlinelicensing -batch "setup; main_coeffdyn_unitcircle"
%
% Outputs
%   results/values_coeffdyn.json
%   results/figures/fig_coeffdyn_unitcircle.pdf
%
% See also FCN_PHASEAVG, FCN_LSUNTRAIN2D, PIDMD.

%% Configuration
rng(0)
cfg = struct( ...
    'nBins',16, ...
    'K',4, ...
    'stride',[4 4], ...
    'overlap',[3 3], ...
    'nModesPod',10);

lsunopts = struct( ...
    'stride',cfg.stride, ...
    'ovlpFactor',cfg.overlap, ...
    'nCoefs',cfg.K, ...
    'maxEpochs',10, ...
    'miniBatchSize',8, ...
    'initialLearnRate',1e-3, ...
    'noDcLeakage',true, ...
    'seed',0);

thisdir = fileparts(mfilename('fullpath'));
assert(exist('piDMD','file')==2, ...
    'piDMD is not on the path. Run setup first.')

%% Data and triple decomposition
disp("Loading cylinder wake snapshots ...")
[Y,gridSize] = fcn_loadcylinder();

timeMean = mean(Y,2);
muPhase = fcn_phaseavg(Y,cfg.nBins);
coherent = muPhase - timeMean;          % space x nBins, one closed cycle

%% LSUN-C on the coherent component
disp("LSUN-C training on the coherent component ...")
Uc = reshape(coherent,gridSize(1),gridSize(2),cfg.nBins);
[net,info] = fcn_lsuntrain2d(Uc,lsunopts);
C = fcn_lsuncoefs2d(net,Uc,info);

% Keep the retained channels only and vectorise
keep = logical(info.coefMask);
Ckept = C(:,:,keep,:);
X = reshape(Ckept,[],cfg.nBins);

%% Snapshot pairs around the closed cycle
Xk  = X;
Xk1 = X(:,[2:end 1]);                   % the cycle closes on itself

r = min(cfg.nModesPod,rank(Xk));
fprintf('  coefficient dimension %d, cycle length %d, POD rank %d\n', ...
    size(X,1),cfg.nBins,r);

%% Identification, unconstrained and unitary
[Aex,valsExact] = piDMD(Xk,Xk1,'exact',r);
[Aorth,valsOrth] = piDMD(Xk,Xk1,'orthogonal',r);

devExact = abs(abs(valsExact)-1);
devOrth  = abs(abs(valsOrth)-1);

fitExact = norm(Xk1-Aex(Xk),'fro')/norm(Xk1,'fro');
fitOrth  = norm(Xk1-Aorth(Xk),'fro')/norm(Xk1,'fro');

fprintf('\n  unconstrained: max |lambda|-1| = %.4g, relative fit error = %.4g\n', ...
    max(devExact),fitExact);
fprintf('  unitary      : max |lambda|-1| = %.4g, relative fit error = %.4g\n\n', ...
    max(devOrth),fitOrth);

%% Export the numbers
metrics = struct( ...
    'maxUnitCircleDeviation',max(devExact), ...
    'medianUnitCircleDeviation',median(devExact), ...
    'maxUnitCircleDeviationUnitary',max(devOrth), ...
    'relFitErrorExact',fitExact, ...
    'relFitErrorUnitary',fitOrth, ...
    'numBins',cfg.nBins, ...
    'numChans',cfg.K, ...
    'numPodModes',r);
fcn_exportvalues("coeffdyn",cfg,metrics);

%% Figure: eigenvalues and their distance to the unit circle
fig = figure('Visible','off','Units','centimeters','Position',[0 0 8.6 4.4]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(tl);
th = linspace(0,2*pi,512);
plot(ax1,cos(th),sin(th),'-','Color',[0.6 0.6 0.6],'HandleVisibility','off')
hold(ax1,'on')
plot(ax1,real(valsExact),imag(valsExact),'o','MarkerSize',4, ...
    'DisplayName','unconstrained')
plot(ax1,real(valsOrth),imag(valsOrth),'x','MarkerSize',5, ...
    'DisplayName','unitary')
hold(ax1,'off')
axis(ax1,'equal')
xlim(ax1,[-1.2 1.2]); ylim(ax1,[-1.2 1.2])
xlabel(ax1,'Re \lambda'); ylabel(ax1,'Im \lambda')
legend(ax1,'Location','southoutside','Box','off','Orientation','horizontal')
box(ax1,'off')

ax2 = nexttile(tl);
histogram(ax2,devExact,'Normalization','count','FaceAlpha',0.7)
xlabel(ax2,'| |\lambda| - 1 |')
ylabel(ax2,'count')
box(ax2,'off')

figpath = fullfile(thisdir,'results','figures','fig_coeffdyn_unitcircle.pdf');
exportgraphics(fig,figpath,'ContentType','vector')
close(fig)
fprintf('Wrote %s\n',figpath);
