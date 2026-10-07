%[text] # **Analysis of the Burgers equation by LSUN**
%[text] First run TanSacNet/code/setpath
%[text] Requirements: MATLAB R2025a or later
%[text] Contact address: Shogo MURAMATSU, Faculty of Engineering, Niigata University, 8050 2-no-cho Ikarashi, Nishi-ku, Niigata, 950-2181, JAPAN http://msiplab.eng.niigata-u.ac.jp  
%[text] Copyright (c) 2023, Hayato Obara and Shogo MURAMATSU, All rights reserved.
clc, clear
close all

%[text] Parameters


nCoefs = 1; 
nDelay = 1;

pt = 2;
stride = 2*pt; % Stride (even)


%%
%[text] Setting the conditions for the Burgers equation
mu = 1;                 % advection coefficient (fixed at 1)
nu = 0.05;              % Viscosity coefficient (corresponding to ν in the differential equation)
% Spatial Mesh
L_x = 10;               % Maximum value in space direction
dx = 0.1; 
N_x = floor(L_x/dx);    % Total number of meshes in spatial direction
X = linspace(0,L_x,N_x);% Coordinates
% Temporal Mesh
L_t = 10;               % Maximum value in time direction
dt = 0.1; 
N_t = floor(L_t/dt);    % Total number of meshes in time direction
T = linspace(0,L_t,N_t);% Coordinates
% Wave number discretization
k = 2*pi*fftfreq(N_x, dx);
% initial condition 
u0 = exp(-(X-3).^2/2); 
%u0 = np.sin(2*np.pi*X/L_x)
ndim = 100;
%[text] ## **Data preparation**
%PDE resolution (ODE system resolution)
opt = odeset('MaxStep',5000);
[~,DataT] = ode45(@(t,u) burg_system(u,t,k,mu,nu),T,u0,opt);
%[text] ## **Data visualisation**
figure
orangered = [255 69 0]/255;
disp_plot3_(T,X,DataT,orangered)

figure
disp_imagesc_(T,X,DataT)
%%
%[text] ## Configuration
%[text] Data size
nT = size(DataT,1);
nX = size(DataT,2);
%[text] Stride (block size)
assert(mod(nX,stride)==0,'stride must be a divisor of nX.');
%[text] Output dimension (per block)

assert(nCoefs<=stride,'nCoefs must be less than or equal to stride.')
%[text] Number of overlapping blocks (number of shifts)
nof = 1;
kx = 2*nof+1; % Number of overlapping blocks (odd)
%[text] ## Setting display
strbuf = "-- Settings --" + newline;
strbuf = strbuf.append("Data size (space): " + num2str(nX) + newline);
strbuf = strbuf.append("Data size (time): " + num2str(nT) + newline);
strbuf = strbuf.append("Block size: " + num2str(stride) + newline);
strbuf = strbuf.append("Output dimension (per block): " + num2str(nCoefs) + newline);
strbuf = strbuf.append("Number of overlapping blocks: " + num2str(kx) + newline);
disp(strbuf)
%%
%[text] ### One-dimensional locally structured unitary networks (1-D LSUNs)
%[text] References.
%[text] -  Lu Gan and Kai-Kuang Ma, "On simplified order-one factorizations of paraunitary filterbanks," in  IEEE Transactions on Signal Processing, vol. 52, no. 3, pp. 674-686, March 2004, doi: 10.1109/ TSP.2003.822356. \
%[text] **Original PUFB configuration**
%[text] Even-channel real coefficient symmetric delay decomposition (Real SDF) configuration \[Fig. 7 (b), Gan et al,. IEEE T-SP, 2004\]
%[text] - Number of channels M = 2m
%[text] - r\_k = m \
%[text] Number of stages (k-1) (polyphase order N) set to even 2n, allowing for spatial non-causality
%[text] Modified non-causal PUFB
%[text] ### LSUN extension of modified non-causal PUFB
%[text] **Custom network construction**
%[text] \- Defining custom deep learning layers - MATLAB & Simulink - MathWorks United Kingdom
import tansacnet.lsun.*
analysislgraph = fcn_createcslsunlgraph1d([],...
 'InputSize',nX,...
 'Stride',stride,...
 'OverlappingFactor',kx,...
 'Mode','Analyzer');
synthesislgraph = fcn_createcslsunlgraph1d([],...
 'InputSize',nX,...
 'Stride',stride,...
 'OverlappingFactor',kx,...
 'Mode','Synthesizer');
figure
subplot(1,2,1)
plot(analysislgraph)
title('Analysis LSUN')
subplot(1,2,2)
plot(synthesislgraph)
title('Synthesis LSUN')

%[text] **Initialisation of design parameters**
% Standard deviation of initial angles
stdInitAng = 1e-9;
% Construction of synthesis network.
analysisnet = dlnetwork(analysislgraph);
% Initialize
nLearnables = height(analysisnet.Learnables);
expanalyzer = '^Lv\d+_Cmp\d+_Q(\w\d|0)+(\w)+$';
nLayers = height(analysislgraph.Layers);
for iLearnable = 1:nLearnables
     if analysisnet.Learnables.Parameter(iLearnable)=="Angles"
         alayerName = analysisnet.Learnables.Layer(iLearnable);
         if ~isempty(regexp(alayerName,expanalyzer,'once'))
         disp("Angles in " + alayerName + " are set to N(-pi/2,"+num2str(stdInitAng^2)+")")
         analysisnet.Learnables.Value(iLearnable) = ...
         cellfun(@(x) x+stdInitAng*randn(size(x))-pi/2, ...
         analysisnet.Learnables.Value(iLearnable),'UniformOutput',false);
         else
         disp("Angles in " + alayerName + " are set to N(0,"+num2str(stdInitAng^2)+")")
         analysisnet.Learnables.Value(iLearnable) = ...
         cellfun(@(x) x+stdInitAng*randn(size(x)), ...
         analysisnet.Learnables.Value(iLearnable),'UniformOutput',false);
         end
     end
end
%[text] **Establishment of concomitant relationships**
%[text] **Copying design parameters**
import tansacnet.lsun.*
% Construction of analysis network
analysislgraph = layerGraph(analysisnet);
synthesislgraph = fcn_cpparamsana2syn(synthesislgraph,analysislgraph);
synthesislgraph = fcn_cpparamsana2syn_csax_(synthesislgraph,analysislgraph);
synthesisnet = dlnetwork(synthesislgraph);
%[text] **Confirmation of the adjoint relationship (complete reconstruction).**
x = rand([1 nX 1 nT],'double');
dlx = dlarray(x,"SSCB"); % Deep learning array (SSCB)
[dls{1:2}] = analysisnet.predict(dlx);
dly = synthesisnet.predict(dls{:});
mse_ = mse(dlx,dly);
display("MSE: " + num2str(mse_))
assert(mse_<1e-6)
%%
%[text] **Design parameter optimisation and signal approximation.**
import tansacnet.lsun.*
analysislgraph = layerGraph(analysisnet);
% Coefficient masking
nChsTotal = prod(stride);
coefMask = reshape([ones(nCoefs,1); zeros(nChsTotal-nCoefs,1)],2,[]).';
coefMask = coefMask(:);
%nLevels = 1;
%for iLv = nLevels:-1:1
iLv = 1;
strLv = sprintf('Lv%0d_',iLv);
% For AC
analysislgraph = analysislgraph.replaceLayer([strLv 'AcOut'],...
 mask1dLayer('Name',[strLv 'AcMask'],'Mask',coefMask(2:end),...
 'NumberOfChannels',nChsTotal-1));
%strLvPre = strLv;
%end
% Output layer
iCmp = 1;
strCmp = sprintf('Cmp%0d_',iCmp);
%analysislgraph = analysislgraph.addLayers([...
% lsunChannelConcatenation1dLayer('Name',[strLv strCmp 'Cn']) ...
% lsunRegressionLayer('Coefficient output')
% ]);
analysislgraph = analysislgraph.addLayers(...
 lsunChannelConcatenation1dLayer('Name',[strLv strCmp 'Cn']));
analysislgraph = analysislgraph.connectLayers(...
 [strLv 'AcMask' ], [strLv strCmp 'Cn/ac']);
analysislgraph = analysislgraph.connectLayers(...
 [strLv 'DcOut' ], [strLv strCmp 'Cn/dc']);
figure
plot(analysislgraph)
title('Analysis LSUN')
%%
%[text] **Reads numerical sequence data as 1-D images from a datastore.**
% Load sequences
arrds = arrayDatastore(DataT,"ReadSize",1,"IterationDimension",1);
%arrds = transform(arrds,@(x) cell2mat(x));
figure
arr = cell2mat(preview(arrds));
for idx = 1:height(arr)
 plot(arr(idx,:))
 hold on
end
hold off
%%
%[text] **Design preparation**
dlX = dlarray(gpuArray(arr(1,:)),"SSCB");
trainnet = dlnetwork(analysislgraph,dlX);
assert(trainnet.Initialized)
figure
monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Iteration");
%%
%[text] **Optimisation design**
%[text] **Parameters for learning**
%[text] \- Creating mini-batches for deep learning - MATLAB - MathWorks United Kingdom
numEpochs = 100; 
miniBatchSize = nT;
numObservationsTrain = nT;
numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;
% Minibatch
mbq = minibatchqueue(arrds,...
 "MinibatchSize",miniBatchSize,...
 "MiniBatchFcn",@(x) permute(cell2mat(x),[3 2 4 1]),...
 "OutputAsDlarray",1,...
 "OutputCast","double",...
 "MiniBatchFormat", "SSCB",...
 "OutputEnvironment", "gpu",...
 "PartialMiniBatch","discard");
% Training
averageGrad = [];
averageSqGrad = [];
iteration = 0;
epoch = 0;
start = tic;
% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
     epoch = epoch + 1;
     % Shuffle data.
     shuffle(mbq);
     % Loop over mini-batches.
     while hasdata(mbq) && ~monitor.Stop
     iteration = iteration + 1;
     % Read mini-batch of data.
     dlX = next(mbq);
     % Evaluate the model gradients, state, and loss using dlfeval and the
     % modelGradients function and update the network state.
    [loss,grad] = dlfeval(@modelLoss,trainnet,dlX);
     %Update the network parameters using the Adam optimizer.
     [trainnet,averageGrad,averageSqGrad] = ...
     adamupdate(trainnet,grad,averageGrad,averageSqGrad,iteration);
     % Display the training progress.
     recordMetrics(monitor,iteration,Loss=loss);
     updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
     monitor.Progress = 100 * iteration/numIterations;
     end
end
%%
%[text] ### Trained Analysis LSUN
import tansacnet.lsun.*
analsunlgraph = fcn_createcslsunlgraph1d([],...
    'InputSize',nX,...
    'Stride',stride,...
    'OverlappingFactor',kx,...
    ...'NumberOfVanishingMoments',noDcLeakage,...
    'Mode','Analyzer');
synlsunlgraph = fcn_createcslsunlgraph1d([],...
    'InputSize',nX,...
    'Stride',stride,...
    'OverlappingFactor',kx,...
    ...'NumberOfVanishingMoments',noDcLeakage,...
    'Mode','Synthesizer');
trainlgraph = layerGraph(trainnet);
% Trained net -> Analyzer
synlsunlgraph = fcn_cpparamsana2syn(synlsunlgraph,trainlgraph);
synlsunlgraph = fcn_cpparamsana2syn_csax_(synlsunlgraph,trainlgraph);
% Analyzer -> Synthesizer
analsunlgraph = fcn_cpparamssyn2ana(analsunlgraph,synlsunlgraph);
analsunlgraph = fcn_cpparamssyn2ana_csax_(analsunlgraph,synlsunlgraph);
%[text] 
nLevels = 1;
%for iLv = nLevels:-1:1S
iLv = 1;

% !!! 完全再構成を確認するためマスク処理を無効化
% coefMask = ones(nChsTotal,1);

strLv = sprintf('Lv%0d_',iLv);
% For analyzer
analsunlgraph = analsunlgraph.replaceLayer([strLv 'DcOut'],...
    regressionLayer('Name',[strLv 'DcOut']));
%analsunlgraph = analsunlgraph.addLayers(...
%    mask1dLayer('Name',[strLv 'AcMask'],'Mask',coefMask(2:end),...
%    'NumberOfChannels',nChsTotal-1));
%analsunlgraph = analsunlgraph.connectLayers([strLv 'AcOut'],[strLv 'AcMask']);
analsunlgraph = analsunlgraph.replaceLayer([strLv 'AcOut'], ...
        mask1dLayer('Name',[strLv 'AcMask'],'Mask',coefMask(2:end),...
    'NumberOfChannels',nChsTotal-1));
analsunlgraph = analsunlgraph.addLayers(regressionLayer('Name',[strLv 'AcOut']));
analsunlgraph = analsunlgraph.connectLayers([strLv 'AcMask'],[strLv 'AcOut']);

% For synthesizer
synlsunlgraph = synlsunlgraph.replaceLayer([strLv 'Out'],...
    regressionLayer('Name',[strLv 'Out']));
%
synlsunlgraph = synlsunlgraph.disconnectLayers([strLv 'Ac feature input'],[strLv 'AcIn']);
synlsunlgraph = synlsunlgraph.addLayers(...
    mask1dLayer('Name',[strLv 'AcMask'],'Mask',coefMask(2:end),...
    'NumberOfChannels',nChsTotal-1));
synlsunlgraph = synlsunlgraph.connectLayers([strLv 'Ac feature input'],[strLv 'AcMask']);
synlsunlgraph = synlsunlgraph.connectLayers([strLv 'AcMask'],[strLv 'AcIn']);

%strLvPre = strLv;
%end

%
figure
subplot(1,2,1)
plot(analsunlgraph)
title('Analysis LSUN')
subplot(1,2,2)
plot(synlsunlgraph)
title('Synthesis LSUN')
% Replace invalid linear layers with empty lattice parameters
fcn_replace_emptyangles_(analsunlgraph)
fcn_replace_emptyangles_(synlsunlgraph)

%%
analsunlgraph.Layers
synlsunlgraph.Layers

%%
analsunnet = assembleNetwork(analsunlgraph)
synlsunnet = assembleNetwork(synlsunlgraph)
%%
%[text] ### Analysis process
analsunseq = zeros(nT,stride,ndim/stride,'like',DataT);
for iT = 1:nT
    [dc,ac] = analsunnet.predict(gpuArray(DataT(iT,:))); % dc: 1 x Pos., ac: Ch. x 1 x Pos
    analsunseq(iT,:,:) = cat(2,permute(dc,[3 1 2]),permute(ac,[2 1 3]));
end
% 3-D array w/ Time x Ch. x Pos.
size(analsunseq)
%%
%[text] ### Synthesis process
synlsunnet.Layers(11).InputSize
synlsunnet.Layers(12).InputSize
synlsunseq = analsunseq;
DataTaprx = zeros(size(DataT),'like',DataT);
for iT = 1:nT
    ac = permute(synlsunseq(iT,2:end,:),[2 1 3]);   
    dc = synlsunseq(iT,1,:);
    DataTaprx(iT,:) = synlsunnet.predict(ac,dc);    
end
%%
figure
disp_plot3_(T,X,DataTaprx)
figure
disp_imagesc_(T,X,DataTaprx)
% MSE評価
mse(DataT,DataTaprx)
%%
%[text] ### td-DMD

nRange = size(analsunseq,1); 

size(analsunseq) % K x Ch. x Pos.
matS = eye(stride);
matS = matS(coefMask==1,:)
dataY = pagemtimes(matS, permute(analsunseq,[2 1 3])); % Ch. x K x Pos.
dataX = reshape(ipermute(dataY,[2 1 3]),nRange,[]) % K x (Ch.Pos.)
%%
% !!! By YASAS

% Construction of Hankel matrix
%{
analsunseq_ = permute(analsunseq,[2,3,1]);
dcac_ = zeros(1,size(analsunseq_,2)*2,100);
for i = 0:size(analsunseq_,2)-1
    dcac_(1,(i*2)+1,:) = analsunseq_(1,i+1,:);
    dcac_(1,(i*2)+2,:) = analsunseq_(3,i+1,:);
end
x = squeeze(dcac_).';
%x = reshape(dcac_,[size(analsunseq_,2)*2,100]).';
%}
%%

% Construction of Hankel matrix
x = dataX;

ts = [];
H = [];
%nRange = size(x,1)-1;
nRange = size(x,1);
for k = 0:nRange-nDelay-1
    xkT = [];
    for iDelay = 0:nDelay
        xkT = cat(2,xkT,x(k+iDelay+1,:));
    end
    H = cat(2,H,xkT.');
end
H
%%
%[text] Step 2 SVD of H
[U,Sgm,V] = svd(H,'econ')
%[text] Step 3 PCT
iMode = 1
ui = reshape(U(:,iMode),size(x,2),[]).'
%[text] Visualize
%myplot(T(1:size(ui,1)),ui)
%[text] 
V0 = V(1:end-1,:);
V1 = V(2:end,:);
dtAplusI = (V1.')*pinv(V0.')
%[text] Step 4 Discover forcinge signal
v0hat = V(1,:).';
vk = v0hat;
dtrk_ = 0;
dtr = [];
for k = 1:nRange-nDelay-1
    % True value
    vkp1 = V(k+1,:).';
    % Forcasted value
    vkp1hat = dtAplusI*vk;
    % Forcing signal
    dtrk_ = vkp1 - vkp1hat;
    % Update
    vk = vkp1;
    dtr = cat(2,dtr,dtrk_);
end
%%
%[text] Learnable parameters
%[text] ![fig02a.png](text:image:7035)
%[text] ![fig02b.png](text:image:9e97)
%[text] 
%%
%[text] Parameters
%[text] $\\hat{\\mathbf{v}}\_0$
v0hat
%[text] $\\{\\Delta t \\mathbf{r}\_k\\}$
dtr
%[text] $\\mathbf{I}+\\Delta t \\hat{\\mathbf{A}}$
dtAplusI
%[text] $\\hat{\\mathbf{U}}$
rRed = nCoefs/stride;
Uhat = U(1:(ndim*rRed),:);
Uhat
%[text] $\\mathbf{\\Sigma}$
Sgm
%[text] 
%%
%[text] Step 5 w/ Discovered forcing signal
vkhat = v0hat;
%dtrk = dtr(:,1);
Xhat = (Uhat*Sgm*vkhat).';
for k = 1:nRange-nDelay-1
    dtrk = dtr(:,k);
    vkp1hat = dtAplusI*vkhat + dtrk;
    Xhat =  cat(1,Xhat,(Uhat*Sgm*vkp1hat).');
    % Update
    vkhat = vkp1hat;
end
%%
% Reshape results
size(Xhat)  % (K-D) x (Ch.Pos.)
nRangeOut = nRange - nDelay;
dataYhat = ipermute((reshape(Xhat,nRangeOut,nCoefs,[])),[2 1 3]); % Ch. x (KxPos)
dataXhat = reshape(ipermute(pagemtimes(matS.',dataYhat),[2 1 3]),nRangeOut,stride,[]); % K x Ch. x Pos.
size(dataXhat)

%%
% Reconstruction of output to be sent throughr LSUN synthesizer
synlsunseq = dataXhat;
DataHatTaprx = zeros(nRangeOut,size(DataT,2),'like',DataT);
for iT = 1:nRangeOut
    ac = permute(synlsunseq(iT,2:end,:),[2 1 3]);   
    dc = synlsunseq(iT,1,:);
    DataHatTaprx(iT,:) = synlsunnet.predict(ac,dc);    
end
%%
figure
disp_imagesc_(T,X,DataHatTaprx)
% MSE評価
mse(DataT(1:nRangeOut,:),DataHatTaprx)
%%
%[text] Step 6 w/o Discovered forcing signal
vkhat = v0hat;
%dtrk = dtr(:,1);
Xhat = (Uhat*Sgm*vkhat).';
for k = 1:nRange-nDelay-1
    %dtrk = dtr(:,k);
    vkp1hat = dtAplusI*vkhat;
    Xhat =  cat(1,Xhat,(Uhat*Sgm*vkp1hat).');
    % Update
    vkhat = vkp1hat;
end
%%
% Reshape results
size(Xhat)  % (K-D) x (Ch.Pos.)
nRangeOut = nRange - nDelay;
dataYhat = ipermute((reshape(Xhat,nRangeOut,nCoefs,[])),[2 1 3]); % Ch. x (KxPos)
dataXhat = reshape(ipermute(pagemtimes(matS.',dataYhat),[2 1 3]),nRangeOut,stride,[]); % K x Ch. x Pos.
size(dataXhat)

%%
% Reconstruction of output to be sent throughr LSUN synthesizer
synlsunseq = dataXhat;
DataHatTaprx = zeros(nRangeOut,size(DataT,2),'like',DataT);
for iT = 1:nRangeOut
    ac = permute(synlsunseq(iT,2:end,:),[2 1 3]);   
    dc = synlsunseq(iT,1,:);
    DataHatTaprx(iT,:) = synlsunnet.predict(ac,dc);    
end
%%
figure
disp_imagesc_(T,X,DataHatTaprx)
% MSE評価
mse(DataT(1:nRangeOut,:),DataHatTaprx)
%%
%[text] **Function definition.**
%[text] Peter Mao (2023). fftfreq(https://www.mathworks.com/matlabcentral/fileexchange/67026-fftfreq), MATLAB Central file exchange. Retrieved, 6 January 2023.
function f=fftfreq(npts,dt,alias_dt)
% returns a vector of the frequencies corresponding to the length
% of the signal and the time step.
% specifying alias_dt > dt returns the frequencies that would
% result from subsampling the raw signal at alias_dt rather than
% dt.
 
 
 if (nargin < 3)
 alias_dt = dt;
 end
 fmin = -1/(2*dt);
 df = 1/(npts*dt);
 f0 = -fmin;
 alias_fmin = -1/(2*alias_dt);
 f0a = -alias_fmin;
 
 ff = mod(linspace(0, 2*f0-df, npts)+f0, 2*f0) - f0;
 fa = mod( ff+f0a, 2*f0a) - f0a;
 % return the aliased frequencies
 f = fa;
end

%[text] Definition of ODE system (PDE ---(FFT)---\> ODE system)
%Definition of ODE system (PDE ---(FFT)---> ODE system)
function u_t_real = burg_system(u, t, k,mu,nu)
 %Spatial derivative in the Fourier domain
 u_hat = fft(u);
 u_hat_x = 1j*k(:).*u_hat;
 u_hat_xx = -(k(:).^2).*u_hat;
 
 %Switching in the spatial domain
 u_x = ifft(u_hat_x);
 u_xx = ifft(u_hat_xx);
 
 %ODE resolution
 u_t = -mu*u.*u_x + nu*u_xx;
 u_t_real = real(u_t);
end
%[text] Function to extract a local patch block from a global array
function y = fcn_extract_blks_(x,iBlk,blksz,kx)
% Extend array x
padsz = [0 (kx-1)*blksz/2];
xx = padarray(x,padsz,"circular");
%
posx = (iBlk-1)*blksz+1;
y = xx(:,posx:posx+kx*blksz-1);
end
%[text] Function to place a local patch block to a global array
function y = fcn_place_blks_(y,blk,iBlk,blksz)
% Extend array x
posx = (iBlk-1)*blksz+1;
y(:,posx:posx+blksz-1) = blk;
end
%[text] Loss function
%[text] 
%[text] where is a unitaly analyzer with a coefficient mask. is guaranteed.
function [loss,gradients] = modelLoss(dlnet, dlX)
% Forward data through the dlnetwork object.
dlY = forward(dlnet,dlX); % F(x)
% Compute loss.
Nx = size(dlX,4);
Ny = size(dlY,4);
loss = sum(dlX.^2,"all")/Nx-sum(dlY.^2,"all")/Ny;
% Compute gradients.
gradients = dlgradient(loss,dlnet.Learnables);
loss = double(gather(extractdata(loss)));
end
%[text] 
function disp_plot3_(T,X,DataT,c)
if nargin < 4
 c = 'blue';
end
for idx = 1:length(X)
 plot3(X,T(idx)*ones(size(X)),DataT(idx,:),...
 'Color',c,...
 'LineWidth',1)
 hold on
end
xlabel('$x$', 'FontSize',14,'Interpreter','latex')
ylabel('$t$', 'FontSize',14,'Interpreter','latex')
zlabel('$u$', 'FontSize',14,'Interpreter','latex')
ax = gca;
ax.View = [30 30];
ax.PlotBoxAspectRatio = [ 1 1 1];
grid on
hold off
end
%[text] 
function disp_imagesc_(T,X,DataT)
imagesc(DataT)
colormap("jet")
colorbar
title('Burgers Equation','FontSize',15)
xlabel('Spatial','FontSize',12)
ylabel('Time','FontSize',12)
ax = gca;
ax.PlotBoxAspectRatio = [1 1 1];
ax.XTick = 1:20:100;
ax.XTickLabel = round(X(ax.XTick));
ax.YTick = 1:20:100;
ax.YTickLabel = round(T(ax.YTick));
end
%[text] 
function disp_plotmv_(ax,T,X,DataT,AprxT)
     a = DataT(1,:);
     b = AprxT(1,:);
     p = plot(ax,X,a,X,b);
     axis([min(X) max(X) min(DataT(:)) max(DataT(:))])
     xlabel('x')
     grid on
     hold on
     drawnow
     for iT = 2:length(T)
     a = DataT(iT,:);
     b = AprxT(iT,:);
     p(1).YData = a;
     p(2).YData = b;
     drawnow
     end
     hold off
end
%[text] 
function synthesislgraph = fcn_cpparamsana2syn_csax_(synthesislgraph,analysislgraph)
expanalyzer = '^Lv\d+_Cmp\d+_Q(\w\d|0)+(\w)+$';
nLayers = height(analysislgraph.Layers);
for iLayer = 1:nLayers
     alayer = analysislgraph.Layers(iLayer);
     alayerName = alayer.Name;
         if ~isempty(regexp(alayerName,expanalyzer,'once'))
         slayer = synthesislgraph.Layers({synthesislgraph.Layers.Name} == alayerName + "~");
         slayer.Angles = alayer.Angles;
         synthesislgraph = synthesislgraph.replaceLayer(slayer.Name,slayer);
         disp("Copy angles from " + alayerName + " to " + slayer.Name)
         end
end
end
%[text] 
function analysislgraph = fcn_cpparamssyn2ana_csax_(analysislgraph,synthesislgraph)
expanalyzer = '^Lv\d+_Cmp\d+_Q(\w\d|0)+(\w)+$';
nLayers = height(analysislgraph.Layers);
for iLayer = 1:nLayers
 alayer = analysislgraph.Layers(iLayer);
 alayerName = alayer.Name;
     if ~isempty(regexp(alayerName,expanalyzer,'once'))
     slayer = synthesislgraph.Layers({synthesislgraph.Layers.Name} == alayerName + "~");
     alayer.Angles = slayer.Angles;
     analysislgraph = analysislgraph.replaceLayer(alayerName,alayer);
     disp("Copy angles from " + slayer.Name + " to " + alayerName)
     end
end
end
%[text] 
function myplot(t,x)

subplot(1,2,1)
plot(t,[x(:,1)+15 x(:,2)-5 x(:,3)-45]);
axis([0 30 -80 80])
set(gca,'ytick',[-40 0 40],'yticklabel',{'x3','x2','x1'})
xlabel('t')
title('Burgers Eq')

subplot(1,2,2)
plot3(x(:,1),x(:,2),x(:,3));
xlabel('x1')
ylabel('x2')
zlabel('x3')

end

%[text] 
function fcn_replace_emptyangles_(lsunlgraph)
explayer = '^Lv\d+_Cmp\d+_V(\w\d|0)+(~|)$';
nLayers = height(lsunlgraph.Layers);
for iLayer = 1:nLayers
    layer_ = lsunlgraph.Layers(iLayer);
    if ~isempty(regexp(layer_.Name,explayer,'once'))
        if isa(layer_,"tansacnet.lsun.lsunIntermediateFullRotation1dLayer") && isempty(layer_.Angles)
            newlayer = tansacnet.lsun.lsunSign1dLayer( ...
                'Name',layer_.Name, ...
                'Stride',layer_.Stride, ...
                'Mode',layer_.Mode, ...
                'NumberOfBlocks',layer_.NumberOfBlocks, ...
                'Mus',layer_.Mus);
            lsunlgraph = lsunlgraph.replaceLayer(layer_.Name,newlayer);
            display("Replaced " + layer_.Name + " to " + class(newlayer))
        end
    end
end
end

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline","rightPanelPercent":40}
%---
%[text:image:7035]
%   data: {"align":"baseline","height":179,"src":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAtIAAACzCAIAAAArP2wrAAAOiUlEQVR42u3dCY7jNhBAUd\/\/UjmaAyTApNO2qOIqLu8jCAZu2ZaqSNYXLVGvNwAAwBBeQgAAAGgHAACgHQAAALQDAADQDgAAQDsAAABoBwAAoB0AAAC0AwCARNl7KXy0AwCAUc7BPGgHAAAjnOPz36AdAADQDtoBAMDizsE8aAcAAKAdAAAAtAMAgJsi90Hir46CdqBNYxUHAEbCq8FwlUFyk6PQHE\/ubEDlKRc244Smu+652SZHYSQ9qk4ICGgHDtSOxDlY86PuGvlhR0E70LIDCAtoB2hHYpJg2phkHcWcB6ICHVonRAYNzxchs4seZsI5pp1ICB7FtAdi+Dj69FR8oDiBdnxudvvKKtox4YEYPk4ZTf76B\/IBxQknZ\/bnka5VrXOPgnZgCu1gHlCcQDsSBbv+PpEBV9UE5WnOG14MH8dpB\/mA4gTakb6YdMJqnXUUtAPTaQf5gOKEMzO7aLXOTRntwIzawTygOIF20A7agXHaQT6gOOG0zK5YrbOOgnZgdu0gH1CcZFZm19KORQ9EI6MdzAOKk8zKbGoWYdHIuIEWC2gH+YDiJLOi8WsxrnW1Y8ID0choR4Z8iKfmpCXI7FEBWT0sEx6IRkY7THtAcZJZmQXtwATaQT6gOMksQDswVDv85gLFSWYB2oGh2mHaA5IuswDtwDjtIB+ak3TLLILxFAfaQTuYBxQnyKx40g6sph3kw2AKmYUJD9qBodpBPhQnyCyElHZgqHYwDyMpZBaJqAos7aAd5AOKE74XSHRCM6MdtIN8gHaAdtAO2oGVtYN50A7QDhgDaQftGI2ORzsgs6IqsLSDdpAPKE6QWSGlHdhRO5iHkRQye2xIxYF20A7yAcUJMiuetAO7awf5MJhCZk11QFxoB\/OA4iSzMgvagU21g3woTpBZ0A7QDvIBxUlmAdqBfbWDeShOkFnQDtAO8gHFSWYB2oFNtYN8KE6QWdAO0A7mAcVJZgHagU21g3woTpBZ0A7QDvIBxUlmAdqBfbWDeShOkFnQDtAO8qE4QWaBgdoxrI3qCbRjNvkY\/NVTdQHFiXY40nmO4iDtGH+0RjraMYl5PNUOJ+kCtEMxXv0wC97znjUsi6bsNSJti4\/4tIN8TNICZ9gB2kE7DnKOubVj0cq4mHYY72jHg\/IxSdVXnCCzQ6c6CsxjYCSXS9xrocMz3tGOB81jhuZHOyCzQ\/tOmXaMnSChHYJLO56XD9qhOEFma\/vOn+1pB+0A7UjLB+1QnCCzjbUj\/nbaQTuMJidox0\/zoB2KE2S2mXbkmgTtGKYdrztoB+3YTzvSTX2zLkA7aMcp2vFr46wJj4\/Nvo4Dx1bGLtrRqTUb72jHzLMdV9\/ecJdoB2T2Ge24ncP4efHpxYWoP4P85\/\/1YT9dOxKKRztoxyHa8Tnb8djQqThBZgv6ztctgxMe19t0mvDYXzvif007R1m4y043jSa0o0kkc8fugtnBybtA1ue\/Mtmgl5UdxQxBOEE7qqY6IhMeYe0Inq7vWhlHaEdDR7vdnnbQjge14515McdyXaBAO8peXLEKVu4w7ZhFOyK\/pOQ7R1aQ22rHbJXx1SVtMX8vPjNoprS0g3b0bPyNz8Am6AK52hEf\/mgH7Zh9qiMy4dFOOzaujK8ezTF4tlcciPRvN1SDdjw7FmfNdqzVBepP1+Kv0w7a8cBBBX9DKfoVZsAJyRKVsf1sx9WY22pMafurDe2gHV2n+no01we7QP24mXj74c8ipx1TzHZEtuysHWWJWKgyvponL3EbbeKVJg7Y9SYC2uHajqzikasdS3QB2kE7djaP+LIcX\/Xi88Winw4LevFalbHL5WxB7fj17ybBLfhM2kE72jp3cJRZsQvQDtpBOy5nNdppR24vXqsy9lql9PaNnwcfXBfhNrhZn0k7aEfDqpB7D\/lyXYB20I5ts1PwsLevK5lef2CWduTeir9QZey+SunX12uu1Du229COJZYLy5r\/W64L1H9FfJI5scx8k9hWrhry+fa0Ud1+1237iazNH7de2nFpEln\/XblL6WpVxfdwLJTZZqNe1lpAZR0+64kwhIN2jBm24g9kuSqZC3WBMdpxW4Y\/541\/BrbsBLEyEbdmED+0xPRPwf1BZjtC+S1zjqsJjzt9LPPIbSrjM0+gLbu8Jb4KCPOgHQ+cLR3QBSbRjuDrWUW6Pgs1y6PV2AntGN9\/u+5n7j4vVxnn0o6aH7bLPpN20I6ptGPyLjDPbEePYtzjspX4xjXaEZwsoR3rakfxtR0TVsYHtOPqZp6au5mLP5N20I7xw9a6XWAD7ci9uqLm7a02ph2HaEdZL16uMj6vHe8Wi6js8UyHQ7Sj384sqh0LdYFttKP4MXVlKYhf8XYbedpxjna8c65CXagyNl74fdjsExbVjn\/3pNP+DHj+6jwTsM82py1nOxKu0HUCIyJJuTtQvKL\/3oOh45pih1c5SM6xgXb83I0euzTmDtKTB5cTtKP5hFPlmW7ZDtCOQ+rIikf0emr00Rlox4raMb5NznXn29aXlL7Da9Fm7XDwc4KzGmm3qF+G5Jwh0VEsph2gHfWXdDTfK5JKO4o3fne4qPNd8cRd2oFthxEhoB3uZMFR2vFu8YNIPDJtlwt7\/39JtOLw0g7QDtAO2rGndsTv5rja\/n29GmzWxokvKg7F1y\/N+q74QreRLNAO0A48UBhoB1o1J8Vp6ZTRjhnGYdAO2lFb13995u3n3y5gQDuWmNsQ4S2dQ2ZpB+1As1LRqq5\/\/ZAyb+g3AWPoHNacRGba0wzaMaeyg3bs39x\/\/d7c3DmKHYJ2bNCcRGZC7ai\/1FdmmQftQG2p6K0dt3+lHVs2J2HZrLzJLPM4SDteGEJ9Lc9dk+PqjbevF++tLA\/D0HmCXEKXoR1YRjsS742\/SDuMoTAO6zW0Q3M\/VDsKPod2GEBBO6DXzKUdGDOI9Cjn02qHBtC7OYmJqxCQ1XFEg3acMoIMmEWgHQc2J2GhHeActANfho+263bkygft2LU5iQztgJDSDvzX1j+bfo9VSmnHsc1JZNRIiCftQHftSCtI0Dwir9COOYdOEVYmUTAOg3Zs3ty\/jia9n75GO45qTrMVy7I7CMoulf36ltXvZdB3aAftQOPRpKaWVy7s8UtNvv67iRgZOo8tTom9iuxw\/LiCKrNcKdJ3QDuwj3bU34hLOxSnypPL+LRHP+2YWT70HdAOTKQdkbtkPZOFdsysHbfzEEFjCBpM\/d7KLGgHjtaOl0fBgXbQDoB2YJh2tLrrlXbQjge14+tmv64MHaMdU4VR34kHauOvox2YRTvqnxPb75ExtENxis8f1GhHXE1q9kRm53eOYVEa\/HW0A7Nox0IYOmkH7aAdAyYeBgRq8NfRDtAO2kE7emnH55afS6KN0Y4JV0DRyGkH7QDtoB2042HtiHgJ7TjEOQaowOCvox2gHbSDdtAOmcWmw4gQ0A7aAdqR+LT0X2kHGTpzHKAdoB20g3YUbvaUdshsepcK3vPW\/aeXDxmiHbQDtIN2TDf7Uva23trRNkpZDw5s8nDBVxFtM2WAph20A7TjdbVl4k+0Y8apjgLzyNm+U5SyTKJyBwqez9w2IAZo2kE7cIR2RNbkGKMdrqLoEp8y7cjZvmaaYTntCDYM2gHaQTtoRzPtSMx10465tOPP9rSjj3nQDtAO2kE7Msa+yMNWgm\/x4PsFtCP+9vDG9ddVzKMdWeZBO0A7aIfilDeTEfwNu7d2zNw+99GO3AmPHO3oN+ExXjtyzYN2gHbQDtrxrr8IP7H91xtYrrav3BOZrapkvzbOmvDI+YWFdtAO0A7aYbYDtOOVN4fx8+LT2IWoV4q5tHa8My\/yoB2gHbSDduB47fi6ZXDCI\/MXlv20o948aAdoh6JIO7C8dlRNdcSVIqYdn5Fpbh4PakcT86AdoB2gHThAOyK\/pDSd6theO7prIu2gHbQDtAPzZLbNVEdELEqnOnqYx7Pa8a5\/ulvWSq\/6G+3YSTgURdqBFTNbsgPB31BqfoUZdY\/SDLMdld+Ysf\/6G+3YSTgURdqBU2Y7Ilu20I4BQXv8ktLKbzTbgVO0g3PQDuyX2VfQJ4Jq8lUvPl8sXRquSdyevYG28htd24EjtINwKE6gHRnmUaodrVyhVQqaX8Rarya0A5trB+FQnLC9drxurxXN+bjvkvFrPfWimYDbUSgrqvWPe80yiSZbWrcDm2sH51CcQDsulxlN\/3flLhXGkB6ICoapq7c0caDcaZX6B8XRDtpBOEA7sLh2lDnH1YRHck2O9GMCIze2VK7+WWwtwY8NQjtwlnYQDsUJtGP18O76dbQDu2kH51CcQDtoB+0A7SAcihNklnYc7Ry0A5toB+FQnEA7aAftAO3gHJqTjMjshrWTc4z8RsMH7SAcoB0yO11mNbNdpUpeaQfhAO2Q2Rkzq71t2Vqkk3ZwDigDMiuzGNXYhIB2EA4oTjIrs6Ad2FY7CIfiBJkF7QDt4BxQnGQWbeIpDrSDdhAOKE6QWdpBO3CAdhAOxQkyK54QFNphkgMGU5mVWSGlHVhfOwiHkRQye3hIRZV20A6\/qkBxgswyD9qBXbTDJIfihHULJDqhmdEO2kE4QDtAO2gH7cCa2qGn0Q7QDhgJaYfRpLt26Ga0AzILUaUd6K4dhMMwCpmFkNIOjNAOzmEkhcxCPGkHumsH4TCYQmaRjqc40A7aQTigOEFmaQftwCLawTk0J0mXWYB2oLt2EA4oTjIL0A501w7CAcVJZgHagRHawTmgOMksQDvQXTsIBxQnmQVoB7prB+GA4iSzAO3ACO3gHFCcZBagHeiuHYQDipPMCgVoB7prB+GA4iSzMgvagRHawTmgOEFmQTvQXTsIBxQnyCxoB7prB+GA4gSZBe3ACO3gHFCcILOgHeiuHYQDihNkFrQDI0YTwgHFCTIL2oFntENk0NZisQcaOWgHGtcJMQHtAO0A7UD3OiEaoB2gHXiWvwFb6sHStEeejQAAAABJRU5ErkJggg==","width":722}
%---
%[text:image:9e97]
%   data: {"align":"baseline","height":247,"src":"data:image\/png;base64,iVBORw0KGgoAAAANSUhEUgAAAw8AAAD3CAAAAAB+njfjAAAMVElEQVR42u3dAZakKAyA4dz\/UnU0dt90t4UYBARC1D\/v7W5vTZWVsfnEIKIE4q0h\/wfZJd9Ks3ixBr8eVmWHh1dzENfJ4YGwbXBCdnhAw29r8+lBBA\/EilMljx5+k\/t88EAYFw4OPWwc8EBYl9HuPHw14IGw5uDNQ6wBD4SxBm8e9hzwQNhq8OUh0YAHwpqDIw8HDXggjDU48qBwwANhq8GNB00DHghrDj486BrwQBhr8OEhxwEPhK0GDx6yGvBAWHNY7uFEAx4IYw3LPZxywANhq2Gxh3MNeCCsOaz0UNKAB8JYw0oPZQ54IGw1rPNQoQEPhDWHRR6qNOCBMNawyEMlBzwQthqWeKjVgAfCmoO9h3oNeCCMNdh7aOGAB8JWg7WHJg14IKw5mHpo1IAHwliDqYdmDnggbDUYemjXgAfCmoOVhysa8EAYa7DycI0DHghbDTYeLmrAA2HNwcDDZQ14IIw1GHjo4IAHwlbDdA89GvBAWHOY66FPAx4IYw1zPfRywANhq2Gmh24NeCCsOUzzMEADHghjDdM8DOGAh9c0ZicaVA\/iREOXB8HDnTiIEw6Kh1HJfT4LPfT8JfCwoHcQFxoUD73JDeRw3UPXX0JQcisPAzWM9zBQgysPEtav6zkx1hcP0rdvwiQPfcmFkRque+jcwxO6dDxM3TdDNzc8u89qD+MHO\/Dg2IPvzX0+z\/PQ2WuO8HCfJnXt3N\/JDQvJs9l\/k+qZPv5EDw7OCZ7pIf5y8ZB85nHUHdPH13voPA\/Aw5K\/mzj1IB0c3Hjo2VMyvE\/HQ8W3y7VsJtcP\/\/7\/+1rrt\/nw0NeIZHifjoeao9f3hfO329bTyTGw+DgVrx6+n202PbxPx0P56yXfP2cbp8n40v6Vs9lNenZ+PFw74KjjS\/s+fcWit3hY0z8kO78w28+lh\/Rv0XbAOfHwPW3Cw+Dvl9rj0xIPJ4fWe3jYfbJtB+vX4\/bHsAXPCHiXh\/wh2b5+OJxqnJxAOfWwDQhdOODgYUV5tD96Scgezow9\/F2OyyV3Gw\/f7BoPOJn5fPExTPAwL4Gf5ubGQzI2kyZ3Cw+y99B2wKn1YPyggLd4EFf1g0jSoO5YPyTDo40HnKyHfYEuwfZJAU\/2kBnhUOccGnrYT1\/6aUTRqVNldm6uTwdJTv56POx31KHXxMNIGskxeu340nlyN+gfErStB5yyB6F+mO9BPHu4Wz3dc8DJDR5JZjwED2Nz+R3j9Ng\/KMnd14N0ejjtNZ17kHt5CLlLwsq05fMGPMFDfgKHnl2pgRs\/b7f5gFPv4Tb1Q+Hmb+f3x13HfOGm9\/YHzpU2lzTpHhDDPLQccIr7UOs1XXso3ez6VA9XbvKd7EE+cmzkq++Pk64\/Dn13V+DhxR7k2EPc38OKk2rpXm9QbuvhagO+dNP73PUE8PC08aWrd+JOW8jjTutriF5CsJ7AbT1cXKNm4tI2t19vBg\/NyZT2w\/RngOhzFCrHMpo\/1rpvXHrYqkyXHsSlh4rmsc2f6fIgneOt0bS2n3HmupYdf0xC10JG2bQzN\/BINIltP1Xh3OZbPIg\/D1UHzG0aWdvplIz2EDfmqj4r2nD6sXEgcslL5ppZcq\/bXA+Seogc5j38\/bFM9iB+PRTberOH46Ip0utBkrus6kG0f6zll3rdgz59e3h2hyZ91j\/8\/s8fmcTPoz1UnlGX3yPatJmxHjaWodmDJB7C4MW4mzyUV48Z6kGqPGg2JILxBTFzXf+LHmRVFDyczUEpbqAxC7H62OVdI+dzjM49DI6SB72v0H76rFnF+oYewrmHgIccCCMP8Q\/7F3JYLDzIEz1oo47pLSvDPLiMG3jY2vr+P3iY5GG\/xOjxpOB8C2\/qHwojXKs9yNl8P4fnS+7qaa1KPXYWpTYwuZ4OXurpIojx2TV4+Gg1R+JhZLV\/Uw+hwkPIepDi5YJ5A6fexlvLIKaNtzZ52F+1eNv1h57rcdpN3Vot0eeh63qc+LkeFwbsi4vX46o8aJfsXuOhatZDab6GNnaodA\/n22ifr1HXri9+rLXFtXoIfeeODVibPSh3zE3yEBx6qG2J+f2gXmwVyRSIvR5uM5+v1D+cji\/N8fBLIP534bbqqR7CTT0Uju1aTxCycxUGXKu8x3zvQj0dDDxsm7vgYXd5Ql4037vXw8mz\/yQo47ADTiavtet5T\/a9MN4alnj4aeO\/\/2Sqh8J1bDy0eAiqhzDWw\/URolk3O4qUXs0vTyOWHjLjqXgY5kFfmitemj+ZzSkDPFw8zM967ntu1+jDC8W1m2bfH3d6prTrTD54aPaQvRCsX5UY4eHiaY\/5\/dNyeE7joduYev\/DNQ+Ch+secpfglTvDQmF2w8Pqh+hLK44fnjwUL9fhwdF6Alsz6hhfklXJL9pc6f64tH6Q82sTePDloe4WViV\/yU+veIkHH+sZ42FwG+h8aFPAAx7wgAc84AEPeMADHvCABzzgAQ94wAMe8IAHPOABD3jAAx7wgAc84AEPeMADHvCABzzgAQ94wAMe8ICHZ3lYtZo+HvCAh1t5GLnd8R4GisADHir3jWcPgoc1e9910rN+mWNBjE5SRorAAx6MRUxZUHOUCDzgwVjEnBVxBoHAAx6MQcxaQ3CICDzgwVjE1CIHD3iw8jBIxKwkh3QReMCDMYh5SQ4QgQc8GIuYX+TgAQ9WHvpFTE2yt4vAAx6MQUxOsk8EHvBgLMKoyMEDHuz+sh0i5ifZ0UXgAQ\/GICySvCwCD3gwFmFZ9OMBD95FGCV5TQQe8GAMwizJKyLwgAdjEeZFPx7w4FiEZZLNIvCAB2MQtkk2isADHoxFrCn68YAHnyLMk2wRgQc8GINYkGS9CDzgwVjEwqIfD3hwJ2JNkpUi8IAHYxCrkqwSgQc8GItYXfTjAQ\/Wjc7pb6QsAg94MO4iliZZEoEHPBiLcFH04wEPTk6aVid5KgIPeDDuItYneSICD3gwFuGm6Bc84GH9SZOLJHMi8IAH4y7CSZK6CDzgwViEr6Jf8ICHpSD8JKmIwAMejEV4SvIgAg94MBbhsOgXPOBhFQhnSe5F4AEPxiK8Fjl4wMMKEf6SjEDgAQ\/GIDwmuYnAAx6MRbgucvCAB\/s25zq5x3u4ZTzRQ\/DsYWV2eHinh5rbSd+YHR7e6sF5knjAw02Koydnd8MS9yZlOHv2lr84dgEeCDzggcADHgg84IHAAx4IPOCBwAMeCDzggcADHgg84IHAAx4IPOCBwAMeCDzggcADHgg84IG4j4fum8IGt547Nyk8PKJ\/2H6J1mt8as\/q0F7FA7HCw\/fHZR4CHgg\/HiIQKzzcvDnh4bEejNoPHgjnHgQPeMDDiiaJBwIPeCDu5EHiX+zPCunfn6Ox2e3pGttjBaJXvm+JtxSP5\/5bef3QnLY3JT9ILgfBAzHbw\/ba7ufDS7u37t+2\/Vu09+YeVJNcCfm+W8IxreTr8UDM9iBBV7EnoDTSuMXHn1AFHTNJt3+2FTwQCzzsmqjsNrT3EG9UJP3G3cfzHrTtH5i4Kzrw8Nh6Oq4ADm+K64CkfYruYfeDlDwk29ffL3ggjD2E+KmUBQ+yq4klbcnRy6M8uHv8Ex6e7uE7iFPtIRSP4YP7B8aXiGke1FP0kocQj8nigXi4h5DW0GmZ\/PWw\/6NjPR166ml9K4HzJWKSB\/0CwH6AKSgDpoce4zC+pL1cPd6qejGdiouHF3qQY2EQdROSvRKnkDoU4eqliurrcSH\/OtcfiIEe1NtFt5GgZMg1mXGREbafyqF+4m++hjZilLxbTqZoMF+DGN8\/EHjAA4EHAg94IPCABwIPeCDwgAcCD3gg8IAHAg94IPCABwIPeCDwgAcCD3ggnuXh1Q0CD3jAAx7w0NAiBA8EHt7YYeABD+l28UDgIfYgeCDw8LddPBB4+G42aRPRogD3frg0HvBwabPJKhrfxTb+PcVB8EC8w4P27OrkeUJ4IN7mIeQ8zKxc8EA48xCfJh2raVn2VF88EEs9hKKHgAfi4R7Uw3\/WA+dLxGs8BDwQL\/eQPAdltwK3Uj\/ggXiPh1Dj4XltBw940Lco2oMW9++W54HAAx6iprCPvy9Sr9IdHkmHB+JBHkR0EMlDTTQbeCAeWD8068EDgYdv+SB4IPAQe6B+IPCwjbYy3krgIXmWLx6It58vPXa34gEPBB7wQOABDwQe8EDgAQ8EHvBA4AEPBB7wYNdIXhQ0LjzgAQ94wAMeHhL\/ATmhz7NSY6tCAAAAAElFTkSuQmCC","width":783}
%---
