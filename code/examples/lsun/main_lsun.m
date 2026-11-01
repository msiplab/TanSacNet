%[text] # Uniform LSUN (Locallay-Structured Unitary Network)
%[text] LSUN, like PCA, is a fully linear transformation.
%[text] 
%[text] Please do not forget to run **setpath** in the top directory of this package, and then return to this directory.
%[text] 
%[text] Requirements: MATLAB R2025a or later
%[text] 【Reference】
%[text] 1. Yasas Godage, Eisuke Kobayashi and Shogo Muramatsu (2024), "Locally-Structured Unitary Network", APSIPA Transactions on Signal and Information Processing: Vol. 13: No. 1, e9. [http://dx.doi.org/10.1561/116.00000308](http://dx.doi.org/10.1561/116.00000308) \
%[text]  Contact address: Shogo MURAMATSU,
%[text]         Faculty of Engineering, Niigata University,
%[text]         8050 2-no-cho Ikarashi, Nishi-ku,
%[text]         Niigata, 950-2181, JAPAN
%[text]  [http://msiplab.eng.niigata-u.ac.jp](http://msiplab.eng.niigata-u.ac.jp) 
%[text] 
%[text]  Copyright (c) 2022-2024, Shogo MURAMATSU, All rights reserved.
%%
clc, clear
isVisible = ~true;
dstfolder = support.fcn_download_img
%%
imgfile = dstfolder + "kodim01.png";
img = im2double(rgb2gray(imread(imgfile)));
if canUseGPU
    disp("Use GPU")
    img = gpuArray(img);
end
%%
% rblk = imd';
% nblocks = 9;
% blk = im2col(rblk,[14 14] ,'distinct')
nof = 1; %[control:slider:1e54]{"position":[7,8]}
ky = 2*nof+1; % # of overlapping blocks (odd number)
kx = 2*nof+1;
disp([ky kx])
blksz = [4 4]; % Block size
% # of coeffs
nCoefs = 2; %[control:slider:8a95]{"position":[10,11]}
%%
%[text] ## Global block PCA for reference
% Reshape
colblks = im2col(img,blksz,"distinct");
% PCA
%Vpca = pca(colblks.')
mu = mean(colblks,2);
colblkszm = colblks - mu;
C = cov(colblkszm.');
[~,S,V] = svd(C,"econ");
[~,idxS] = sort(diag(S),"descend");
V = V(:,idxS(1:nCoefs));
%norm(Vpca - Vsvd,'fro')
% Approxiamtion
gbpcaimg = col2im(V*V.'*(colblks-mu)+mu,blksz,size(img),"distinct");

% Display
imshow(gbpcaimg)
title("Approx. by global block PCA (PSNR: "+num2str(psnr(img,gbpcaimg),"%6.2f")+" dB)")

%%
%[text] ## Local block PCA for reference
% % Create Sub-images
% s_img = createsubimg(blk);
[szy,szx] = size(img);
lbpcaimg = zeros(szy,szx,'like',img);
for iBlkCol = 1:szx/blksz(2)
    for iBlkRow = 1:szy/blksz(1)
        % Extract ky x kx blocks
        subblks = fcn_extract_blks_(img,[iBlkRow,iBlkCol],blksz,[ky,kx]);
        % Reshape
        colblks = im2col(subblks,blksz,"distinct");
        % PCA
        %Vpca = pca(colblks.')
        mu = mean(colblks,2);
        colblkszm = colblks - mu;
        C = cov(colblkszm.');
        [~,S,V] = svd(C,"econ");
        [~,idxS] = sort(diag(S),"descend");
        V = V(:,idxS(1:nCoefs));
        %norm(Vpca - Vsvd,'fro')
        % Approxiamtion
        targetblk = fcn_extract_blks_(img,[iBlkRow,iBlkCol],blksz,[1 1]);
        targetblk = reshape(V*V.'*(targetblk(:)-mu)+mu,size(targetblk));
        % Place block
        lbpcaimg = fcn_place_blks_(lbpcaimg,targetblk,[iBlkRow,iBlkCol],blksz);
    end
end

% Display
imshow(lbpcaimg)
title("Approx. by local block PCA (PSNR: "+num2str(psnr(img,lbpcaimg),"%6.2f")+" dB)")
%%
%[text] ## Locally Structured Unitary Network (LSUN) for 2-D Grayscale image
% Decimation factor (Strides)
stride = blksz; % [My Mx]

% Number of overlapping blocks (Polyphase order plus one)
ovlpFactor = [ky kx];

% Max epochs
maxEpochs = 64;

% Standard deviation of initial angles
stdInitAng = 0;

% No DC-leakage
noDcLeakage = true;
%%
%[text] ## Bivariate lattice-structure of filter banks 
%[text] 
%[text] As a base system for LSUN, let us adopt a multidimensional linear-phase paraunitary filter banks (MD-LPPUFB) ( or non-separable oversampled lapped transform (NSOLT)) of  type-I with the number of channels (the numbers of even and odd symmetric channels are identical to each other) and polyphase order (even):
%[text]  $\\mathbf{E}(z\_\\mathrm{v},z\_\\mathbf{h})\n=\n\\left(\\prod\_{k\_\\mathrm{h}=1}^{N\_\\mathrm{h}/2}\n{\\mathbf{V}\_{2k\_\\mathrm{h}}^{\\{\\mathrm{h}\\}}}\\bar{\\mathbf{Q}}(z\_\\mathrm{h}){\\mathbf{V}\_{2k\_\\mathrm{h}-1}^{\\{\\mathrm{h}\\}}}{\\mathbf{Q}}(z\_\\mathrm{h})\\right)\n%\n\\left(\\prod\_{k\_{\\mathrm{v}}=1}^{N\_\\mathrm{v}/2}{\\mathbf{V}\_{2k\_\\mathrm{v}}^{\\{\\mathrm{v}\\}}}\\bar{\\mathbf{Q}}(z\_\\mathrm{v}){\\mathbf{V}\_{2k\_\\mathrm{v}-1}^{\\{\\mathrm{v}\\}}}{\\mathbf{Q}}(z\_\\mathrm{v})\\right)\n%\n\\mathbf{V}\_0\\mathbf{E}\_0,$
%[text]  $\\mathbf{R}(z\_\\mathrm{v},z\_\\mathbf{h})\n=\\mathbf{E}^T(z\_\\mathrm{v}^{-1},z\_\\mathrm{h}^{-1}),$
%[text] where
%[text] - $\\mathbf{E}(z\_\\mathrm{v},z\_\\mathrm{h})$:  Type-I polyphase matrix of the analysis filter bank
%[text] - $\\mathbf{R}(z\_\\mathrm{v},z\_\\mathrm{h})$: Type-II polyphase matrix in the synthesis filter bank
%[text] - $z\_d\\in\\mathbb{C}, d\\in\\{\\mathrm{v},\\mathrm{h}\\}$: The parameter of Z-transformation direction
%[text] - $N\_d\\in \\mathbb{N}, d\\in\\{\\mathrm{v},\\mathrm{h}\\}$: Polyphase order in direction $d$ (number of overlapping blocks)
%[text] - $\\mathbf{V}\_0=\\left(\\begin{array}{cc}\\mathbf{W}\_{0} & \\mathbf{O} \\\\\\mathbf{O} & \\mathbf{U}\_0\\end{array}\\right)\n%\n\\left(\\begin{array}{c}\\mathbf{I}\_{M/2} \\\\ \n\\mathbf{O} \\\\\n\\mathbf{I}\_{M/2} \\\\\n\\mathbf{O}\n\\end{array}\\right)\\in\\mathbb{R}^{P\\times M}$,$\\mathbf{V}\_n^{\\{d\\}}=\\left(\\begin{array}{cc}\\mathbf{I}\_{P/2} & \\mathbf{O} \\\\\\mathbf{O} & \\mathbf{U}\_n^{\\{d\\}}\\end{array}\\right)\\in\\mathbb{R}^{P\\times P}, d\\in\\{\\mathrm{v},\\mathrm{h}\\}$, where$\\mathbf{W}\_0, \\mathbf{U}\_0,\\mathbf{U}\_n^{\\{d\\}}\\in\\mathbb{R}^{P/2\\times P/2}$are orthonromal matrices.
%[text] - $\\mathbf{Q}(z)=\\mathbf{B}\_{P}\\left(\\begin{array}{cc} \\mathbf{I}\_{P/2} &  \\mathbf{O} \\\\ \\mathbf{O} &  z^{-1}\\mathbf{I}\_{P/2}\\end{array}\\right)\\mathbf{B}\_{P}$, $\\bar{\\mathbf{Q}}(z)=\\mathbf{B}\_{P}\\left(\\begin{array}{cc} z\\mathbf{I}\_{P/2} &  \\mathbf{O} \\\\ \\mathbf{O} &  \\mathbf{I}\_{P/2}\\end{array}\\right)\\mathbf{B}\_{P}$, $\\mathbf{B}\_{P}=\\frac{1}{\\sqrt{2}}\\left(\\begin{array}{cc} \\mathbf{I}\_{P/2} &  \\mathbf{I}\_{P/2} \\\\ \\mathbf{I}\_{P/2} &  -\\mathbf{I}\_{P/2}\\end{array}\\right)$ \
%[text] 【Example】For $P/2=3$, a parametric orthonormal matrix $\\mathbf{U}(\\mathbf{\\theta},\\mathbf{\\mu})$ can be constructed by 
%[text]  $\\mathbf{U}(\\mathbf{\\theta},\\mathbf{\\mu}) \\colon = \\left(\\begin{array}{cc} \\mu\_1 & 0& 0\\\\ 0 & \\mu\_1 & 0 \\\\ 0 & 0 & \\mu\_2 \\end{array}\\right)\n%\n\\left(\\begin{array}{ccc} \n 1 & 0 & 0 \\\\\n0 & \\cos\\theta\_2& -\\sin\\theta\_2 \\\\ \n0 & \\sin\\theta\_2 & \\cos\\theta\_2 \n\\end{array}\\right)\n%\n\\left(\\begin{array}{ccc} \n\\cos\\theta\_1& 0 & -\\sin\\theta\_1  \\\\ \n 0 & 1 & 0 \\\\\n\\sin\\theta\_1 & 0 &  \\cos\\theta\_1 \n \\end{array}\\right)\n%\n\\left(\\begin{array}{ccc} \n\\cos\\theta\_0& -\\sin\\theta\_0 & 0 \\\\ \n\\sin\\theta\_0 & \\cos\\theta\_0 & 0 \\\\\n 0 & 0 & 1 \\end{array}\\right),$
%[text]  ${\\mathbf{U}(\\mathbf{\\theta},\\mathbf{\\mu})}^T = \n%\n\\left(\\begin{array}{ccc} \n\\cos\\theta\_0& \\sin\\theta\_0 & 0 \\\\ \n-\\sin\\theta\_0 & \\cos\\theta\_0 & 0 \\\\\n 0 & 0 & 1 \\end{array}\\right)\n%\n\\left(\\begin{array}{ccc} \n\\cos\\theta\_1& 0 & \\sin\\theta\_1  \\\\ \n 0 & 1 & 0 \\\\\n-\\sin\\theta\_1 & 0 &  \\cos\\theta\_1 \n \\end{array}\\right)\n%\n\\left(\\begin{array}{ccc} \n1 & 0 & 0 \\\\\n0 & \\cos\\theta\_2& \\sin\\theta\_2 \\\\ \n0 & -\\sin\\theta\_2 & \\cos\\theta\_2 \n\\end{array}\\right)\n%\n\\left(\\begin{array}{cc} \\mu\_0 & 0& 0\\\\ 0 & \\mu\_1 & 0 \\\\ 0 & 0 & \\mu\_2 \\end{array}\\right),$
%[text] where $\\mathbf{\\theta}\\in\\mathbb{R}^{(P-2)P/8}$ and $\\mathbf{\\mu}=\\{-1,1\\}^{P/2}$. For the sake of simplification, the sign parameters $\\mu\_k$ are fixed to $-1$for $\\mathbf{U}\_n^{\\{d\\}}$ witn odd $n$, otherwise they are fixed to $+1$.
%[text] Partial differentiation can be, for examle, conducted as
%[text]  $\\frac{\\partial}{\\partial \\theta\_1}{\\mathbf{U}(\\mathbf{\\theta},\\mathbf{\\mu})}^T = \n%\n\\left(\\begin{array}{ccc} \n\\cos\\theta\_0& \\sin\\theta\_0 & 0 \\\\ \n-\\sin\\theta\_0 & \\cos\\theta\_0 & 0 \\\\\n 0 & 0 & 1 \\end{array}\\right)\n%\n\\left(\\begin{array}{ccc} \n-\\sin\\theta\_1& 0 & \\cos\\theta\_1  \\\\ \n 0 & 0 & 0 \\\\\n-\\cos\\theta\_1 & 0 &  -\\sin\\theta\_1 \n \\end{array}\\right)\n%\n\\left(\\begin{array}{ccc} \n1 & 0 & 0 \\\\\n0 & \\cos\\theta\_2& \\sin\\theta\_2 \\\\ \n0 & -\\sin\\theta\_2 & \\cos\\theta\_2 \n\\end{array}\\right)\n%\n\\left(\\begin{array}{cc} \\mu\_0 & 0& 0\\\\ 0 & \\mu\_1 & 0 \\\\ 0 & 0 & \\mu\_2 \\end{array}\\right).$
%[text] 
%[text] A locally-structured unitary network (LSUN) allows to change the parameters block by block.
%[text] 【References】 
%[text] - MATLAB SaivDr Package: [https://github.com/msiplab/SaivDr](https://github.com/msiplab/SaivDr)
%[text] - S. Muramatsu, K. Furuya and N. Yuki, "Multidimensional Nonseparable Oversampled Lapped Transforms: Theory and Design," in IEEE Transactions on Signal Processing, vol. 65, no. 5, pp. 1251-1264, 1 March1, 2017, [doi: 10.1109/TSP.2016.2633240](https://ieeexplore.ieee.org/document/7762239).
%[text] - S. Muramatsu, T. Kobayashi, M. Hiki and H. Kikuchi, "Boundary Operation of 2-D Nonseparable Linear-Phase Paraunitary Filter Banks," in IEEE Transactions on Image Processing, vol. 21, no. 4, pp. 2314-2318, April 2012, doi: 10.1109/TIP.2011.2181527.
%[text] - S. Muramatsu, M. Ishii and Z. Chen, "Efficient parameter optimization for example-based design of nonseparable oversampled lapped transform," 2016 IEEE International Conference on Image Processing (ICIP), Phoenix, AZ, 2016, pp. 3618-3622, doi: 10.1109/ICIP.2016.7533034.
%[text] - Furuya, K., Hara, S., Seino, K., & Muramatsu, S. (2016). Boundary operation of 2D non-separable oversampled lapped transforms. *APSIPA Transactions on Signal and Information Processing, 5*, E9. doi:10.1017/ATSIP.2016.3.
%[text] - S. Muramatsu, A. Yamada and H. Kiya, "A design method of multidimensional linear-phase paraunitary filter banks with a lattice structure," in IEEE Transactions on Signal Processing, vol. 47, no. 3, pp. 690-700, March 1999,[doi: 10.1109/78.747776](https://ieeexplore.ieee.org/document/747776). \
%%
%[text] ### Definition of custom layers and networks 
%[text] Use a custom layer of Deep Learning Toolbox to implement Analysis LSUN.
%[text] #### Definition of layers w/ Learnable properties
%[text] - Initial rotation: $\\mathbf{V}\_{0,b}$ (tansacnet.lsun.lsunInitialRotationLayer)
%[text] - Intermediate rotation: ${\\mathbf{V}\_{n,b}^{\\{d\\}}}$ (tansacnet.lsun.lsunIntermediateRotationLayer) \
%[text] #### Definition of layers w/o Learnable properties
%[text] - Bivariate DCT (2-D IDCT): $\\mathbf{E}\_0$ (tansacnet.lsun.lsunBlockDctLayer)
%[text] - Vertical up extension: $\\mathbf{Q}(z\_\\mathrm{v})$ (tansacnet.lsun.lsunAtomExtensionLayer)
%[text] - Vertical down extension: $\\bar{\\mathbf{Q}}(z\_\\mathrm{v})$  (tansacnet.lsun.lsunAtomExtensionLayer)
%[text] - Horizontal left extension: $\\mathbf{Q}(z\_\\mathrm{h})$ (tansacnet.lsun.lsunAtomExtensionLayer)
%[text] - Horizontal right extension: $\\bar{\\mathbf{Q}}(z\_\\mathrm{h})$ (tansacnet.lsun.lsunAtomExtensionLayer) \
import tansacnet.lsun.*
analysislgraph = fcn_createlsunlgraph2d([],...
    'InputSize',[szy szx],...
    'Stride',stride,...
    'OverlappingFactor',ovlpFactor,...
    'NumberOfVanishingMoments',noDcLeakage,...
    'Mode','Analyzer');
synthesislgraph = fcn_createlsunlgraph2d([],...
    'InputSize',[szy szx],...
    'Stride',stride,...
    'OverlappingFactor',ovlpFactor,...
    'NumberOfVanishingMoments',noDcLeakage,...
    'Mode','Synthesizer');
figure
subplot(1,2,1)
plot(analysislgraph)
title('Analysis LSUN')
subplot(1,2,2)
plot(synthesislgraph)
title('Synthesis LSUN')
%%

% Construction of synthesis network.
analysisnet = dlnetwork(analysislgraph);

% Initialize
nLearnables = height(analysisnet.Learnables);
for iLearnable = 1:nLearnables
    if analysisnet.Learnables.Parameter(iLearnable)=="Angles"
        analysisnet.Learnables.Value(iLearnable) = ...
            cellfun(@(x) x+stdInitAng*randn(size(x)), ...
            analysisnet.Learnables.Value(iLearnable),'UniformOutput',false);
    end
end
%%
import tansacnet.lsun.*
% Construction of analysis network
analysislgraph = layerGraph(analysisnet);
synthesislgraph = fcn_cpparamsana2syn(synthesislgraph,analysislgraph);
synthesisnet = dlnetwork(synthesislgraph);

%%
%[text] ### Confirmation of the adjoint relation (perfect reconstruction)
x = rand([szy szx],'single');
dlx = dlarray(x,'SSCB'); % Deep learning array (SSCB: Spatial,Spatial,Channel,Batch)
[dls{1:2}] = analysisnet.predict(dlx);
dly = synthesisnet.predict(dls{:});
display("MSE: " + num2str(mse(dlx,dly)))
%%
%[text] ## Parameter optimization and approximation
import tansacnet.lsun.*
analysislgraph = layerGraph(analysisnet);

% Coefficient masking
nChsTotal = prod(stride);
coefMask = reshape([ones(nCoefs,1); zeros(nChsTotal-nCoefs,1)],2,[]).'; % Revised on Jan. 16, 2023
coefMask = coefMask(:);
%nLevels = 1;
%for iLv = nLevels:-1:1
iLv = 1;
strLv = sprintf('Lv%0d_',iLv);
% For AC
analysislgraph = analysislgraph.replaceLayer([strLv 'AcOut'],...
    maskLayer('Name',[strLv 'AcMask'],'Mask',coefMask(2:end),...
    'NumberOfChannels',nChsTotal-1));
%strLvPre = strLv;
%end

% Output layer
iCmp = 1;
strCmp = sprintf('Cmp%0d_',iCmp);
%analysislgraph = analysislgraph.addLayers([...
%      lsunChannelConcatenation2dLayer('Name',[strLv strCmp 'Cn']) ...
%      lsunRegressionLayer('Coefficient output')
%     ]);
analysislgraph = analysislgraph.addLayers(...
      lsunChannelConcatenation2dLayer('Name',[strLv strCmp 'Cn']));
analysislgraph = analysislgraph.connectLayers(...
    [strLv 'AcMask' ], [strLv strCmp 'Cn/ac']);
analysislgraph = analysislgraph.connectLayers(...
    [strLv 'DcOut' ], [strLv strCmp 'Cn/dc']);

figure
plot(analysislgraph)
title('Analysis LSUN')
%%
% Image data store
fs = matlab.io.datastore.FileSet(imgfile);
imds = imageDatastore(imgfile,"ReadFcn",@(x) im2single(rgb2gray(imread(imgfile))));
%patchds = randomPatchExtractionDatastore(imds,imds,[szy szx],'PatchesPerImage',1); 
figure
%minibatch = preview(patchds);
%inputimg = minibatch.InputImage;
imshow(preview(imds));
drawnow
%figure
%responses = minibatch.ResponseImage;
%montage(responses,'Size',[2 4]);
%%

dlX = dlarray(gpuArray(readimage(imds,1)),"SSCB");
trainnet = dlnetwork(analysislgraph,dlX);
assert(trainnet.Initialized)
%miniBatchSize = 1;
%mbq = minibatchqueue(patchds,...
%    'MiniBatchSize',miniBatchSize,...
%    'MiniBatchFormat',{'SSBC','SSBC'});
figure
monitor = trainingProgressMonitor(Metrics="Loss",Info="Epoch",XLabel="Iteration");
% lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
% ylim([0 inf])
% xlabel("Iteration")
% ylabel("Loss")
% grid on

%%
numIterations = maxEpochs;

% Training
velocity = [];
iteration = 0;
momentum = 0.9;
decay = 0.01;
initialLearnRate = 1e-1;
epoch = 0;
start = tic;

% Loop over epochs.
while epoch < maxEpochs && ~monitor.Stop
    epoch = epoch + 1;
    % Shuffle data.
    %shuffle(mbq);
    shuffle(imds);

    % Loop over mini-batches.
    while hasdata(imds) && ~monitor.Stop % hasdata(mbq)
        iteration = iteration + 1;

        % Read mini-batch of data.
        %[dlX, T] = next(mbq);
        dlX = dlarray(gpuArray(read(imds)),"SSCB");

        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelGradients function and update the network state.
        [gradients,loss] = dlfeval(@modelGradients,trainnet,dlX);

        % Determine learning rate for time-based decay learning rate schedule.
        learnRate = initialLearnRate/(1 + decay*iteration);
        
        % Update the network parameters using the SGDM optimizer.
        [trainnet,velocity] = sgdmupdate(trainnet,gradients,velocity,learnRate,momentum);
        
        % Display the training progress.
        %D = duration(0,0,toc(start),'Format','hh:mm:ss');
        %addpoints(lineLossTrain,iteration,loss)
        %title("Epoch: " + epoch + ", Elapsed: " + string(D))
        %drawnow
        recordMetrics(monitor,iteration,Loss=loss);
        updateInfo(monitor,Epoch=epoch + " of " + maxEpochs);
        monitor.Progress = 100 * iteration/numIterations;
    end
    
    reset(imds);
end
%%
%[text] ## Approximation by LSUN-base linear autoencoder 
import tansacnet.lsun.*
lsunlgraph = fcn_createlsunlgraph2d([],...
    'InputSize',[szy szx],...
    'Stride',stride,...
    'OverlappingFactor',ovlpFactor,...
    'NumberOfVanishingMoments',noDcLeakage,...
    'Mode','Whole');
trainlgraph = layerGraph(trainnet);
lsunlgraph = fcn_cpparamsana2syn(lsunlgraph,trainlgraph);
lsunlgraph = fcn_cpparamssyn2ana(lsunlgraph,lsunlgraph);

%nLevels = 1;
%for iLv = nLevels:-1:1
iLv = 1;
strLv = sprintf('Lv%0d_',iLv);
lsunlgraph = lsunlgraph.disconnectLayers([strLv 'AcOut'],[strLv 'AcIn']);
% For AC
lsunlgraph = lsunlgraph.addLayers(...
    maskLayer('Name',[strLv 'AcMask'],'Mask',coefMask(2:end),...
    'NumberOfChannels',nChsTotal-1));
lsunlgraph = lsunlgraph.connectLayers([strLv 'AcOut'],[strLv 'AcMask']);
lsunlgraph = lsunlgraph.connectLayers([strLv 'AcMask'],[strLv 'AcIn']);
%strLvPre = strLv;
%end

figure
plot(lsunlgraph)
title('Linear autoencoder with LSUN')

%[text] Predict
lsunnet = assembleNetwork(lsunlgraph);
lsunaimg = lsunnet.predict(img);
%%
%[text] ## Results 
figure
imshow(img)
title("Original")
figure
imshow(lbpcaimg)
title("Approx. by Local block PCA (PSNR: " + num2str(psnr(img,lbpcaimg))+" dB)")
figure
imshow(gbpcaimg)
title("Approx. by Global block PCA (PSNR: " + num2str(psnr(img,gbpcaimg))+" dB)")
figure
imshow(lsunaimg)
title("Approx. by LSUN (PSNR: " + num2str(psnr(img,im2double(lsunaimg)))+" dB)")

%%
%[text] ## Definitions of local functions
%[text] Function to extract a local patch block from a global array
function y = fcn_extract_blks_(x,iBlk,blksz,k)
% Extend array x
ky = k(1);
kx = k(2);
iBlkRow = iBlk(1);
iBlkCol = iBlk(2);
padsz = [(ky-1)/2, (kx-1)/2].*blksz;
xx = padarray(x,padsz,"circular");
%
posy = (iBlkRow-1)*blksz(1)+1;
posx = (iBlkCol-1)*blksz(2)+1;
y = xx(posy:posy+ky*blksz(1)-1,posx:posx+kx*blksz(2)-1);
end
%[text] Function to place a local patch block to a global array
function y = fcn_place_blks_(y,blk,iBlk,blksz)
% Place array blk
iBlkRow = iBlk(1);
iBlkCol = iBlk(2);
posy = (iBlkRow-1)*blksz(1)+1;
posx = (iBlkCol-1)*blksz(2)+1;
y(posy:posy+blksz(1)-1,posx:posx+blksz(2)-1) = blk;
end
%[text] Loss function
%[text]  $L(\\mathbf{\\theta}) = \\|\\mathbf{x}\\|\_2^2-\\|F\_\\mathbf{\\theta}(\\mathbf{x})\\|\_2^2$,
%[text] where $F\_\\mathbf{\\theta}(\\cdot)$ is a unitary analyzer with a coefficient mask. $L(\\mathbf{\\theta})\\geq 0$ is guaranteed.
function [gradients, loss] = modelGradients(dlnet, dlX)
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

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":40}
%---
%[control:slider:1e54]
%   data: {"defaultValue":1,"label":"ord","max":5,"min":0,"run":"Nothing","runOn":"ValueChanged","step":1}
%---
%[control:slider:8a95]
%   data: {"defaultValue":1,"label":"ord","max":64,"min":1,"run":"Nothing","runOn":"ValueChanged","step":1}
%---
