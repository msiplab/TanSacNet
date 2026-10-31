%[text] # Quickstart of **TanSacNet Package**
%[text] A brief introduction to the locally structured unitary network (LSUN) with **TanSacNet Package**
%[text] ## Summary
%[text] **TanSacNet** is an abbreviation of *Tangent Space Adaptive Control Networks*. This package is developed for
%[text] - Experiments,
%[text] - Development and
%[text] - Implementation \
%[text] of sampling embedded manifold of image and volumetric data as a set of tangent space bases.
%[text] A locally structured unitary network (LSUN) is a lattice-structured unitary (Parseval tight) transform whose parameters can be changed block by block. We have prepared custom layer classes with Deep Learning Toolbox. It is easy to incorporate them into flexible configurations and parts of your network.
%[text] The Python (PyTorch) counterpart of this live script is quickstart.ipynb in the same directory.
%%
%[text] ## Add the package to search path
%[text] Before using **TanSacNet Package**, it is required to add the package to the MATLAB's search path. In this directory 'code', execute the following command:
setpath
%%
%[text] ## Set the parameters
%[text] An LSUN of the following configuration is created:
%[text] - Input size: $32\\times 32$
%[text] - Stride (decimation factor): $4\\times 4$, which gives $16$ channels
%[text] - Overlapping factor: $3\\times 3$
%[text] - Number of tree levels: $1$
%[text] - No DC leakage, i.e. one vanishing moment \
szx = 32;            % Width
szy = 32;            % Height
stride = [4 4];      % Stride [vertical horizontal]
ovlpFactor = [3 3];  % Overlapping factor [vertical horizontal]
nLevels = 1;         % Number of tree levels
noDcLeakage = true;  % No DC leakage
stdInitAng = pi/6;   % Standard deviation of the initial angles
nChs = prod(stride); % Number of channels
%%
%[text] ## Create the analysis and synthesis networks
%[text] The function tansacnet.lsun.fcn\_createlsunlgraph2d creates the layer graphs of the analysis and synthesis LSUNs.
import tansacnet.lsun.*
analysislgraph = fcn_createlsunlgraph2d([],...
    'InputSize',[szy szx],...
    'Stride',stride,...
    'OverlappingFactor',ovlpFactor,...
    'NumberOfLevels',nLevels,...
    'NumberOfVanishingMoments',noDcLeakage,...
    'Mode','Analyzer');
synthesislgraph = fcn_createlsunlgraph2d([],...
    'InputSize',[szy szx],...
    'Stride',stride,...
    'OverlappingFactor',ovlpFactor,...
    'NumberOfLevels',nLevels,...
    'NumberOfVanishingMoments',noDcLeakage,...
    'Mode','Synthesizer');
%[text] The angles of the rotation layers are the learnable parameters. They are set to random values here to obtain a non-trivial transform.
rng(0)
analysisnet = dlnetwork(analysislgraph);
nLearnables = height(analysisnet.Learnables);
for iLearnable = 1:nLearnables
    if analysisnet.Learnables.Parameter(iLearnable)=="Angles"
        analysisnet.Learnables.Value(iLearnable) = ...
            cellfun(@(x) x+stdInitAng*randn(size(x)), ...
            analysisnet.Learnables.Value(iLearnable),'UniformOutput',false);
    end
end
%[text] The synthesis LSUN is the adjoint of the analysis LSUN. The function tansacnet.lsun.fcn\_cpparamsana2syn copies the parameters from the analysis LSUN to the synthesis LSUN.
synthesislgraph = fcn_cpparamsana2syn(synthesislgraph,layerGraph(analysisnet));
synthesisnet = dlnetwork(synthesislgraph);
%%
%[text] ## Confirm the perfect reconstruction
%[text] Since LSUN is unitary, the synthesis LSUN reconstructs the input of the analysis LSUN from its output, i.e.
%[text]{"align":"center"} $\\mathbf{x} = \\mathbf{D}\\mathbf{D}^T\\mathbf{x}$,
%[text] where $\\mathbf{D}^T$ and $\\mathbf{D}$ denote the analysis and synthesis LSUNs, respectively.
%[text] The analysis LSUN of one level outputs the AC and DC coefficients separately. The images are given as deep learning arrays of the format 'SSCB' (spatial, spatial, channel, batch).
x = dlarray(rand(szy,szx,'single'),'SSCB');
[s{1:2}] = analysisnet.predict(x); % {AC, DC} coefficients
sizeOfAcCoefs = size(s{1})
sizeOfDcCoefs = size(s{2})
y = synthesisnet.predict(s{:});
display("MSE: " + num2str(mse(x,y)))
%%
%[text] ## Show the atomic images
%[text] The atomic images (basis images) of LSUN are the columns of $\\mathbf{D}$. They are obtained by synthesizing a unit impulse placed at the center of each channel of the coefficients.
cy = szy/stride(1)/2+1; % Center of the coefficients
cx = szx/stride(2)/2+1;
atoms = zeros(szy,szx,nChs);
for iCh = 1:nChs
    coefs = cellfun(@(c) zeros(size(c),'like',c),s,'UniformOutput',false);
    if iCh == 1 % DC channel
        coefs{2}(cy,cx,1) = 1;
    else % AC channels
        coefs{1}(cy,cx,iCh-1) = 1;
    end
    atoms(:,:,iCh) = gather(extractdata(synthesisnet.predict(coefs{:})));
end
% Crop the support of the atoms
rows = find(any(abs(atoms)>1e-6,[2 3]));
cols = find(any(abs(atoms)>1e-6,[1 3]));
atoms = atoms(rows(1):rows(end),cols(1):cols(end),:);
vmax = max(abs(atoms(:))); % Zero is shown in gray
tl = tiledlayout(stride(1),stride(2),'TileSpacing','compact');
for iCh = 1:nChs
    nexttile
    imshow(atoms(:,:,iCh),[-vmax vmax],'InitialMagnification','fit')
end
title(tl,'Atomic images of LSUN')
%%
%[text] ## Next steps
%[text] Several example codes are found under the directory 'examples'. For the training of LSUN as an image approximation like PCA, see examples/lsun/main\_lsun.m.

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"inline"}
%---
