function lsunLgraph = ...
    fcn_createlsunlgraph1d(lsunLgraph,varargin)
%FCN_CREATELSUNLGRAPH1D
%
%   1-D counterpart of tansacnet.lsun.fcn_createlsunlgraph2d, which has
%   the same structure as the 2-D and 3-D LSUNs (not the CS type built by
%   tansacnet.lsun.fcn_createcslsunlgraph1d).
%
%   InputSize (sequence length), Stride and OverlappingFactor are scalars.
%   The sequences are arranged as 1 x InputSize x 1 x nSamples (SSCB).
%
%   NOTE: Stride 2 gives one symmetric and one antisymmetric channel,
%   i.e., (ps,pa) = (1,1), so that the rotation layers have no angles.
%   The network then always realizes the block DCT (Haar transform)
%   regardless of OverlappingFactor. Use Stride 4 or larger for a
%   trainable overlapped transform. A warning
%   'LsunLayer:NoAngles' is issued for Stride 2 with OverlappingFactor
%   greater than 1.
%
%   Outputs of the analyzer at each level:
%      AC: 1 x nBlks x (Stride-1) x nSamples
%      DC: 1 x nBlks x 1 x nSamples
%
% Requirements: MATLAB R2022b
%
% Copyright (c) 2026, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%
if isempty(lsunLgraph)
    lsunLgraph = layerGraph;
end

import tansacnet.lsun.*
p = inputParser;
addParameter(p,'InputSize',32)
addParameter(p,'NumberOfComponents',1)
addParameter(p,'Stride',2)
addParameter(p,'OverlappingFactor',1)
addParameter(p,'NumberOfLevels',1);
addParameter(p,'NumberOfVanishingMoments',[1 1]);
addParameter(p,'Mode','Whole');
addParameter(p,'Prefix','');
addParameter(p,'AppendInOutLayers',true);
addParameter(p,'BoundaryOperation','Circular');
%
if canUseGPU
    addParameter(p,'Device','cuda');
else
    addParameter(p,'Device','cpu');
end
addParameter(p,'DType','single');
%
parse(p,varargin{:})

% Layer constructor function goes here.
nComponents = p.Results.NumberOfComponents;
inputSize = p.Results.InputSize;
stride = p.Results.Stride;
ovlpFactor = p.Results.OverlappingFactor;
nLevels = p.Results.NumberOfLevels;
%
device = p.Results.Device;
dtype = p.Results.DType;
devOpts = {'Device',device,'DType',dtype};
%
noDcLeakage = p.Results.NumberOfVanishingMoments;
if isscalar(noDcLeakage)
    noDcLeakage = [1 1]*noDcLeakage;
end
mode = p.Results.Mode;
prefix = p.Results.Prefix;
isapndinout = p.Results.AppendInOutLayers;

% Boundary operation of the odd-order intermediate rotation layers,
% i.e., those right after the shift of the difference channels.
% 'Termination' fixes their rotation matrices of the boundary blocks to -I
% so that no coefficients wrap around to the opposite boundary.
% The even-order layers are always 'Circular'.
boundaryOperation = p.Results.BoundaryOperation;
if ~any(strcmp(boundaryOperation,{'Circular','Termination'}))
    throw(MException('LsunLayer:InvalidBoundaryOperation',...
        '%s : BoundaryOperation should be either of Circular or Termination',...
        boundaryOperation))
end

if strcmp(mode,'Whole')
    isAnalyzer = true;
    isSynthesizer = true;
elseif strcmp(mode,'Analyzer')
    isAnalyzer = true;
    isSynthesizer = false;
elseif strcmp(mode,'Synthesizer')
    isAnalyzer = false;
    isSynthesizer = true;
else
    error('Mode should be in { ''Whole'', ''Analyzer'', ''Synthesizer'' }');
end

if ~isscalar(stride) || ~isscalar(ovlpFactor) || ~isscalar(inputSize)
    error('InputSize, Stride and OverlappingFactor should be scalars.')
end
if nComponents ~= 1
    throw(MException('LsunLayer:InvalidNumberOfComponents',...
        '%d : Currently, single component is only supported.',...
        nComponents))
end

nDecs = prod(stride);
nChannels = [ceil(nDecs/2) floor(nDecs/2)];

if nChannels(1) ~= nChannels(2)
    throw(MException('LsunLayer:InvalidStride',...
        '%d : Stride should be even.',stride))
end
if stride == 2 && ovlpFactor > 1
    % (ps,pa) = (1,1): No angles in the rotation layers
    warning('LsunLayer:NoAngles',...
        ['Stride 2 has no angles in the rotation layers. ' ...
        'The network is always the block DCT (Haar transform) ' ...
        'regardless of OverlappingFactor %d. ' ...
        'Use Stride 4 or larger for an overlapped transform.'],ovlpFactor)
end
if ~all(mod(ovlpFactor,2))
    throw(MException('LsunLayer:InvalidOverlappingFactor',...
        '%d : Currently, odd overlapping factors are only supported.',...
        ovlpFactor))
end

%%
if mod(inputSize/(stride^nLevels),1)
    error('%d : Input size should be multiple of stride^nLevels.',...
        inputSize)
end

%%
blockDctLayers = cell(nLevels,1);
analysisLayers = cell(nLevels,nComponents);
blockIdctLayers = cell(nLevels,1);
synthesisLayers = cell(nLevels,nComponents);
for iLv = 1:nLevels
    strLv = sprintf('Lv%0d_',iLv);
    nBlocks = inputSize/(stride^iLv);

    % Initial blocks
    blockDctLayers{iLv} = lsunBlockDct1dLayer('Name',[prefix strLv 'E0'],...
        'Stride',stride,...
        'NumberOfComponents',nComponents);
    % Final blocks
    blockIdctLayers{iLv} = lsunBlockIdct1dLayer('Name',[prefix strLv 'E0~'],...
        'Stride',stride,...
        'NumberOfComponents',nComponents);

    % NOTE on the parameter Mus
    %
    % The default Mus of the intermediate rotation layers is -1 so that
    % the default setting becomes a block DCT, as in
    % fcn_createlsunlgraph2d.
    %
    % Reference:
    % S. Muramatsu, T. Kobayashi, M. Hiki and H. Kikuchi, "Boundary Operation of 2-D Nonseparable Linear-Phase Paraunitary Filter Banks," in IEEE Transactions on Image Processing, vol. 21, no. 4, pp. 2314-2318, April 2012, doi: 10.1109/TIP.2011.2181527.
    %
    for iCmp = 1:nComponents
        strCmp = sprintf('Cmp%0d_',iCmp);
        s = [prefix strLv strCmp];
        rotOpts = [{'Stride',stride,'NumberOfBlocks',nBlocks},devOpts];
        %
        ana = lsunInitialRotation1dLayer('Name',[s 'V0'],...
            rotOpts{:},'NoDcLeakage',noDcLeakage(1));
        syn = lsunFinalRotation1dLayer('Name',[s 'V0~'],...
            rotOpts{:},'NoDcLeakage',noDcLeakage(2));
        % Atom extension
        boundaryOpts = {'BoundaryOperation',boundaryOperation};
        for iOrder = 2:2:ovlpFactor-1
            o1 = num2str(iOrder-1);
            o2 = num2str(iOrder);
            ana = [ ana
                lsunAtomExtension1dLayer('Name',[s 'Qx' o1 'rd'],...
                'Stride',stride,'Direction','Right','TargetChannels','Difference')
                lsunIntermediateRotation1dLayer('Name',[s 'Vx' o1],...
                rotOpts{:},'Mode','Analysis','Mus',-1,boundaryOpts{:})
                lsunAtomExtension1dLayer('Name',[s 'Qx' o2 'ls'],...
                'Stride',stride,'Direction','Left','TargetChannels','Sum')
                lsunIntermediateRotation1dLayer('Name',[s 'Vx' o2],...
                rotOpts{:},'Mode','Analysis','Mus',-1)
                ]; %#ok<AGROW>
            syn = [ syn
                lsunAtomExtension1dLayer('Name',[s 'Qx' o1 'rd~'],...
                'Stride',stride,'Direction','Left','TargetChannels','Difference')
                lsunIntermediateRotation1dLayer('Name',[s 'Vx' o1 '~'],...
                rotOpts{:},'Mode','Synthesis','Mus',-1,boundaryOpts{:})
                lsunAtomExtension1dLayer('Name',[s 'Qx' o2 'ls~'],...
                'Stride',stride,'Direction','Right','TargetChannels','Sum')
                lsunIntermediateRotation1dLayer('Name',[s 'Vx' o2 '~'],...
                rotOpts{:},'Mode','Synthesis','Mus',-1)
                ]; %#ok<AGROW>
        end
        % Channel separation and concatenation
        analysisLayers{iLv,iCmp} = [ ana
            lsunChannelSeparation1dLayer('Name',[s 'Sp'],'ChannelDimension',3)
            ];
        synthesisLayers{iLv,iCmp} = [ syn
            lsunChannelConcatenation1dLayer('Name',[s 'Cn'],'ChannelDimension',3)
            ];
    end

end

%% Analysis layers
if isAnalyzer
    for iLv = 1:nLevels
        strLv = sprintf('Lv%0d_',iLv);
        layers = [
            lsunIdentityLayer('Name',[prefix strLv 'In'])
            blockDctLayers{iLv}
            ];
        if iLv == 1 && isapndinout
            layers = [
                imageInputLayer([1 inputSize nComponents],...
                'Name',[prefix 'Sequence input'],...
                'Normalization','none')
                layers
                ];
        end
        lsunLgraph = lsunLgraph.addLayers(layers);
        if iLv > 1
            strLvPre = sprintf('Lv%0d_',iLv-1);
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLvPre 'DcOut'],[prefix strLv 'In']);
        end
        for iCmp = 1:nComponents
            strCmp = sprintf('Cmp%0d_',iCmp);
            lsunLgraph = lsunLgraph.addLayers(analysisLayers{iLv,iCmp});
            if nComponents > 1
                lsunLgraph = lsunLgraph.connectLayers(...
                    [prefix strLv 'E0/out' num2str(iCmp)], [prefix strLv strCmp 'V0']);
            else
                lsunLgraph = lsunLgraph.connectLayers(...
                    [prefix strLv 'E0'], [prefix strLv strCmp 'V0']);
            end
        end
        % Output
        if nComponents > 1
            lsunLgraph = lsunLgraph.addLayers(...
                depthConcatenationLayer(nComponents,'Name',[prefix strLv 'AcOut']));
            lsunLgraph = lsunLgraph.addLayers(...
                depthConcatenationLayer(nComponents,'Name',[prefix strLv 'DcOut']));
            for iCmp = 1:nComponents
                strCmp = sprintf('Cmp%0d_',iCmp);
                lsunLgraph = lsunLgraph.connectLayers(...
                    [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' num2str(iCmp)]);
                lsunLgraph = lsunLgraph.connectLayers(...
                    [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' num2str(iCmp)]);
            end
        else
            lsunLgraph = lsunLgraph.addLayers(...
                lsunIdentityLayer('Name',[prefix strLv 'AcOut']));
            lsunLgraph = lsunLgraph.addLayers(...
                lsunIdentityLayer('Name',[prefix strLv 'DcOut']));
            strCmp = 'Cmp1_';
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' ]);
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' ]);
        end
    end
end

%% Synthesis layers
if isSynthesizer
    for iLv = nLevels:-1:1
        strLv = sprintf('Lv%0d_',iLv);
        lsunLgraph = lsunLgraph.addLayers(...
            lsunComponentSeparation1dLayer(nComponents,'Name',[prefix strLv 'DcIn']));
        lsunLgraph = lsunLgraph.addLayers(...
            lsunComponentSeparation1dLayer(nComponents,'Name',[prefix strLv 'AcIn']));
        if iLv < nLevels
            strLvPre = sprintf('Lv%0d_',iLv+1);
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLvPre 'Out'],[prefix strLv 'DcIn']);
        end
        for iCmp = 1:nComponents
            strCmp = sprintf('Cmp%0d_',iCmp);
            if nComponents > 1
                strOut = ['/out' num2str(iCmp)];
            else
                strOut = '/out';
            end
            lsunLgraph = lsunLgraph.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLv 'AcIn' strOut], [prefix strLv strCmp 'Cn/ac']);
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLv 'DcIn' strOut], [prefix strLv strCmp 'Cn/dc']);
        end
        lsunLgraph = lsunLgraph.addLayers([
            blockIdctLayers{iLv}
            lsunIdentityLayer('Name',[prefix strLv 'Out'])
            ]);
        for iCmp = 1:nComponents
            strCmp = sprintf('Cmp%0d_',iCmp);
            if nComponents > 1
                lsunLgraph = lsunLgraph.connectLayers(...
                    [prefix strLv strCmp 'V0~'],[prefix strLv 'E0~/in' num2str(iCmp)]);
            else
                lsunLgraph = lsunLgraph.connectLayers(...
                    [prefix strLv strCmp 'V0~'],[prefix strLv 'E0~']);
            end
        end
    end
end
if ~isAnalyzer && isapndinout
    for iLv = 1:nLevels
        strLv = sprintf('Lv%0d_',iLv);
        inputSubSize = [ 1 inputSize/(stride^iLv) ...
            nComponents*(sum(nChannels)-1) ];
        lsunLgraph = lsunLgraph.addLayers(...
            imageInputLayer(inputSubSize,...
            'Name',[prefix strLv 'Ac feature input'],...
            'Normalization','none'));
        lsunLgraph = lsunLgraph.connectLayers(...
            [prefix strLv 'Ac feature input'],[prefix strLv 'AcIn'] );
    end
    strLv = sprintf('Lv%0d_',nLevels);
    inputSubSize = [ 1 inputSize/(stride^nLevels) nComponents ];
    lsunLgraph = lsunLgraph.addLayers(...
        imageInputLayer(inputSubSize,...
        'Name',[prefix strLv 'Dc feature input'],...
        'Normalization','none'));
    lsunLgraph = lsunLgraph.connectLayers(...
        [prefix strLv 'Dc feature input'],[prefix strLv 'DcIn']);
end

%% Connect analyzer and synthesizer
if isAnalyzer && isSynthesizer
    strLv = sprintf('Lv%0d_',nLevels);
    lsunLgraph = lsunLgraph.connectLayers([prefix strLv 'DcOut'],[prefix strLv 'DcIn']);
    for iLv = nLevels:-1:1
        strLv = sprintf('Lv%0d_',iLv);
        lsunLgraph = lsunLgraph.connectLayers(...
            [prefix strLv 'AcOut'],[prefix strLv 'AcIn']);
    end
    if isapndinout
        lsunLgraph = lsunLgraph.addLayers(...
            regressionLayer('Name',[prefix 'Sequence output']));
        lsunLgraph = lsunLgraph.connectLayers(...
            [prefix 'Lv1_Out'],[prefix 'Sequence output']);
    end
end
end
