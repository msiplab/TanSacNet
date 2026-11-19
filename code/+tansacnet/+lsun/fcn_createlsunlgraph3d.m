function lsunLgraph = ...
    fcn_createlsunlgraph3d(lsunLgraph,varargin)
%FCN_CREATELSUNLGRAPH3D
%
%   3-D counterpart of tansacnet.lsun.fcn_createlsunlgraph2d.
%
%   The atoms are extended in the depth, horizontal and vertical
%   directions in this order. Stride, OverlappingFactor and InputSize are
%   given in the order of [vertical horizontal depth].
%
% Requirements: MATLAB R2022a
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
addParameter(p,'InputSize',[32 32 32])
addParameter(p,'NumberOfComponents',1)
addParameter(p,'Stride',[2 2 2])
addParameter(p,'OverlappingFactor',[1 1 1])
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
inputSize = [p.Results.InputSize nComponents];
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

if numel(stride) ~= 3 || numel(ovlpFactor) ~= 3 || numel(inputSize) ~= 4
    error('InputSize, Stride and OverlappingFactor should have three elements.')
end

nDecs = prod(stride);
nChannels = [ceil(nDecs/2) floor(nDecs/2)];

if nChannels(1) ~= nChannels(2)
    throw(MException('LsunLayer:InvalidStride',...
        '[%d %d %d] : The product of stride should be even.',...
        stride(1),stride(2),stride(3)))
end
if ~all(mod(ovlpFactor,2))
    throw(MException('LsunLayer:InvalidOverlappingFactor',...
        '[%d %d %d] : Currently, odd overlapping factors are only supported.',...
        ovlpFactor(1),ovlpFactor(2),ovlpFactor(3)))
end

%%
if any(mod(inputSize(1:3)./(stride.^nLevels),1))
    error('[%d %d %d] : Input size should be multiple of stride^nLevels.',...
        inputSize(1),inputSize(2),inputSize(3))
end

%%
% Atom extension steps
% (overlapping-factor index, label, analysis directions, synthesis
%  directions, suffixes, boundary direction)
steps = {
    3, 'd', {'Back','Front'}, {'Front','Back'}, {'bd','fs'}, 'Depth'
    2, 'h', {'Right','Left'}, {'Left','Right'}, {'rd','ls'}, 'Horizontal'
    1, 'v', {'Down','Up'},    {'Up','Down'},    {'dd','us'}, 'Vertical'
    };

blockDctLayers = cell(nLevels,1);
analysisLayers = cell(nLevels,nComponents);
blockIdctLayers = cell(nLevels,1);
synthesisLayers = cell(nLevels,nComponents);
for iLv = 1:nLevels
    strLv = sprintf('Lv%0d_',iLv);
    nBlocks = inputSize(1:3)./(stride.^iLv);

    % Initial blocks
    blockDctLayers{iLv} = lsunBlockDct3dLayer('Name',[prefix strLv 'E0'],...
        'Stride',stride,...
        'NumberOfComponents',nComponents);
    % Final blocks
    blockIdctLayers{iLv} = lsunBlockIdct3dLayer('Name',[prefix strLv 'E0~'],...
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
        ana = lsunInitialRotation3dLayer('Name',[s 'V0'],...
            rotOpts{:},'NoDcLeakage',noDcLeakage(1));
        syn = lsunFinalRotation3dLayer('Name',[s 'V0~'],...
            rotOpts{:},'NoDcLeakage',noDcLeakage(2));
        % Atom extension in depth, horizontal and vertical
        for iStep = 1:size(steps,1)
            [idx,lbl,adirs,sdirs,sfx,bdir] = steps{iStep,:};
            boundaryOpts = {'BoundaryOperation',boundaryOperation,'Direction',bdir};
            for iOrder = 2:2:ovlpFactor(idx)-1
                o1 = num2str(iOrder-1);
                o2 = num2str(iOrder);
                ana = [ ana
                    lsunAtomExtension3dLayer('Name',[s 'Q' lbl o1 sfx{1}],...
                    'Stride',stride,'Direction',adirs{1},'TargetChannels','Difference')
                    lsunIntermediateRotation3dLayer('Name',[s 'V' lbl o1],...
                    rotOpts{:},'Mode','Analysis','Mus',-1,boundaryOpts{:})
                    lsunAtomExtension3dLayer('Name',[s 'Q' lbl o2 sfx{2}],...
                    'Stride',stride,'Direction',adirs{2},'TargetChannels','Sum')
                    lsunIntermediateRotation3dLayer('Name',[s 'V' lbl o2],...
                    rotOpts{:},'Mode','Analysis','Mus',-1)
                    ]; %#ok<AGROW>
                syn = [ syn
                    lsunAtomExtension3dLayer('Name',[s 'Q' lbl o1 sfx{1} '~'],...
                    'Stride',stride,'Direction',sdirs{1},'TargetChannels','Difference')
                    lsunIntermediateRotation3dLayer('Name',[s 'V' lbl o1 '~'],...
                    rotOpts{:},'Mode','Synthesis','Mus',-1,boundaryOpts{:})
                    lsunAtomExtension3dLayer('Name',[s 'Q' lbl o2 sfx{2} '~'],...
                    'Stride',stride,'Direction',sdirs{2},'TargetChannels','Sum')
                    lsunIntermediateRotation3dLayer('Name',[s 'V' lbl o2 '~'],...
                    rotOpts{:},'Mode','Synthesis','Mus',-1)
                    ]; %#ok<AGROW>
            end
        end
        % Channel separation and concatenation
        analysisLayers{iLv,iCmp} = [ ana
            lsunChannelSeparation3dLayer('Name',[s 'Sp'])
            ];
        synthesisLayers{iLv,iCmp} = [ syn
            lsunChannelConcatenation3dLayer('Name',[s 'Cn'])
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
                image3dInputLayer(inputSize,...
                'Name',[prefix 'Image input'],...
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
            % Concatenation in the channel dimension of SSSCB
            lsunLgraph = lsunLgraph.addLayers(...
                concatenationLayer(4,nComponents,'Name',[prefix strLv 'AcOut']));
            lsunLgraph = lsunLgraph.addLayers(...
                concatenationLayer(4,nComponents,'Name',[prefix strLv 'DcOut']));
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
            lsunComponentSeparation3dLayer(nComponents,'Name',[prefix strLv 'DcIn']));
        lsunLgraph = lsunLgraph.addLayers(...
            lsunComponentSeparation3dLayer(nComponents,'Name',[prefix strLv 'AcIn']));
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
        inputSubSize = [ inputSize(1:3)./(stride.^iLv) ...
            nComponents*(sum(nChannels)-1) ];
        lsunLgraph = lsunLgraph.addLayers(...
            image3dInputLayer(inputSubSize,...
            'Name',[prefix strLv 'Ac feature input'],...
            'Normalization','none'));
        lsunLgraph = lsunLgraph.connectLayers(...
            [prefix strLv 'Ac feature input'],[prefix strLv 'AcIn'] );
    end
    strLv = sprintf('Lv%0d_',nLevels);
    inputSubSize = [ inputSize(1:3)./(stride.^nLevels) nComponents ];
    lsunLgraph = lsunLgraph.addLayers(...
        image3dInputLayer(inputSubSize,...
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
            regressionLayer('Name',[prefix 'Image output']));
        lsunLgraph = lsunLgraph.connectLayers(...
            [prefix 'Lv1_Out'],[prefix 'Image output']);
    end
end
end
