function lsunLgraph = ...
    fcn_createcslsunlgraph1d(lsunLgraph,varargin)
%FCN_CREATECSLSUNLGRAPHS1D
%
% Requirements: MATLAB R2022b
%
% Copyright (c) 2023, Shogo MURAMATSU
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
%addParameter(p,'NumberOfComponents',1)
addParameter(p,'Stride',1)
addParameter(p,'OverlappingFactor',1)
%addParameter(p,'NumberOfLevels',1);
%addParameter(p,'NumberOfVanishingMoments',[1 1]);
addParameter(p,'Mode','Whole');
addParameter(p,'Prefix','');
addParameter(p,'AppendInOutLayers',true);
parse(p,varargin{:})

% Layer constructor function goes here.
nComponents = 1; %p.Results.NumberOfComponents;
inputSize = [1 p.Results.InputSize nComponents];
stride = p.Results.Stride;
ovlpFactor = p.Results.OverlappingFactor;
nLevels = 1; %p.Results.NumberOfLevels;
%noDcLeakage = p.Results.NumberOfVanishingMoments;
%{
if isscalar(noDcLeakage)
    noDcLeakage = [1 1]*noDcLeakage; 
end
%}
mode = p.Results.Mode;
prefix = p.Results.Prefix;
isapndinout = p.Results.AppendInOutLayers;

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

nDecs = prod(stride);
nChannels = [ceil(nDecs/2) floor(nDecs/2)];

if nChannels(1) ~= nChannels(2)
    throw(MException('LsunLayer:InvalidStride',...
        '[%d %d] : The product of stride should be even.',...
        stride(1),stride(2)))
end
if ~all(mod(ovlpFactor,2))
    throw(MException('LsunLayer:InvalidOverlappingFactor',...
        '%d : Currently, odd overlapping factors are only supported.',...
        ovlpFactor))
end

%%
nBlocks = inputSize(2)/stride;
if mod(nBlocks,1)
    error('%d : Input size should be multiple of stride.')
end

%%
blockDctLayers = cell(nLevels);
analysisLayers = cell(nLevels,nComponents);
blockIdctLayers = cell(nLevels);
synthesisLayers = cell(nLevels,nComponents);
for iLv = 1:nLevels
    strLv = sprintf('Lv%0d_',iLv);
    
    % Initial blocks
    blockDctLayers{iLv} = lsunBlockDct1dLayer('Name',[prefix strLv 'E0'],...
        'Stride',stride,...
        'NumberOfComponents',nComponents);
    for iCmp = 1:nComponents
        strCmp = sprintf('Cmp%0d_',iCmp);
        analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
            lsunInitialFullRotation1dLayer('Name',[prefix strLv strCmp 'V0'],...
            'NumberOfBlocks',nBlocks,'Stride',stride) %,...
            ...'NoDcLeakage',noDcLeakage(1))
            ];
    end
    % Final blocks
    blockIdctLayers{iLv} = lsunBlockIdct1dLayer('Name',[prefix strLv 'E0~'],...
        'Stride',stride,...
        'NumberOfComponents',nComponents);
    for iCmp = 1:nComponents
        strCmp = sprintf('Cmp%0d_',iCmp);
        synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
            lsunFinalFullRotation1dLayer('Name',[prefix strLv strCmp 'V0~'],...
            'NumberOfBlocks',nBlocks,'Stride',stride)%,...
            ...'NoDcLeakage',noDcLeakage(2))
            ];
    end
    
    for iCmp = 1:nComponents
        strCmp = sprintf('Cmp%0d_',iCmp);
       
        % Atom extension 
        for iOrderV = 2:2:ovlpFactor-1
            analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                lsunCSAtomExtension1dLayer('Name',[prefix strLv strCmp 'Qx' num2str(iOrderV-1) 'rb'],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Direction','Right','TargetChannels','Bottom','Mode','Analysis')
                lsunIntermediateFullRotation1dLayer('Name',[prefix strLv strCmp 'Vx' num2str(iOrderV-1)],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis') %,'Mus',-1)
                lsunCSAtomExtension1dLayer('Name',[prefix strLv strCmp 'Qx' num2str(iOrderV) 'lt'],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Direction','Left','TargetChannels','Top','Mode','Analysis')
                lsunIntermediateFullRotation1dLayer('Name',[prefix strLv strCmp 'Vx' num2str(iOrderV)],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis')
                ];
            synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                lsunCSAtomExtension1dLayer('Name',[prefix strLv strCmp 'Qx' num2str(iOrderV-1) 'rb~'],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Direction','Left','TargetChannels','Bottom','Mode','Synthesis')
                lsunIntermediateFullRotation1dLayer('Name',[prefix strLv strCmp 'Vx' num2str(iOrderV-1) '~'],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis') %,'Mus',-1)
                lsunCSAtomExtension1dLayer('Name',[prefix strLv strCmp 'Qx' num2str(iOrderV) 'lt~'],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Direction','Right','TargetChannels','Top','Mode','Synthesis')
                lsunIntermediateFullRotation1dLayer('Name',[prefix strLv strCmp 'Vx' num2str(iOrderV) '~'],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis')
                ];
        end
        
        % Channel separation and concatenation
        analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
            lsunChannelSeparation1dLayer('Name',[prefix strLv strCmp 'Sp'])
            ];
        synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
            lsunChannelConcatenation1dLayer('Name',[prefix strLv strCmp 'Cn'])
            ];
    end
    
end



%% Analysis layers
if isAnalyzer
    % Level 1
    iLv = 1;
    strLv = sprintf('Lv%0d_',iLv);
    if isapndinout
        lsunLgraph = lsunLgraph.addLayers(...
            [ imageInputLayer(inputSize,...
            'Name',[prefix 'Seq. input'],...
            'Normalization','none'),...
            lsunIdentityLayer(...
            'Name',[prefix strLv 'In']),...
            blockDctLayers{iLv}
            ]);
    else
        lsunLgraph = lsunLgraph.addLayers(...
            [lsunIdentityLayer(...
            'Name',[prefix strLv 'In']),...
            blockDctLayers{iLv}]);
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
    else
        lsunLgraph = lsunLgraph.addLayers(...
            lsunIdentityLayer('Name',[prefix strLv 'AcOut']));
        lsunLgraph = lsunLgraph.addLayers(...
            lsunIdentityLayer('Name',[prefix strLv 'DcOut']));
    end
    if nComponents > 1
        for iCmp = 1:nComponents
            strCmp = sprintf('Cmp%0d_',iCmp);
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' num2str(iCmp)]);
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' num2str(iCmp)]);
        end
    else
        strCmp = 'Cmp1_';
        lsunLgraph = lsunLgraph.connectLayers(...
            [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' ]);
        lsunLgraph = lsunLgraph.connectLayers(...
            [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' ]);
    end
    % Level n > 1
    for iLv = 2:nLevels
        strLv = sprintf('Lv%0d_',iLv);
        strLvPre = sprintf('Lv%0d_',iLv-1);
        lsunLgraph = lsunLgraph.addLayers([
            lsunIdentityLayer('Name',[prefix strLv 'In']),...
            blockDctLayers{iLv}]);
        lsunLgraph = lsunLgraph.connectLayers([prefix strLvPre 'DcOut'],[prefix strLv 'In']);
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
        else
            lsunLgraph = lsunLgraph.addLayers(...
                lsunIdentityLayer('Name',[prefix strLv 'AcOut']));
            lsunLgraph = lsunLgraph.addLayers(...
                lsunIdentityLayer('Name',[prefix strLv 'DcOut']));
        end
        if nComponents > 1
            for iCmp = 1:nComponents
                strCmp = sprintf('Cmp%0d_',iCmp);
                lsunLgraph = lsunLgraph.connectLayers(...
                    [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' num2str(iCmp)]);
                lsunLgraph = lsunLgraph.connectLayers(...
                    [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' num2str(iCmp)]);
            end
        else
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
    % Level N
    iLv = nLevels;
    strLv = sprintf('Lv%0d_',iLv);
    lsunLgraph = lsunLgraph.addLayers(...
        lsunComponentSeparation1dLayer(nComponents,'Name',[prefix strLv 'DcIn']));
    lsunLgraph = lsunLgraph.addLayers(...
        lsunComponentSeparation1dLayer(nComponents,'Name',[prefix strLv 'AcIn']));
    if nComponents > 1
        for iCmp = 1:nComponents
            strCmp = sprintf('Cmp%0d_',iCmp);
            lsunLgraph = lsunLgraph.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLv 'AcIn/out' num2str(iCmp) ], [prefix strLv strCmp 'Cn/ac']);
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLv 'DcIn/out' num2str(iCmp) ], [prefix strLv strCmp 'Cn/dc']);
        end
    else
        strCmp = 'Cmp1_';
        lsunLgraph = lsunLgraph.addLayers(synthesisLayers{iLv,1}(end:-1:1));
        lsunLgraph = lsunLgraph.connectLayers(...
            [prefix strLv 'AcIn/out' ], [prefix strLv strCmp 'Cn/ac']);
        lsunLgraph = lsunLgraph.connectLayers(...
            [prefix strLv 'DcIn/out' ], [prefix strLv strCmp 'Cn/dc']);
    end
    lsunLgraph = lsunLgraph.addLayers([
        blockIdctLayers{iLv},...
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
    
    % Level n < N
    for iLv = nLevels-1:-1:1
        strLv = sprintf('Lv%0d_',iLv);
        strLvPre = sprintf('Lv%0d_',iLv+1);
        if nComponents > 1
            lsunLgraph = lsunLgraph.addLayers(...
                lsunComponentSeparation1dLayer(nComponents,'Name',[prefix strLv 'DcIn']));
            lsunLgraph = lsunLgraph.addLayers(...
                lsunComponentSeparation1dLayer(nComponents,'Name',[prefix strLv 'AcIn']));            
        else
            lsunLgraph = lsunLgraph.addLayers(...
                lsunIdentityLayer('Name',[prefix strLv 'DcIn']));
            lsunLgraph = lsunLgraph.addLayers(...
                lsunIdentityLayer('Name',[prefix strLv 'AcIn']));            
        end
        lsunLgraph = lsunLgraph.connectLayers([prefix strLvPre 'Out'],[prefix strLv 'DcIn']);
        if nComponents > 1
            for iCmp = 1:nComponents
                strCmp = sprintf('Cmp%0d_',iCmp);
                lsunLgraph = lsunLgraph.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
                lsunLgraph = lsunLgraph.connectLayers(...
                    [prefix strLv 'AcIn/out' num2str(iCmp) ], [prefix strLv strCmp 'Cn/ac']);
                lsunLgraph = lsunLgraph.connectLayers(...
                    [prefix strLv 'DcIn/out' num2str(iCmp) ], [prefix strLv strCmp 'Cn/dc']);
            end
        else
            strCmp = 'Cmp1_';
            lsunLgraph = lsunLgraph.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLv 'AcIn/out' ], [prefix strLv strCmp 'Cn/ac']);
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLv 'DcIn/out'  ], [prefix strLv strCmp 'Cn/dc']);
        end
        lsunLgraph = lsunLgraph.addLayers([
            blockIdctLayers{iLv},...
            lsunIdentityLayer('Name',[prefix strLv 'Out'])
            ]);
        for iCmp = 1:nComponents
            strCmp = sprintf('Cmp%0d_',iCmp);
            if nComponents > 1
                lsunLgraph = lsunLgraph.connectLayers(...
                    [prefix strLv strCmp 'V0~'],[prefix strLv 'E0~/in' num2str(iCmp)]);
            else
                lsunLgraph = lsunLgraph.connectLayers(...
                    [prefix strLv strCmp 'V0~'], [prefix strLv 'E0~']);
            end
        end
    end

end
if ~isAnalyzer
    for iLv = 1:nLevels
        strLv = sprintf('Lv%0d_',iLv);
        inputSubSize(1) = sum(nChannels)-1;
        inputSubSize(2) = 1;
        inputSubSize(3) = inputSize(2)/(stride.^iLv);
        if isapndinout
            lsunLgraph = lsunLgraph.addLayers(...
                imageInputLayer(inputSubSize,...
                'Name',[prefix  strLv 'Ac feature input'],...
                'Normalization','none'));
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLv 'Ac feature input'],[prefix strLv 'AcIn'] );
        end
    end
    strLv = sprintf('Lv%0d_',nLevels);
    inputSubSize(1) = 1;
    inputSubSize(2) = 1;
    inputSubSize(3) = inputSize(2)/(stride.^nLevels);
    if isapndinout
        lsunLgraph = lsunLgraph.addLayers(...
            imageInputLayer(inputSubSize,...
            'Name',[prefix  strLv 'Dc feature input'],...
            'Normalization','none'));
        if iLv == nLevels
            lsunLgraph = lsunLgraph.connectLayers(...
                [prefix strLv 'Dc feature input'],[prefix strLv 'DcIn']);
        end
    end
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
            regressionLayer('Name',[prefix 'Seq. output']));
        lsunLgraph = lsunLgraph.connectLayers('Lv1_Out',[prefix 'Seq. output']);
    end
end
end
