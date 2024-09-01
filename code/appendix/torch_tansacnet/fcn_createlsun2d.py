import math
import argparse
import torch.nn as nn

def fcn_createlsun2d(lsunLgraph, **kwargs):
    """
    FCN_CREATELSUN2D

    Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
    Copyright (c) 2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        https://www.eng.niigata-u.ac.jp/~msiplab/
    """
    from torch_tansacnet import Direction, InvalidMode, InvalidStride, InvalidOverlappingFactor

    if lsunLgraph is None:
        lsunLgraph = nn.Sequential()

    parser = argparse.ArgumentParser()
    parser.add_argument('--InputSize', type=list, default=[32, 32])
    parser.add_argument('--NumberOfComponents', type=int, default=1)
    parser.add_argument('--Stride', type=list, default=[2, 2])
    parser.add_argument('--OverlappingFactor', type=list, default=[1, 1])
    parser.add_argument('--NumberOfLevels', type=int, default=1)
    parser.add_argument('--NumberOfVanishingMoments', type=list, default=[1, 1])
    parser.add_argument('--Mode', type=str, default='Whole')
    parser.add_argument('--Prefix', type=str, default='')
    parser.add_argument('--AppendInOutLayers', type=bool, default=True)
    args = parser.parse_args(kwargs)

    # Layer constructor function goes here.
    nComponents = args.NumberOfComponents
    inputSize = (args.InputSize[Direction.VERTICAL], args.InputSize[Direction.HORIZONTAL], nComponents)
    stride = args.Stride
    ovlpFactor = args.OverlappingFactor
    nLevels = args.NumberOfLevels
    noDcLeakage = args.NumberOfVanishingMoments
    if isinstance(noDcLeakage, int):
        noDcLeakage = (noDcLeakage, noDcLeakage)
    mode = args.Mode
    prefix = args.Prefix
    isapndinout = args.AppendInOutLayers

    if mode == 'Whole':
        isAnalyzer = True
        isSynthesizer = True
    elif mode == 'Analyzer':
        isAnalyzer = True
        isSynthesizer = False
    elif mode == 'Synthesizer':
        isAnalyzer = False
        isSynthesizer = True
    else:
        raise InvalidMode('Mode should be in { ''Whole'', ''Analyzer'', ''Synthesizer'' }')

    nDecs = math.prod(stride)
    nChannels = (math.ceil(nDecs/2), math.floor(nDecs/2))

    if nChannels[Direction.VERTICAL] != nChannels[Direction.HORIZONTAL]:
        raise InvalidStride(f'[{stride[Direction.VERTICAL]} {stride[Direction.HORIZONTAL]}] : The product of stride should be even.')
    ovlpFactor_remainder = [ovlpFactor[Direction.VERTICAL] % 2, ovlpFactor[Direction.HORIZONTAL] % 2]
    if not all(ovlpFactor_remainder):
        raise InvalidOverlappingFactor(f'[{ovlpFactor[Direction.VERTICAL]} {ovlpFactor[Direction.HORIZONTAL]}] : Currently, odd overlapping factors are only supported.')

    # 
    nBlocks = (inputSize[Direction.VERTICAL]//stride[Direction.VERTICAL], inputSize[Direction.HORIZONTAL]//stride[Direction.HORIZONTAL])
    if any([nBlocks[Direction.VERTICAL] % 1, nBlocks[Direction.HORIZONTAL] % 1]):
        raise InvalidStride(f'[{inputSize[Direction.VERTICAL]} {inputSize[Direction.HORIZONTAL]}] : Input size should be multiple of stride.')

    return lsunLgraph
"""
%%
blockDctLayers = cell(nLevels);
analysisLayers = cell(nLevels,nComponents);
blockIdctLayers = cell(nLevels);
synthesisLayers = cell(nLevels,nComponents);
for iLv = 1:nLevels
    strLv = sprintf('Lv%0d_',iLv);
    
    % Initial blocks
    blockDctLayers{iLv} = lsunBlockDct2dLayer('Name',[prefix strLv 'E0'],...
        'Stride',stride,...
        'NumberOfComponents',nComponents);
    for iCmp = 1:nComponents
        strCmp = sprintf('Cmp%0d_',iCmp);
        analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
            lsunInitialRotation2dLayer('Name',[prefix strLv strCmp 'V0'],...
            'NumberOfBlocks',nBlocks,'Stride',stride,...
            'NoDcLeakage',noDcLeakage(1))
            ];
    end
    % Final blocks
    blockIdctLayers{iLv} = lsunBlockIdct2dLayer('Name',[prefix strLv 'E0~'],...
        'Stride',stride,...
        'NumberOfComponents',nComponents);
    for iCmp = 1:nComponents
        strCmp = sprintf('Cmp%0d_',iCmp);
        synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
            lsunFinalRotation2dLayer('Name',[prefix strLv strCmp 'V0~'],...
            'NumberOfBlocks',nBlocks,'Stride',stride,...
            'NoDcLeakage',noDcLeakage(2))
            ];
    end
    
    for iCmp = 1:nComponents
        strCmp = sprintf('Cmp%0d_',iCmp);
        % Atom extension in horizontal
        for iOrderH = 2:2:ovlpFactor(2)-1
            analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qh' num2str(iOrderH-1) 'rd'],...
                'Stride',stride,'Direction','Right','TargetChannels','Difference')
                lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vh' num2str(iOrderH-1)],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',-1)
                lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qh' num2str(iOrderH) 'ls'],...
                'Stride',stride,'Direction','Left','TargetChannels','Sum')
                lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vh' num2str(iOrderH) ],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis')
                ];
            synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qh' num2str(iOrderH-1) 'rd~'],...
                'Stride',stride,'Direction','Left','TargetChannels','Difference')
                lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vh' num2str(iOrderH-1) '~'],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',-1)
                lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qh' num2str(iOrderH) 'ls~'],...
                'Stride',stride,'Direction','Right','TargetChannels','Sum')
                lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vh' num2str(iOrderH) '~'],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis')
                ];
        end
        % Atom extension in vertical
        for iOrderV = 2:2:ovlpFactor(1)-1
            analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qv' num2str(iOrderV-1) 'dd'],...
                'Stride',stride,'Direction','Down','TargetChannels','Difference')
                lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vv' num2str(iOrderV-1)],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',-1)
                lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qv' num2str(iOrderV) 'us'],...
                'Stride',stride,'Direction','Up','TargetChannels','Sum')
                lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vv' num2str(iOrderV)],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis')
                ];
            synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qv' num2str(iOrderV-1) 'dd~'],...
                'Stride',stride,'Direction','Up','TargetChannels','Difference')
                lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vv' num2str(iOrderV-1) '~'],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',-1)
                lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qv' num2str(iOrderV) 'us~'],...
                'Stride',stride,'Direction','Down','TargetChannels','Sum')
                lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vv' num2str(iOrderV) '~'],...
                'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis')
                ];
        end
        
        % Channel separation and concatenation
        analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
            lsunChannelSeparation2dLayer('Name',[prefix strLv strCmp 'Sp'])
            ];
        synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
            lsunChannelConcatenation2dLayer('Name',[prefix strLv strCmp 'Cn'])
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
            'Name',[prefix 'Image input'],...
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
%{
if ~isSynthesizer
    for iLv = 1:nLevels
        strLv = sprintf('Lv%0d_',iLv);
        lsunLgraph = lsunLgraph.addLayers(...
            regressionLayer('Name',[prefix  strLv 'Ac feature output']));
        lsunLgraph = lsunLgraph.connectLayers(...
            [prefix strLv 'AcOut'],[prefix strLv 'Ac feature output']);
    end
    strLv = sprintf('Lv%0d_',nLevels);
    lsunLgraph = lsunLgraph.addLayers(...
        regressionLayer('Name',[prefix  strLv 'Dc feature output']));
    lsunLgraph = lsunLgraph.connectLayers(...
            [prefix strLv 'DcOut'],[prefix strLv 'Dc feature output']);    
end
%}


%% Synthesis layers
if isSynthesizer
    % Level N
    iLv = nLevels;
    strLv = sprintf('Lv%0d_',iLv);
    lsunLgraph = lsunLgraph.addLayers(...
        lsunComponentSeparation2dLayer(nComponents,'Name',[prefix strLv 'DcIn']));
    lsunLgraph = lsunLgraph.addLayers(...
        lsunComponentSeparation2dLayer(nComponents,'Name',[prefix strLv 'AcIn']));
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
                lsunComponentSeparation2dLayer(nComponents,'Name',[prefix strLv 'DcIn']));
            lsunLgraph = lsunLgraph.addLayers(...
                lsunComponentSeparation2dLayer(nComponents,'Name',[prefix strLv 'AcIn']));            
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
    
    % Level 1
    %{
    lsunLgraph = lsunLgraph.addLayers(...
        regressionLayer('Name','Image output'));
    lsunLgraph = lsunLgraph.connectLayers('Lv1_Out','Image output');
    %}
end
if ~isAnalyzer
    for iLv = 1:nLevels
        strLv = sprintf('Lv%0d_',iLv);
        inputSubSize(1:2) = inputSize(1:2)./(stride.^iLv);
        inputSubSize(3) = nComponents*(sum(nChannels)-1);
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
    inputSubSize(1:2) = inputSize(1:2)./(stride.^nLevels);
    inputSubSize(3) = nComponents;
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
            regressionLayer('Name',[prefix 'Image output']));
        lsunLgraph = lsunLgraph.connectLayers('Lv1_Out',[prefix 'Image output']);
    end
end
end
"""
