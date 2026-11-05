function lgraph = fcn_createparamestimator2dlgraph(lgraph,varargin)
%FCN_CREATEPARAMESTIMATOR2DLGRAPH
%
%   The default angle estimator of tansacnet.salsun.fcn_createsalsunlgraph2d
%   (see there for the estimator interface): local state extraction,
%   standardization, an optional bottleneck projection, residual blocks
%   and an output layer.
%
%   Input (unconnected):
%     [Prefix 'Ext']   : p x nRows x nCols x nSamples -- the block
%                        state to be watched by the estimator (e.g.
%                        the output of the input-layer orthonormal
%                        transform).
%   Output (unconnected):
%     [Prefix 'Theta'] : nAngles x (nRows*nCols) x nSamples, to be
%                        connected to the 'theta' input of the
%                        corresponding rotation layer.
%
%   NumberOfBlocks, Stride, Side and StageName, which the builder passes
%   to every estimator, are accepted and not used.
%
% Requirements: MATLAB R2022a
%
% Copyright (c) 2026, Motoyasu SUZUKI and Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%                Faculty of Engineering, Niigata University,
%                8050 2-no-cho Ikarashi, Nishi-ku,
%                Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/

if isempty(lgraph)
    lgraph = layerGraph;
end

import tansacnet.salsun.*
p = inputParser;
addParameter(p,'Prefix','')
addParameter(p,'TargetType','Initial')
addParameter(p,'NumberOfChannels',[])
addParameter(p,'NumberOfNeighborBlocks',[3 3])
addParameter(p,'NumberOfResidualBlocks',3)
addParameter(p,'Width',2)
addParameter(p,'BottleneckSize',0)
addParameter(p,'NoDcLeakage',false)
% Context passed by fcn_createsalsunlgraph2d to every estimator (unused)
addParameter(p,'NumberOfBlocks',[])
addParameter(p,'Stride',[])
addParameter(p,'Side','Analysis')
addParameter(p,'StageName','')
parse(p,varargin{:})

prefix = p.Results.Prefix;
targetType = p.Results.TargetType;
nChs = p.Results.NumberOfChannels;
nNeighbor = p.Results.NumberOfNeighborBlocks;
nResBlocks = p.Results.NumberOfResidualBlocks;
width = p.Results.Width;
bottleneck = p.Results.BottleneckSize;
noDcLeakage = p.Results.NoDcLeakage;

[nAnglesFull,nZeroPad] = fcn_salsunanglecounts(targetType,nChs,noDcLeakage);
pa = nChs/2;
if strcmp(targetType,'Initial')
    if noDcLeakage
        channels = 2:nChs;
    else
        channels = 1:nChs;
    end
else
    channels = pa+1:nChs; % antisymmetric half only
end
nAnglesPredicted = nAnglesFull-nZeroPad;
nFeat = numel(channels)*prod(nNeighbor);

layers = [
    salsunLocalStateExtraction2dLayer('Name',[prefix 'Ext'],...
        'NumberOfNeighborBlocks',nNeighbor,'Channels',channels)
    salsunStateStandardization2dLayer('Name',[prefix 'Std'])
    ];
% Optional bottleneck: a learned linear projection of the standardized
% local state to BottleneckSize features, on which the residual blocks
% and the output layer operate (fewer multiply-adds for the same
% neighborhood)
if bottleneck > 0
    layers = [layers
        salsunFeatureProjection2dLayer('Name',[prefix 'Proj'],...
            'InputSize',nFeat,'OutputSize',bottleneck)
        ];
    nInner = bottleneck;
else
    nInner = nFeat;
end
for iBlock = 1:nResBlocks
    layers = [layers %#ok<AGROW>
        salsunResidualEstimatorBlock2dLayer('Name',[prefix 'ResBlk' num2str(iBlock)],...
            'InputSize',nInner,'Width',width)
        ];
end
layers = [layers
    salsunAngleEstimatorOutput2dLayer('Name',[prefix 'Theta'],...
        'InputSize',nInner,'NumberOfAngles',nAnglesPredicted,...
        'NumberOfZeroPadAngles',nZeroPad)
    ];

lgraph = lgraph.addLayers(layers);

end
