function lgraph = fcn_createparamestimator2dlgraph(lgraph,varargin)
%FCN_CREATEPARAMESTIMATOR2DLGRAPH
%
%   Input (unconnected):
%     [Prefix 'Extract'] : p x nRows x nCols x nSamples -- the block
%                          state to be watched by the estimator (e.g.
%                          the output of the input-layer orthonormal
%                          transform).
%   Output (unconnected):
%     [Prefix 'Theta']   : nAngles x (nRows*nCols) x nSamples, to be
%                          connected to the 'theta' input of the
%                          corresponding rotation layer.
%
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
addParameter(p,'NoDcLeakage',false)
parse(p,varargin{:})

prefix = p.Results.Prefix;
targetType = p.Results.TargetType;
nChs = p.Results.NumberOfChannels;
nNeighbor = p.Results.NumberOfNeighborBlocks;
nResBlocks = p.Results.NumberOfResidualBlocks;
width = p.Results.Width;
noDcLeakage = p.Results.NoDcLeakage;

if mod(nChs,2)~=0
    throw(MException('SaLsunLayer:InvalidNumberOfChannels',...
        '%d : NumberOfChannels must be even.',nChs))
end
ps = nChs/2;
pa = nChs/2;

if strcmp(targetType,'Initial')
    nAnglesFull = (nChs-2)*nChs/4; % combined [anglesW; anglesU]
    if noDcLeakage
        channels = 2:nChs;
        nZeroPad = ps-1;
    else
        channels = 1:nChs;
        nZeroPad = 0;
    end
elseif strcmp(targetType,'Intermediate')
    channels = pa+1:nChs; % antisymmetric half only
    nAnglesFull = (nChs-2)*nChs/8;
    nZeroPad = 0; % no-DC-leakage does not apply to intermediate rotations
else
    error('TargetType should be either of ''Initial'' or ''Intermediate''')
end
nAnglesPredicted = nAnglesFull-nZeroPad;
nFeat = numel(channels)*prod(nNeighbor);

layers = [
    salsunLocalStateExtraction2dLayer('Name',[prefix 'Extract'],...
        'NumberOfNeighborBlocks',nNeighbor,'Channels',channels)
    salsunStateStandardization2dLayer('Name',[prefix 'Standardize'])
    ];
for iBlock = 1:nResBlocks
    layers = [layers %#ok<AGROW>
        salsunResidualEstimatorBlock2dLayer('Name',[prefix 'ResBlock' num2str(iBlock)],...
            'InputSize',nFeat,'Width',width)
        ];
end
layers = [layers
    salsunAngleEstimatorOutput2dLayer('Name',[prefix 'Theta'],...
        'InputSize',nFeat,'NumberOfAngles',nAnglesPredicted,...
        'NumberOfZeroPadAngles',nZeroPad)
    ];

lgraph = lgraph.addLayers(layers);

end
