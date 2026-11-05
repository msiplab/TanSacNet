function lgraph = fcn_createconstestimatorlgraph(lgraph,varargin)
%FCN_CREATECONSTESTIMATORLGRAPH
%
%   Constant angle estimator for the estimator interface of
%   tansacnet.salsun.fcn_createsalsunlgraph2d: each block has its own
%   learnable angles, which do not depend on the input, so each rotation
%   behaves as that of LSUN. With ThetaMode 'Reuse', SA-LSUN with this
%   estimator reconstructs its input perfectly for any angles. The
%   estimator does not depend on the number of spatial dimensions.
%
%   Usage:
%     lgraph = fcn_createsalsunlgraph2d([],...,...
%         'EstimatorFunction',@tansacnet.salsun.fcn_createconstestimatorlgraph)
%
%   Input (unconnected):
%     [Prefix 'Ext']   : the block state (only its number of samples is used)
%   Output (unconnected):
%     [Prefix 'Theta'] : nAngles x prod(NumberOfBlocks) x nSamples
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

if isempty(lgraph)
    lgraph = layerGraph;
end

import tansacnet.lsun.*
import tansacnet.salsun.*
p = inputParser;
addParameter(p,'Prefix','')
addParameter(p,'TargetType','Initial')
addParameter(p,'NumberOfChannels',[])
addParameter(p,'NoDcLeakage',false)
addParameter(p,'NumberOfBlocks',[])
addParameter(p,'Stride',[])
addParameter(p,'Side','Analysis')
addParameter(p,'StageName','')
parse(p,varargin{:})

prefix = p.Results.Prefix;
[nAnglesFull,nZeroPad] = fcn_salsunanglecounts(p.Results.TargetType,...
    p.Results.NumberOfChannels,p.Results.NoDcLeakage);

layers = [
    lsunIdentityLayer('Name',[prefix 'Ext'])
    salsunConstantAngleLayer('Name',[prefix 'Theta'],...
        'NumberOfBlocks',p.Results.NumberOfBlocks,...
        'NumberOfAngles',nAnglesFull-nZeroPad,...
        'NumberOfZeroPadAngles',nZeroPad)
    ];
lgraph = lgraph.addLayers(layers);

end
