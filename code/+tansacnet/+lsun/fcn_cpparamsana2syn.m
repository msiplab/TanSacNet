function synthesislgraph = fcn_cpparamsana2syn(synthesislgraph,analysislgraph)
%FCN_CPPARAMSSYN2ANA
%
% Setting up the synthesis dictionary (adjoint operator) by copying
% analysis dictionary parameters to the synthesis dictionary
%
% Requirements: MATLAB R2022a
%
% Copyright (c) 2022, Shogo MURAMATSU
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
import tansacnet.lsun.*
%
expanalyzer = '^Lv\d+_Cmp\d+_V(\w\d|0)+$';
nLayers = height(analysislgraph.Layers);
for iLayer = 1:nLayers
    alayer = analysislgraph.Layers(iLayer);
    alayerName = alayer.Name;
    if ~isempty(regexp(alayerName,expanalyzer,'once'))
        slayer = synthesislgraph.Layers({synthesislgraph.Layers.Name} == alayerName + "~");
        slayer.Angles = alayer.Angles;
        slayer.Mus = alayer.Mus;
        if isa(alayer,'tansacnet.lsun.lsunInitialRotation2dLayer')
            slayer.NoDcLeakage = alayer.NoDcLeakage;
        end
        synthesislgraph = synthesislgraph.replaceLayer(slayer.Name,slayer);
        disp("Copy angles from " + alayerName + " to " + slayer.Name)
    end
end
end