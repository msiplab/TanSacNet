function net = salsun2d_create_test_network(inputSize,coefMask)
%SALSUN2D_CREATE_TEST_NETWORK SA-LSUN dlnetwork with random parameters
%
%   net = salsun2d_create_test_network(inputSize,coefMask) builds the
%   reconstruction network of main_salsun2d.m (Mode 'Whole', ThetaMode
%   'Reuse', same settings) on the CPU, inserts the coefficient mask
%   coefMask (16 x 1, DC first) unless it is empty, and perturbs all
%   learnable parameters randomly. Without the perturbation the angle
%   estimators output zero angles (Wo = 0) and the rotations go untested.
%
%   Used by the test cases; set the random seed before calling.
%
% Requirements: MATLAB R2026b, Deep Learning Toolbox
%
% Copyright (c) 2026, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%
arguments
    inputSize (1,2) double
    coefMask = []
end
import tansacnet.salsun.*
lg = fcn_createsalsunlgraph2d([], ...
    'InputSize',inputSize, ...
    'Stride',[4 4], ...
    'OverlappingFactor',[3 3], ...
    'NumberOfVanishingMoments',true, ...
    'NumberOfNeighborBlocks',[3 3], ...
    'NumberOfResidualBlocks',3, ...
    'Width',2, ...
    'Mode','Whole', ...
    'ThetaMode','Reuse', ...
    'Device','cpu');
if ~isempty(coefMask)
    lg = lg.disconnectLayers('Lv1_AcOut','Lv1_AcIn');
    lg = lg.addLayers(maskLayer('Name','Lv1_AcMask', ...
        'Mask',coefMask(2:end),'NumberOfChannels',numel(coefMask)-1));
    lg = lg.connectLayers('Lv1_AcOut','Lv1_AcMask');
    lg = lg.connectLayers('Lv1_AcMask','Lv1_AcIn');
end
net = dlnetwork(lg);
net = dlupdate(@(w) w + 0.1*randn(size(w),'like',w),net);
end
