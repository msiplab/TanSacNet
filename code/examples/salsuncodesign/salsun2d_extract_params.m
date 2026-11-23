function params = salsun2d_extract_params(net,options)
%SALSUN2D_EXTRACT_PARAMS Extract SA-LSUN parameters from a dlnetwork
%
%   params = salsun2d_extract_params(net) returns a struct holding every
%   parameter that salsun2d_infer needs, as plain single arrays.
%
%   net must be a dlnetwork built by fcn_createsalsunlgraph2d with
%   Mode 'Whole' and ThetaMode 'Reuse' (one level, one component). An
%   optional maskLayer named 'Lv1_AcMask' between 'Lv1_AcOut' and
%   'Lv1_AcIn' is taken as the coefficient mask, as in main_salsun2d.m.
%
%   Fields of params:
%     Stride      - block size [My Mx]
%     Cvh         - prod(Stride) x prod(Stride) block DCT matrix
%     Mask        - prod(Stride) x 1 coefficient mask (DC first)
%     V0          - initial rotation: MusW, MusU (one column when the
%                   signs are the same for all blocks), Estimator
%     V0t         - final rotation (synthesis): MusW, MusU
%     Stages      - intermediate stages in analysis order, each with
%                   Shift, Target, Mus, Estimator (analysis side) and
%                   SynShift, SynTarget, SynMus (synthesis side)
%     BoundaryOperation - 'Circular' or 'Termination' (see
%                   fcn_createsalsunlgraph2d); with 'Termination' every
%                   stage has Termination, the direction of its atom
%                   extension when its rotation is -I at the blocks of a
%                   boundary ('' for the other stages)
%   The estimators of the default estimator function
%   (fcn_createparamestimator2dlgraph) have Channels, Neighbor, Padding
%   ('circular' or 'zero'), Epsilon, ResBlocks (Gamma, Beta, W1, B1, W2,
%   B2; Gamma and Beta empty without layer normalization) and Wo, Bo,
%   NumberOfZeroPadAngles. Those of fcn_createconstestimatorlgraph have
%   Type 'constant', Angles (with the zero-padded leading angles) and
%   NumberOfZeroPadAngles.
%
%   params = salsun2d_extract_params(net,EstimatorExtractor=f) extracts
%   the estimators of another estimator function (plug-in estimators):
%   est = f(getLayer,prefix) with getLayer(name) the layer of net named
%   name (empty if none) and prefix the prefix of the estimator (its
%   layers [prefix 'Ext'] and [prefix 'Theta']). est must have a field
%   Function, a handle theta = Function(Y,est,q) that salsun2d_infer calls
%   with the coefficients Y (prod(Stride) x nRows x nCols) before the
%   rotation and the quantizer q, and that returns the angles (nAngles x
%   nRows*nCols, blocks in column-major order, the zero-padded leading
%   angles included).
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
    net dlnetwork
    options.EstimatorExtractor = []
end
extractor = options.EstimatorExtractor;

prefix = 'Lv1_Cmp1_';
names = {net.Layers.Name};
getLayer = @(name) net.Layers(strcmp(names,name));
assert(any(strcmp(names,[prefix 'V0'])) && ~any(startsWith(names,'Lv2_')), ...
    'salsun2d_extract_params:unsupported', ...
    'Only one level and one component are supported.')

% Block DCT
e0 = getLayer('Lv1_E0');
params.Stride = e0.Stride;
nDec = prod(params.Stride);
% The DCT matrix is private, so apply the layer to unit impulses
params.Cvh = zeros(nDec,'single');
for k = 1:nDec
    impulse = zeros(params.Stride,'single');
    impulse(k) = 1;
    params.Cvh(:,k) = reshape(e0.predict(impulse),nDec,1);
end

% Coefficient mask (DC is always kept)
params.Mask = ones(nDec,1,'single');
if any(strcmp(names,'Lv1_AcMask'))
    m = getLayer('Lv1_AcMask');
    params.Mask(2:end) = single(m.Mask(:));
end

% Initial and final rotations
ps = nDec/2;
v0 = getLayer([prefix 'V0']);
[params.V0.MusW,params.V0.MusU] = splitMus(v0.Mus,ps);
params.V0.Estimator = extractEstimator(getLayer,[prefix 'V0_'],extractor);
v0t = getLayer([prefix 'V0~']);
[params.V0t.MusW,params.V0t.MusU] = splitMus(v0t.Mus,ps);

% Intermediate stages: follow the analysis chain from V0 to the separation
src = string(net.Connections.Source);
dst = extractBefore(string(net.Connections.Destination) + "/","/");
% Data-path successor: skip the theta inputs and the estimator inputs (and
% the termination of the state for an estimator)
next = @(name) dst(src == name & ...
    ~endsWith(string(net.Connections.Destination),"/theta") & ~endsWith(dst,"Ext") ...
    & ~endsWith(dst,"TrmX"));
stages = struct('Shift',{},'Target',{},'Mus',{},'Estimator',{}, ...
    'SynShift',{},'SynTarget',{},'SynMus',{},'Termination',{});
cur = string([prefix 'V0']);
while true
    qName = next(cur);
    qName = qName(1);
    if qName == string([prefix 'Sp'])
        break
    end
    vName = next(qName);
    vName = vName(1);
    q = getLayer(char(qName));
    v = getLayer(char(vName));
    qt = getLayer(char(qName + "~"));
    vt = getLayer(char(extractBefore(vName,strlength(vName)) + "~"));
    stage.Shift = directionToShift(q.Direction);
    stage.Target = char(q.TargetChannels);
    stage.Mus = collapseMus(v.Mus);
    stage.Estimator = extractEstimator(getLayer,char(vName),extractor);
    stage.SynShift = directionToShift(qt.Direction);
    stage.SynTarget = char(qt.TargetChannels);
    stage.SynMus = collapseMus(vt.Mus);
    % boundary termination of the angles (fcn_createsalsunlgraph2d)
    trm = getLayer(char(vName + "Trm"));
    if isempty(trm)
        stage.Termination = '';
    else
        stage.Termination = char(trm.Direction);
    end
    stages(end+1) = stage; %#ok<AGROW>
    cur = vName;
end
params.Stages = stages;
if any(arrayfun(@(st) ~isempty(st.Termination),stages))
    params.BoundaryOperation = 'Termination';
else
    params.BoundaryOperation = 'Circular';
end
end

function [muW,muU] = splitMus(mus,ps)
mus = collapseMus(mus);
muW = mus(1:ps,:);
muU = mus(ps+1:end,:);
end

function mus = collapseMus(mus)
% The layers store the sign flips per block (nChannels x nBlocks). When
% all blocks have the same signs, keep one column, so that the
% parameters do not depend on the image size (bands, tiles, the FPGA).
mus = single(gather(mus));
if size(mus,2) > 1 && all(mus == mus(:,1),'all')
    mus = mus(:,1);
end
end

function shift = directionToShift(direction)
% [vertical horizontal] circular shift of the block grid
switch char(direction)
    case 'Right', shift = [0 1];
    case 'Left',  shift = [0 -1];
    case 'Down',  shift = [1 0];
    case 'Up',    shift = [-1 0];
    otherwise
        error('salsun2d_extract_params:direction','Unknown direction %s',direction)
end
end

function est = extractEstimator(getLayer,prefix,extractor)
toSingle = @(v) single(gather(extractdata(dlarray(v))));
if ~isempty(extractor)                    % a plug-in estimator
    est = extractor(getLayer,prefix);
    assert(isfield(est,'Function') && isa(est.Function,'function_handle'), ...
        'salsun2d_extract_params:extractor', ...
        'The estimator of %s from EstimatorExtractor has no function handle Function.',prefix)
    return
end
outl = getLayer([prefix 'Theta']);
if isa(outl,'tansacnet.salsun.salsunConstantAngleLayer')
    % constant angles per block (fcn_createconstestimatorlgraph)
    est.Type = 'constant';
    est.NumberOfZeroPadAngles = double(outl.NumberOfZeroPadAngles);
    est.Angles = [zeros(est.NumberOfZeroPadAngles,size(outl.Angles,2),'single'); toSingle(outl.Angles)];
    return
end
std_ = getLayer([prefix 'Std']);
assert(~isempty(std_) && isa(getLayer([prefix 'Ext']),'tansacnet.salsun.salsunLocalStateExtraction2dLayer'), ...
    'salsun2d_extract_params:estimator', ...
    ['The estimator %s is not one of fcn_createparamestimator2dlgraph or ' ...
    'fcn_createconstestimatorlgraph: give EstimatorExtractor.'],prefix)
ext = getLayer([prefix 'Ext']);
est.Channels = double(ext.Channels(:));
est.Neighbor = double(ext.NumberOfNeighborBlocks);
% neighbor blocks beyond the boundary: 'circular' or 'zero'
est.Padding = 'circular';
if isprop(ext,'Padding')
    est.Padding = char(ext.Padding);
end
est.Epsilon = single(std_.Epsilon);
% Standardization scope: 'image', 'neighborhood' (statistics of every
% channel over the neighbor blocks of a block), 'layer' (layer
% normalization of the state of a block, with StdGamma and StdBeta) or 'none'
if isprop(std_,'Scope')
    est.StdScope = char(std_.Scope);
else
    est.StdScope = 'image';
end
% Activation of the residual blocks and mapping of the output layer
blk1 = getLayer([prefix 'ResBlk1']);
est.Activation = 'gelu';
if ~isempty(blk1) && isprop(blk1,'Activation')
    est.Activation = char(blk1.Activation);
end
est.OutputMapping = 'linear';
if isprop(outl,'OutputMapping')
    est.OutputMapping = char(outl.OutputMapping);
end
if strcmp(est.StdScope,'layer')
    est.StdGamma = toSingle(std_.Gamma);
    est.StdBeta = toSingle(std_.Beta);
end
% Optional bottleneck projection of the standardized state (empty if none)
proj = getLayer([prefix 'Proj']);
if isempty(proj)
    est.Wp = zeros(0,0,'single');
    est.Bp = zeros(0,1,'single');
else
    est.Wp = toSingle(proj.Wp);
    est.Bp = toSingle(proj.Bp);
end
iBlk = 1;
resBlocks = struct('Gamma',{},'Beta',{},'W1',{},'B1',{},'W2',{},'B2',{});
while true
    blk = getLayer(sprintf('%sResBlk%d',prefix,iBlk));
    if isempty(blk)
        break
    end
    if isprop(blk,'LayerNorm') && ~blk.LayerNorm   % no layer normalization: empty
        resBlocks(iBlk).Gamma = zeros(0,1,'single');
        resBlocks(iBlk).Beta = zeros(0,1,'single');
    else
        resBlocks(iBlk).Gamma = toSingle(blk.Gamma);
        resBlocks(iBlk).Beta = toSingle(blk.Beta);
    end
    resBlocks(iBlk).W1 = toSingle(blk.W1);
    resBlocks(iBlk).B1 = toSingle(blk.B1);
    resBlocks(iBlk).W2 = toSingle(blk.W2);
    resBlocks(iBlk).B2 = toSingle(blk.B2);
    iBlk = iBlk + 1;
end
est.ResBlocks = resBlocks;
out = getLayer([prefix 'Theta']);
est.Wo = toSingle(out.Wo);
est.Bo = toSingle(out.Bo);
est.NumberOfZeroPadAngles = double(out.NumberOfZeroPadAngles);
end
