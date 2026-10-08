function params = salsun2d_extract_params(net)
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
%   Each Estimator has Channels, Neighbor, Epsilon, ResBlocks (Gamma,
%   Beta, W1, B1, W2, B2) and Wo, Bo, NumberOfZeroPadAngles.
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
end

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
params.V0.Estimator = extractEstimator(getLayer,[prefix 'V0_']);
v0t = getLayer([prefix 'V0~']);
[params.V0t.MusW,params.V0t.MusU] = splitMus(v0t.Mus,ps);

% Intermediate stages: follow the analysis chain from V0 to the separation
src = string(net.Connections.Source);
dst = extractBefore(string(net.Connections.Destination) + "/","/");
% Data-path successor: skip the theta inputs and the estimator inputs
next = @(name) dst(src == name & ...
    ~endsWith(string(net.Connections.Destination),"/theta") & ~endsWith(dst,"Ext"));
stages = struct('Shift',{},'Target',{},'Mus',{},'Estimator',{}, ...
    'SynShift',{},'SynTarget',{},'SynMus',{});
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
    stage.Estimator = extractEstimator(getLayer,char(vName));
    stage.SynShift = directionToShift(qt.Direction);
    stage.SynTarget = char(qt.TargetChannels);
    stage.SynMus = collapseMus(vt.Mus);
    stages(end+1) = stage; %#ok<AGROW>
    cur = vName;
end
params.Stages = stages;
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

function est = extractEstimator(getLayer,prefix)
toSingle = @(v) single(gather(extractdata(dlarray(v))));
ext = getLayer([prefix 'Ext']);
est.Channels = double(ext.Channels(:));
est.Neighbor = double(ext.NumberOfNeighborBlocks);
std_ = getLayer([prefix 'Std']);
est.Epsilon = single(std_.Epsilon);
iBlk = 1;
resBlocks = struct('Gamma',{},'Beta',{},'W1',{},'B1',{},'W2',{},'B2',{});
while true
    blk = getLayer(sprintf('%sResBlk%d',prefix,iBlk));
    if isempty(blk)
        break
    end
    resBlocks(iBlk).Gamma = toSingle(blk.Gamma);
    resBlocks(iBlk).Beta = toSingle(blk.Beta);
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
