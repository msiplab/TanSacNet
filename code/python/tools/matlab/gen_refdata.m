function gen_refdata(varargin)
%GEN_REFDATA Generate reference data for the PyTorch port of tansacnet.lsun
%
%   gen_refdata() writes MAT-files to code/python/refdata, which
%   torch_tansacnet/test_lsunMatlabRefdata.py compares with the PyTorch
%   implementation:
%     layers/   : lsunIntermediateRotation2dLayer with BoundaryOperation
%                 'Circular' and 'Termination' (outputs and gradients)
%     networks/ : the analyzer and synthesizer of fcn_createlsunlgraph2d
%                 with random angles
%   and their 3-D counterparts (lsunBlockDct3dLayer,
%   lsunIntermediateRotation3dLayer and fcn_createlsunlgraph3d) and 1-D
%   counterparts (lsunBlockDct1dLayer, lsunIntermediateRotation1dLayer and
%   fcn_createlsunlgraph1d).
%
%   gen_refdata('OutDir',folder) writes to the given folder.
%
%   gen_refdata('Dimensions',3) generates only the 3-D data (default:
%   [1 2 3]). The data are reproducible, since the random numbers are seeded
%   by the case names.
%
%   Arrays are saved in the MATLAB layout. The angles and mus are those of
%   the MATLAB layers (nAngles x nBlocks with the blocks in column-major
%   order); the Python side converts them.
%
%   Run in the folder 'code' after setpath.
%
% Requirements: MATLAB R2022a or later, Deep Learning Toolbox
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

p = inputParser;
here = fileparts(mfilename('fullpath'));
addParameter(p,'OutDir',fullfile(here,'..','..','refdata'))
addParameter(p,'Dimensions',[1 2 3])
parse(p,varargin{:})
outDir = char(p.Results.OutDir);
dims = p.Results.Dimensions;

if any(dims == 1)
    genlayers1d_(fullfile(outDir,'layers'));
    gennetworks1d_(fullfile(outDir,'networks'));
end
if any(dims == 2)
    genlayers_(fullfile(outDir,'layers'));
    gennetworks_(fullfile(outDir,'networks'));
end
if any(dims == 3)
    genlayers3d_(fullfile(outDir,'layers'));
    gennetworks3d_(fullfile(outDir,'networks'));
end
end

function genlayers_(folder)
if ~isfolder(folder), mkdir(folder); end
nSamples = 2;
nBlocks = [4 5]; % non-square so that the directions cannot be confused
bopts = {{'Circular',''},{'Termination','Vertical'},{'Termination','Horizontal'}};
for stride = {[2 2],[4 4]}
    for mode = {'Analysis','Synthesis'}
        for iBo = 1:numel(bopts)
            [bo,dir] = bopts{iBo}{:};
            name = sprintf('IntermediateRotation2d_str%s_%s_%s%s',...
                strrep(num2str(stride{1}),' ',''),lower(mode{1}),lower(bo),...
                ifelse_(isempty(dir),'',['_' lower(dir)]));
            rng(seed_(name),'twister');
            layer = tansacnet.lsun.lsunIntermediateRotation2dLayer(...
                'Stride',stride{1},'NumberOfBlocks',nBlocks,'Mode',mode{1},...
                'BoundaryOperation',bo,'Direction',dir,'Device','cpu');
            nAngles = size(layer.Angles,1);
            pa = floor(prod(stride{1})/2);
            angles = pi*(2*rand(nAngles,prod(nBlocks))-1);
            mus = sign(randn(pa,prod(nBlocks)));
            mus(mus==0) = 1;
            layer.Angles = angles;
            layer.Mus = mus;
            X = randn([prod(stride{1}) nBlocks nSamples]);
            Z = layer.predict(X);
            dLdZ = randn(size(Z));
            [dLdX,dLdW] = layer.backward(X,[],dLdZ,[]);
            s = struct('meta_stride',stride{1},'meta_mode',mode{1},...
                'meta_boundary_operation',bo,'meta_direction',dir,...
                'meta_number_of_blocks',nBlocks,...
                'x',X,'z',Z,'angles',angles,'mus',mus,...
                'dldz',dLdZ,'dldx',dLdX,'dldw',dLdW);
            save(fullfile(folder,[name '.mat']),'-struct','s','-v7');
            fprintf('  %s\n',name);
        end
    end
end
end

function gennetworks_(folder)
if ~isfolder(folder), mkdir(folder); end
import tansacnet.lsun.*
nSamples = 2;
% stride, overlapping factor, input size (non-square)
combos = {
    [2 2], [3 3], [12 16]
    [2 2], [5 3], [12 16]
    [4 4], [3 5], [24 32]
    };
for iCombo = 1:size(combos,1)
    [stride,ovlp,szIn] = combos{iCombo,:};
    for bo = {'Circular','Termination'}
        name = sprintf('Lsun2d_str%d%d_ovlp%d%d_sz%dx%d_%s',stride,ovlp,szIn,lower(bo{1}));
        rng(seed_(name),'twister');
        args = {'InputSize',szIn,'Stride',stride,'OverlappingFactor',ovlp,...
            'NumberOfLevels',1,'NumberOfVanishingMoments',[1 1],...
            'BoundaryOperation',bo{1},'Device','cpu','DType','double'};
        analysislgraph = fcn_createlsunlgraph2d([],args{:},'Mode','Analyzer');
        synthesislgraph = fcn_createlsunlgraph2d([],args{:},'Mode','Synthesizer'); %#ok<NASGU> used in evalc
        % Random angles and mus of every rotation layer
        for iLayer = 1:numel(analysislgraph.Layers)
            layer = analysislgraph.Layers(iLayer);
            if isprop(layer,'Angles')
                layer.Angles = pi*(2*rand(size(layer.Angles))-1);
                if ~isa(layer,'tansacnet.lsun.lsunInitialRotation2dLayer')
                    mus = sign(randn(size(layer.Mus)));
                    mus(mus==0) = 1;
                    layer.Mus = mus;
                end
                analysislgraph = analysislgraph.replaceLayer(layer.Name,layer);
            end
        end
        [~,synthesislgraph] = evalc('fcn_cpparamsana2syn(synthesislgraph,analysislgraph)');
        % dlnetwork casts the learnables to single; cast them back to double
        analysisnet = dlupdate(@double,dlnetwork(analysislgraph));
        synthesisnet = dlupdate(@double,dlnetwork(synthesislgraph));
        X = randn([szIn 1 nSamples]);
        c = cell(1,2);
        [c{1:2}] = analysisnet.predict(dlarray(X,'SSCB')); % {AC, DC}
        % Synthesis of random coefficients (not of the analysis output, so
        % that a wrong synthesizer cannot pass by the perfect reconstruction)
        sAc = randn(size(c{1}));
        sDc = randn(size(c{2}));
        Y = synthesisnet.predict(dlarray(sAc,'SSCB'),dlarray(sDc,'SSCB'));
        s = struct('meta_stride',stride,'meta_overlapping_factor',ovlp,...
            'meta_input_size',szIn,'meta_boundary_operation',bo{1},...
            'x',X,'ac',double(extractdata(c{1})),'dc',double(extractdata(c{2})),...
            's_ac',sAc,'s_dc',sDc,'y',double(extractdata(Y)));
        % Parameters of the analyzer as stored in the network (in double)
        alayers = layerGraph(analysisnet).Layers;
        names = {};
        for iLayer = 1:numel(alayers)
            layer = alayers(iLayer);
            if isprop(layer,'Angles')
                key = matlab.lang.makeValidName(layer.Name);
                s.(['angles_' key]) = double(layer.Angles);
                s.(['mus_' key]) = double(layer.Mus);
                names{end+1} = layer.Name; %#ok<AGROW>
            end
        end
        s.meta_layer_names = strjoin(names,',');
        save(fullfile(folder,[name '.mat']),'-struct','s','-v7');
        fprintf('  %s\n',name);
    end
end
end

function genlayers3d_(folder)
if ~isfolder(folder), mkdir(folder); end
nSamples = 2;
nBlocks = [2 3 4]; % non-cubic so that the directions cannot be confused
% Block DCT
for stride = {[2 2 2],[1 2 2],[2 4 2]}
    name = sprintf('BlockDct3d_str%s',strrep(num2str(stride{1}),' ',''));
    rng(seed_(name),'twister');
    layer = tansacnet.lsun.lsunBlockDct3dLayer('Stride',stride{1});
    X = randn([nBlocks.*stride{1} 1 nSamples]);
    Z = layer.predict(X);
    s = struct('meta_stride',stride{1},'x',X,'z',Z);
    save(fullfile(folder,[name '.mat']),'-struct','s','-v7');
    fprintf('  %s\n',name);
end
% Intermediate rotation
bopts = {{'Circular',''},{'Termination','Vertical'},...
    {'Termination','Horizontal'},{'Termination','Depth'}};
for stride = {[2 2 2],[1 2 2]}
    for mode = {'Analysis','Synthesis'}
        for iBo = 1:numel(bopts)
            [bo,dir] = bopts{iBo}{:};
            name = sprintf('IntermediateRotation3d_str%s_%s_%s%s',...
                strrep(num2str(stride{1}),' ',''),lower(mode{1}),lower(bo),...
                ifelse_(isempty(dir),'',['_' lower(dir)]));
            rng(seed_(name),'twister');
            layer = tansacnet.lsun.lsunIntermediateRotation3dLayer(...
                'Stride',stride{1},'NumberOfBlocks',nBlocks,'Mode',mode{1},...
                'BoundaryOperation',bo,'Direction',dir,'Device','cpu');
            nAngles = size(layer.Angles,1);
            pa = floor(prod(stride{1})/2);
            angles = pi*(2*rand(nAngles,prod(nBlocks))-1);
            mus = sign(randn(pa,prod(nBlocks)));
            mus(mus==0) = 1;
            layer.Angles = angles;
            layer.Mus = mus;
            X = randn([prod(stride{1}) nBlocks nSamples]);
            Z = layer.predict(X);
            dLdZ = randn(size(Z));
            [dLdX,dLdW] = layer.backward(X,[],dLdZ,[]);
            s = struct('meta_stride',stride{1},'meta_mode',mode{1},...
                'meta_boundary_operation',bo,'meta_direction',dir,...
                'meta_number_of_blocks',nBlocks,...
                'x',X,'z',Z,'angles',angles,'mus',mus,...
                'dldz',dLdZ,'dldx',dLdX,'dldw',dLdW);
            save(fullfile(folder,[name '.mat']),'-struct','s','-v7');
            fprintf('  %s\n',name);
        end
    end
end
end

function gennetworks3d_(folder)
if ~isfolder(folder), mkdir(folder); end
import tansacnet.lsun.*
nSamples = 2;
% stride, overlapping factor, input size (non-cubic), number of levels
combos = {
    [2 2 2], [3 3 3], [ 4  6  8], 1
    [2 2 2], [1 3 5], [ 4  6  8], 1
    [1 2 2], [5 3 1], [ 3  4  6], 1
    [2 2 2], [3 3 3], [ 8 12 16], 2
    };
for iCombo = 1:size(combos,1)
    [stride,ovlp,szIn,nLevels] = combos{iCombo,:};
    for bo = {'Circular','Termination'}
        name = sprintf('Lsun3d_str%d%d%d_ovlp%d%d%d_sz%dx%dx%d_lv%d_%s',...
            stride,ovlp,szIn,nLevels,lower(bo{1}));
        rng(seed_(name),'twister');
        args = {'InputSize',szIn,'Stride',stride,'OverlappingFactor',ovlp,...
            'NumberOfLevels',nLevels,'NumberOfVanishingMoments',[1 1],...
            'BoundaryOperation',bo{1},'Device','cpu','DType','double'};
        analysislgraph = fcn_createlsunlgraph3d([],args{:},'Mode','Analyzer');
        synthesislgraph = fcn_createlsunlgraph3d([],args{:},'Mode','Synthesizer'); %#ok<NASGU> used in evalc
        % Random angles and mus of every rotation layer
        for iLayer = 1:numel(analysislgraph.Layers)
            layer = analysislgraph.Layers(iLayer);
            if isprop(layer,'Angles')
                layer.Angles = pi*(2*rand(size(layer.Angles))-1);
                if ~isa(layer,'tansacnet.lsun.lsunInitialRotation3dLayer')
                    mus = sign(randn(size(layer.Mus)));
                    mus(mus==0) = 1;
                    layer.Mus = mus;
                end
                analysislgraph = analysislgraph.replaceLayer(layer.Name,layer);
            end
        end
        [~,synthesislgraph] = evalc('fcn_cpparamsana2syn(synthesislgraph,analysislgraph)');
        % dlnetwork casts the learnables to single; cast them back to double
        analysisnet = dlupdate(@double,dlnetwork(analysislgraph));
        synthesisnet = dlupdate(@double,dlnetwork(synthesislgraph));
        X = randn([szIn 1 nSamples]);
        % { AC of level 1, ..., AC of level nLevels, DC of level nLevels }
        c = cell(1,nLevels+1);
        [c{:}] = analysisnet.predict(dlarray(X,'SSSCB'));
        % Synthesis of random coefficients (not of the analysis output, so
        % that a wrong synthesizer cannot pass by the perfect reconstruction)
        sc = cellfun(@(x) randn(size(x)),c,'UniformOutput',false);
        dlsc = cellfun(@(x) dlarray(x,'SSSCB'),sc,'UniformOutput',false);
        Y = synthesisnet.predict(dlsc{:});
        s = struct('meta_stride',stride,'meta_overlapping_factor',ovlp,...
            'meta_input_size',szIn,'meta_boundary_operation',bo{1},...
            'meta_number_of_levels',nLevels,...
            'meta_output_names',strjoin(analysisnet.OutputNames,','),...
            'meta_input_names',strjoin(synthesisnet.InputNames,','),...
            'x',X,'y',double(extractdata(Y)));
        for iLv = 1:nLevels
            s.(sprintf('ac%d',iLv)) = double(extractdata(c{iLv}));
            s.(sprintf('s_ac%d',iLv)) = sc{iLv};
        end
        s.dc = double(extractdata(c{end}));
        s.s_dc = sc{end};
        % Parameters of the analyzer as stored in the network (in double)
        alayers = layerGraph(analysisnet).Layers;
        names = {};
        for iLayer = 1:numel(alayers)
            layer = alayers(iLayer);
            if isprop(layer,'Angles')
                key = matlab.lang.makeValidName(layer.Name);
                s.(['angles_' key]) = double(layer.Angles);
                s.(['mus_' key]) = double(layer.Mus);
                names{end+1} = layer.Name; %#ok<AGROW>
            end
        end
        s.meta_layer_names = strjoin(names,',');
        save(fullfile(folder,[name '.mat']),'-struct','s','-v7');
        fprintf('  %s\n',name);
    end
end
end

function genlayers1d_(folder)
if ~isfolder(folder), mkdir(folder); end
nSamples = 2;
nBlocks = 5;
% Block DCT
for stride = {2,4,8}
    name = sprintf('BlockDct1d_str%d',stride{1});
    rng(seed_(name),'twister');
    layer = tansacnet.lsun.lsunBlockDct1dLayer('Stride',stride{1});
    X = randn([1 nBlocks*stride{1} 1 nSamples]);
    Z = layer.predict(X);
    s = struct('meta_stride',stride{1},'x',X,'z',Z);
    save(fullfile(folder,[name '.mat']),'-struct','s','-v7');
    fprintf('  %s\n',name);
end
% Intermediate rotation
for stride = {4,8}
    for mode = {'Analysis','Synthesis'}
        for bo = {'Circular','Termination'}
            name = sprintf('IntermediateRotation1d_str%d_%s_%s',...
                stride{1},lower(mode{1}),lower(bo{1}));
            rng(seed_(name),'twister');
            layer = tansacnet.lsun.lsunIntermediateRotation1dLayer(...
                'Stride',stride{1},'NumberOfBlocks',nBlocks,'Mode',mode{1},...
                'BoundaryOperation',bo{1},'Device','cpu');
            nAngles = size(layer.Angles,1);
            pa = stride{1}/2;
            angles = pi*(2*rand(nAngles,nBlocks)-1);
            mus = sign(randn(pa,nBlocks));
            mus(mus==0) = 1;
            layer.Angles = angles;
            layer.Mus = mus;
            X = randn([stride{1} 1 nBlocks nSamples]);
            Z = layer.predict(X);
            dLdZ = randn(size(Z));
            [dLdX,dLdW] = layer.backward(X,[],dLdZ,[]);
            s = struct('meta_stride',stride{1},'meta_mode',mode{1},...
                'meta_boundary_operation',bo{1},...
                'meta_number_of_blocks',nBlocks,...
                'x',X,'z',Z,'angles',angles,'mus',mus,...
                'dldz',dLdZ,'dldx',dLdX,'dldw',dLdW);
            save(fullfile(folder,[name '.mat']),'-struct','s','-v7');
            fprintf('  %s\n',name);
        end
    end
end
end

function gennetworks1d_(folder)
if ~isfolder(folder), mkdir(folder); end
import tansacnet.lsun.*
nSamples = 2;
% stride, overlapping factor, input size, number of levels
% (stride 2 has no angles, i.e., (ps,pa) = (1,1))
combos = {
    4, 3, 24, 1
    8, 5, 48, 1
    2, 3, 12, 1
    4, 5, 48, 2
    };
for iCombo = 1:size(combos,1)
    [stride,ovlp,szIn,nLevels] = combos{iCombo,:};
    for bo = {'Circular','Termination'}
        name = sprintf('Lsun1d_str%d_ovlp%d_sz%d_lv%d_%s',...
            stride,ovlp,szIn,nLevels,lower(bo{1}));
        rng(seed_(name),'twister');
        args = {'InputSize',szIn,'Stride',stride,'OverlappingFactor',ovlp,...
            'NumberOfLevels',nLevels,'NumberOfVanishingMoments',[1 1],...
            'BoundaryOperation',bo{1},'Device','cpu','DType','double'};
        analysislgraph = fcn_createlsunlgraph1d([],args{:},'Mode','Analyzer');
        synthesislgraph = fcn_createlsunlgraph1d([],args{:},'Mode','Synthesizer'); %#ok<NASGU> used in evalc
        % Random angles and mus of every rotation layer
        for iLayer = 1:numel(analysislgraph.Layers)
            layer = analysislgraph.Layers(iLayer);
            if isprop(layer,'Angles')
                layer.Angles = pi*(2*rand(size(layer.Angles))-1);
                if ~isa(layer,'tansacnet.lsun.lsunInitialRotation1dLayer')
                    mus = sign(randn(size(layer.Mus)));
                    mus(mus==0) = 1;
                    layer.Mus = mus;
                end
                analysislgraph = analysislgraph.replaceLayer(layer.Name,layer);
            end
        end
        [~,synthesislgraph] = evalc('fcn_cpparamsana2syn(synthesislgraph,analysislgraph)');
        % dlnetwork casts the learnables to single; cast them back to double
        analysisnet = dlupdate(@double,dlnetwork(analysislgraph));
        synthesisnet = dlupdate(@double,dlnetwork(synthesislgraph));
        X = randn([1 szIn 1 nSamples]);
        % { AC of level 1, ..., AC of level nLevels, DC of level nLevels }
        c = cell(1,nLevels+1);
        [c{:}] = analysisnet.predict(dlarray(X,'SSCB'));
        % Synthesis of random coefficients (not of the analysis output, so
        % that a wrong synthesizer cannot pass by the perfect reconstruction)
        sc = cellfun(@(x) randn(size(x)),c,'UniformOutput',false);
        dlsc = cellfun(@(x) dlarray(x,'SSCB'),sc,'UniformOutput',false);
        Y = synthesisnet.predict(dlsc{:});
        s = struct('meta_stride',stride,'meta_overlapping_factor',ovlp,...
            'meta_input_size',szIn,'meta_boundary_operation',bo{1},...
            'meta_number_of_levels',nLevels,...
            'meta_output_names',strjoin(analysisnet.OutputNames,','),...
            'meta_input_names',strjoin(synthesisnet.InputNames,','),...
            'x',X,'y',double(extractdata(Y)));
        for iLv = 1:nLevels
            s.(sprintf('ac%d',iLv)) = double(extractdata(c{iLv}));
            s.(sprintf('s_ac%d',iLv)) = sc{iLv};
        end
        s.dc = double(extractdata(c{end}));
        s.s_dc = sc{end};
        % Parameters of the analyzer as stored in the network (in double)
        alayers = layerGraph(analysisnet).Layers;
        names = {};
        for iLayer = 1:numel(alayers)
            layer = alayers(iLayer);
            if isprop(layer,'Angles')
                key = matlab.lang.makeValidName(layer.Name);
                s.(['angles_' key]) = double(layer.Angles);
                s.(['mus_' key]) = double(layer.Mus);
                names{end+1} = layer.Name; %#ok<AGROW>
            end
        end
        s.meta_layer_names = strjoin(names,',');
        save(fullfile(folder,[name '.mat']),'-struct','s','-v7');
        fprintf('  %s\n',name);
    end
end
end

function v = seed_(name)
% Seed derived from the case name
v = mod(sum(double(name).*(1:numel(name))),2^31-1);
end

function v = ifelse_(cond,a,b)
if cond, v = a; else, v = b; end
end
