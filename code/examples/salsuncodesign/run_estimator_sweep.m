function run_estimator_sweep(configs,seeds,options)
%RUN_ESTIMATOR_SWEEP Train SA-LSUN with smaller angle estimators
%
%   run_estimator_sweep(configs,seeds) trains the network of
%   train_salsun2d_variants (base field 'none', image statistics) for each
%   row [numResidualBlocks width], [numResidualBlocks width nv nh]
%   (neighbor blocks, default 3 x 3) or [numResidualBlocks width nv nh p]
%   (bottleneck size p, default 0: none) of configs and each seed, and
%   saves the results to
%   OutputDir/R<r>_W<w>[_N<nv>x<nh>][_P<p>]/seed<s>/none.mat. Existing results are
%   skipped, so the sweep can be split over machines and resumed.
%   summarize_estimator_sweep collects them.
%
% Requirements: MATLAB R2026b, Deep Learning Toolbox, a GPU
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
    configs double
    seeds (1,:) double
    options.OutputDir = fullfile(fileparts(mfilename('fullpath')),'results','arch')
    options.MaxEpochs (1,1) double = 100
    options.MiniBatchSize (1,1) double = 8
end
for iSeed = 1:numel(seeds)
    for iCfg = 1:size(configs,1)
        r = configs(iCfg,1);
        w = configs(iCfg,2);
        if size(configs,2) >= 4
            nb = configs(iCfg,3:4);
        else
            nb = [3 3];
        end
        if size(configs,2) >= 5
            pb = configs(iCfg,5);
        else
            pb = 0;
        end
        name = sprintf('R%d_W%g',r,w);
        if ~isequal(nb,[3 3])
            name = sprintf('%s_N%dx%d',name,nb);
        end
        if pb > 0
            name = sprintf('%s_P%d',name,pb);
        end
        outDir = fullfile(options.OutputDir,name,sprintf('seed%d',seeds(iSeed)));
        if isfile(fullfile(outDir,'none.mat'))
            fprintf('skip %s\n',outDir);
            continue
        end
        fprintf('\n######## R=%d W=%g neighbors %dx%d bottleneck %d seed %d\n',r,w,nb,pb,seeds(iSeed));
        train_salsun2d_variants(Variants={'none'},StatsModes={'image'}, ...
            NumResidualBlocks=r,Width=w,NeighborBlocks=nb,BottleneckSize=pb,Seed=seeds(iSeed),OutputDir=outDir, ...
            MaxEpochs=options.MaxEpochs,MiniBatchSize=options.MiniBatchSize);
    end
end
end
