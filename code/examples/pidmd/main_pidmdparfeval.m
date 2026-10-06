%[text] # Material reproduction of piDMD via LSUN
%[text] 
%[text] References
%[text] - Baddoo Peter J., Herrmann Benjamin, McKeon Beverley J., Nathan Kutz J. and Brunton Steven L. 2023Physics-informed dynamic mode decompositionProc. R. Soc. A.4792022057620220576, [http://doi.org/10.1098/rspa.2022.0576](http://doi.org/10.1098/rspa.2022.0576)
%[text] - Brunton, S. L., & Kutz, J. N. (2022). *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control* (2nd ed.). Cambridge: Cambridge University Press.
%[text] - jitKutz JN, Brunton SL, Brunton BW, Proctor JL. 2016 *Dynamic mode decomposition: data-driven modeling of complex systems*. Philadelprehash toolboxcache
%[text] - hia, PA: SIAM. \
%%
clear
close all
ccd = pwd;
cd("../../")
setpath
cd(ccd)
setup
rng default;
%%
%[text] ## Source Data Preparation
nrows = 199;
ncols = 449;
load("../../../data/databook_DATA/DATA/VORTALL");
ut = reshape(VORTALL,nrows,ncols,[]);
save("../../../results/sourceflow","ut");
%%
%[text] ## TODO: Evaluate with PARFEVAL for multiple combinations of configuration parameters 
%[text] ## piDMD setup
methodlist{1} = [ false, "exact" ];
methodlist{2} = [ false, "orthogonal" ];
methodlist{3} = [ true, "orthogonal" ];
%[text] ## Parameter Setup
nModes4PODtrunc = 15;
varu = var(ut,0,'all');
nTrials = 10;
%[text] ### Variable parameters
nCoefs4LSUNtruncSet = 1:1:4;
snRatioSet = 0.1:0.1:0.5;
ovlpFactorSet = 1:2:5;
%%
%[text] ## Configurations
%[text] Noise energy set to snRaio\*100% of signal energy
iConfig = 1;
nConfigs = ... nTrials*...
    length(snRatioSet)*...
    (...
    2 + ... exact DMD, orthogonal DMD
    length(nCoefs4LSUNtruncSet)*length(ovlpFactorSet) ... orhogonal DMD w/ LSUN
    )
configs = cell(nConfigs,1);
%config_.iTrial = iTrial;
for snRatio = snRatioSet
    config.snRatio = snRatio;
    %
    varw = snRatio*varu;
    sigmaw = sqrt(varw);
    noisydatafile = "../../../results/noisyflow_"+replace(num2str(sigmaw,"%6.4f"),'.','_');

    for iTrial = 1:nTrials
        noisydatafile_itrial = noisydatafile+"_trial"+num2str(iTrial,"%02d");
        if ~exist(noisydatafile_itrial+".mat","file")
            disp("Create " + noisydatafile_itrial)
            vt = ut + sigmaw * randn(size(ut),'like',ut);
            save(noisydatafile_itrial,"vt");
        else
            disp(noisydatafile_itrial + " exists.")
        end
    end
    config.noisydatafile = noisydatafile;
    %
    for imethod = 1:length(methodlist)
        islsun = strcmp(methodlist{imethod}(1),"true");
        dmdmethod = methodlist{imethod}(2);
        config.islsun = islsun;
        config.dmdmethod = dmdmethod;
        %
        if islsun
            for ovlpFactor = ovlpFactorSet
                config.ovlpFactor = ovlpFactor;
                %
                for nCoefs4LSUNtrunc = nCoefs4LSUNtruncSet
                    config.nCoefs4LSUNtrunc = nCoefs4LSUNtrunc;
                    %
                    configs{iConfig} = config;
                    iConfig = iConfig + 1;
                end
            end
        else
            config.nCoefs4LSUNtrunc = [];
            config.ovlpFactor = [];
            %
            configs{iConfig} = config;
            iConfig = iConfig + 1;
        end
    end
end
%end
configs
%%
% Save configs
configsfile = "../../../results/parfevalconfigs";
save(configsfile,"configs");
%%
%[text] ## Par pool
%[text] ["クライアントからワーカー X への接続が切断されました" というエラーのトラブルシュートの方法はありますか？ - MATLAB Answers - MATLAB Central (mathworks.com)](https://jp.mathworks.com/matlabcentral/answers/1690595-x)
% nPools = 16;
%[text] [https://jp.mathworks.com/matlabcentral/answers/2097811-why-do-i-receive-the-error-your-mathworks-account-is-not-linked-to-a-matlab-parallel-server-license](https://jp.mathworks.com/matlabcentral/answers/2097811-why-do-i-receive-the-error-your-mathworks-account-is-not-linked-to-a-matlab-parallel-server-license)
%%
delete(gcp('nocreate'))
c = parcluster('Processes');
c.hSetPropertyNoCheck('RequiresOnlineLicensing',false);
pool = c.parpool();
for iTrial = 1:nTrials
    %pool = parpool('Processes',nPools) %,'SpmdEnabled',false,'IdleTimeout',120)
    %pool.addAttachedFiles(["../../+tansacnet/+lsun/" "../../+tansacnet/+utility/" "../../mexcodes/"])
    %
    fprintf("--- Trial: "+num2str(iTrial,"%02d")+" ---"+newline)
    F(1:nConfigs) = parallel.FevalFuture;
    %
    for iConfig = 1:nConfigs
        config = configs{iConfig};
        islsun = config.islsun;

        % Get default options
        [~,~,~,~,options] = fcn_pidmdvialsun([],[],[],islsun);

        % Conduct piDMD
        dmdmethod = config.dmdmethod;
        if islsun
            options.ovlpFactor = config.ovlpFactor*[1 1];
            options.nCoefs = config.nCoefs4LSUNtrunc;
            options.useGPU = false;
            options.outputEnvironment = 'cpu';
        end
        %disp(options)

        S = load(config.noisydatafile+"_trial"+num2str(iTrial,"%02d"),"vt");
        vt = S.vt;

        %[rt,rmse] = myfunc_(vt,ut,dmdmethod,nModes4PODtrunc,islsun,options);
        numout = 2;
        f = parfeval(pool,@myfunc_,numout,vt,ut,dmdmethod,nModes4PODtrunc,islsun,options);
        F(iConfig) = f;
        %myfunc_(vt,ut,dmdmethod,nModes4PODtrunc,islsun,options);

    end

    % Collect the results as they become available.
    % Build a waitbar to track progress
    h = waitbar(0,"Waiting for FevalFutures to complete...");
    for iConfig = 1:nConfigs
        % fetchNext blocks until next results are available.
        [completedIdx,rt,rmse] = fetchNext(F);

        % Read configulation
        config = configs{completedIdx};
        islsun = config.islsun;
        dmdmethod = config.dmdmethod;
        snRatio = config.snRatio;
        fprintf('Got result with index: %03d, LSUN: %d, MODE: %s, S/N: %4.2f, RMSE(end): %f\n', ...
            completedIdx,islsun,dmdmethod,snRatio,rmse(end));

        % Save results
        resultfile = "../../../results/result_config"+num2str(completedIdx,"%03d")+"_trial"+num2str(iTrial,"%02d");
        save(resultfile,"completedIdx","rt","rmse","config","iTrial");

        % Update waitbar
         waitbar(iConfig/nConfigs,h,sprintf("Latest completedIdx: %03d in trial: %02d", ...
            completedIdx,iTrial))
    end
    delete(h)
end
%%
% load results
%resultfile = result_config"+num2str(completedIdx,"%03d")+"_trial"+num2str(iTrial,"%02d");
%S = load(resultfile,"completedIdx","rt","rmse","config");
%S.completedIdx
%S.rt
%S.rmse
%S.config
%%
%[text] ## Function for evaluation
function [rt,rmse] = myfunc_(vt,ut,dmdmethod,nModes4PODtrunc,islsun,options)
%[dmdA, dmdVals, dmdVecs, dmdProjA,~,rt] = fcn_pidmdvialsun(vt,dmdmethod,nModes4PODtrunc,islsun,options);
[~,~,~,~,~,rt] = fcn_pidmdvialsun(vt,dmdmethod,nModes4PODtrunc,islsun,options);
rmse = squeeze(sqrt(mean((rt-ut).^2,[1 2])));
end
%[text] ### 

%[appendix]{"version":"1.0"}
%---
%[metadata:view]
%   data: {"layout":"onright","rightPanelPercent":29.3}
%---
