%[text] # Material reproduction of piDMD via LSUN
%[text] 
%[text] References
%[text] - Baddoo Peter J., Herrmann Benjamin, McKeon Beverley J., Nathan Kutz J. and Brunton Steven L. 2023Physics-informed dynamic mode decompositionProc. R. Soc. A.4792022057620220576, [http://doi.org/10.1098/rspa.2022.0576](http://doi.org/10.1098/rspa.2022.0576)
%[text] - Brunton, S. L., & Kutz, J. N. (2022). *Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control* (2nd ed.). Cambridge: Cambridge University Press.
%[text] - jitKutz JN, Brunton SL, Brunton BW, Proctor JL. 2016 *Dynamic mode decomposition: data-driven modeling of complex systems*. Philadelphia, PA: SIAM. \
%%
clear 
close all
setup
%%
isDisplayTraining = false; %[control:checkbox:9168]{"position":[21,26]}
isVisualize = true; %[control:checkbox:8307]{"position":[15,19]}
%%
%[text] ## Source Data Preparation
nrows = 199;
ncols = 449;
load("../../../data/databook_DATA/DATA/VORTALL");
ut = reshape(VORTALL,nrows,ncols,[]);
%%
%[text] ## Source Data Visualization
if isVisualize
    clims = 1.1*[min(ut,[],'all') max(ut,[],'all')];
    him = [];
    iFrame = 151;
    x = ut(:,:,iFrame);
    fcn_cylinderplot(him,x,clims);
    title("Source data (" + "$[\mathbf{u}]_{"+num2str(iFrame-1)+"}$)","Interpreter","latex")
    drawnow
    % exportgraphics
end
%%
%[text] ## TODO: Evaluate with PARFEVAL for multiple combinations of configuration parameters 
%[text] ## piDMD setup
methodlist{1} = [ false, "exact" ];
methodlist{2} = [ false, "orthogonal" ];
methodlist{3} = [ true, "orthogonal" ];
%[text] ## Parameter Setup
nModes4PODtrunc = 15;
nCoefs4LSUNtrunc = 2; % TODO: vary
snRatio = 0.2; % TODO: vary
%%
%[text] ## Noisy Data Preparation
%[text] Noise energy set to snRaio\*100% of signal energy
varu = var(ut,0,'all');
varw = snRatio*varu;
sigmaw = sqrt(varw);
vt = ut + sigmaw * randn(size(ut),'like',ut);
%%
%[text] ## Noisy Data Visualization
if isVisualize
    him = [];
    x = vt(:,:,iFrame);
    fcn_cylinderplot(him,x,clims);
    title("Noisy data  (" + "$[\mathbf{v}]_{"+num2str(iFrame-1)+"}$), S/N ratio:"+ num2str(100*snRatio,"%4.1f")+"\%","Interpreter","latex")
    drawnow
    % exportgraphics
    % save("../../../results/noisyflow_"+replace(num2str(sigmaw),'.','_'),"ut","vt")
end
%%
for imethod = 1:length(methodlist)
    islsun = strcmp(methodlist{imethod}(1),"true");
    disp("islsun: "+islsun)
    dmdmethod = methodlist{imethod}(2);
    disp(dmdmethod)

    % Get default options
    [~,~,~,~,options] = fcn_pidmdvialsun([],[],[],islsun);

    if isDisplayTraining && islsun
        figure
        lineLossTrain = animatedline('Color',[0.85 0.325 0.098]);
        ylim([0 inf])
        xlabel("Iteration")
        ylabel("Loss")
        grid on
        options.lineLossTrain = lineLossTrain;
    end
    disp(options)

    % Conduct DMD
    %[dmdA, dmdVals, dmdVecs, dmdProjA,~,rt] = fcn_pidmdvialsun(vt,dmdmethod,nModes4PODtrunc,islsun,options);
    [rt,rmse] = myfunc_(vt,ut,dmdmethod,nModes4PODtrunc,islsun,options);

    % Visualization
    if isVisualize
        if islsun
            strlsun = "via LSUN ";
        else
            strlsun = " ";
        end
        figure
        him = [];
        x = rt(:,:,iFrame);
        fcn_cylinderplot(him,x,clims);
        title("Reconstruction data w/ "+dmdmethod+" DMD "+ strlsun +"(" + "$[\hat{\mathbf{u}}]_{"+num2str(iFrame-1)+"}$)","Interpreter","latex")
        drawnow
        % exportgraphics

        figure
        plot(rmse)
        axis([0 length(rmse) 0 1])
        title("RMSE w/ "+dmdmethod+" DMD "+ strlsun +"(" + "$\hat{\mathbf{u}}$)","Interpreter","latex")
        xlabel("$k$","Interpreter","latex")
        ylabel("RMSE")
        drawnow
        % exportgraphics
    end

end
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
%   data: {"layout":"inline","rightPanelPercent":40}
%---
%[control:checkbox:9168]
%   data: {"defaultValue":false,"label":"isDisplayTraining","run":"Section"}
%---
%[control:checkbox:8307]
%   data: {"defaultValue":false,"label":"isVisualize","run":"Section"}
%---
