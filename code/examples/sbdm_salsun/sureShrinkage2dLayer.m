classdef sureShrinkage2dLayer < nnet.layer.Layer
    %SURESHRINKAGE2DLAYER Soft tanh shrinkage of the AC coefficients
    %
    %   Z = X - sigma*(kappa/g)*tanh(g*X/sigma), per AC channel p,
    %   kappa_p = sigmoid(A_p), g_p = exp(B_p).
    %   Same map and same initialisation as examples/sbdm/fcn_lsundenoise
    %   and fcn_lsuninitprm (kappa0 = 0.9, tau0 = 2), so that the uniform
    %   LSUN arm and the SA-LSUN arm share the 2*NumberOfChannels
    %   parameters of step 2.
    %
    %   Input/Output: nRows x nCols x NumberOfChannels x nSamples (SSCB),
    %   as the AcOut/AcIn layers of tansacnet.salsun.fcn_createsalsunlgraph2d.

    properties
        NumberOfChannels
        Sigma
    end

    properties (Learnable)
        A   % logit of kappa (NumberOfChannels x 1)
        B   % log of gain    (NumberOfChannels x 1)
    end

    methods
        function layer = sureShrinkage2dLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'NumberOfChannels',15)
            addParameter(p,'Sigma',25/255)
            addParameter(p,'Kappa0',0.9)
            addParameter(p,'Tau0',2.0)
            parse(p,varargin{:})
            layer.Name = p.Results.Name;
            layer.NumberOfChannels = p.Results.NumberOfChannels;
            layer.Sigma = single(p.Results.Sigma);
            k0 = p.Results.Kappa0; t0 = p.Results.Tau0;
            layer.A = single(log(k0/(1-k0)))*ones(layer.NumberOfChannels,1,'single');
            layer.B = single(log(k0/t0))*ones(layer.NumberOfChannels,1,'single');
            layer.Description = "SURE tanh shrinkage, " + layer.NumberOfChannels + " channels";
        end

        function Z = predict(layer, X)
            [kappa,gain] = kappagain(layer);
            s = cast(layer.Sigma,'like',X);
            Z = X - s.*(kappa./gain).*tanh(gain.*X./s);
        end

        function [kappa,gain] = kappagain(layer)
            kappa = reshape(1./(1+exp(-layer.A)),1,1,[]);
            gain  = reshape(exp(layer.B),1,1,[]);
        end
    end
end
