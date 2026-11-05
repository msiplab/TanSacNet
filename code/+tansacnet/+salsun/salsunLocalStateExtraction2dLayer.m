classdef salsunLocalStateExtraction2dLayer < nnet.layer.Layer
    %SALSUNLOCALSTATEEXTRACTION2DLAYER
    %
    %   Input  'in'  : nChs x nRows x nCols x nSamples
    %   Output 'out' : (numel(Channels)*nv*nh) x nRows x nCols x nSamples
    %
    % Requirements: MATLAB R2022a
    %
    % Copyright (c) 2026, Motoyasu SUZUKI and Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/

    properties
        NumberOfNeighborBlocks % [nv nh], both odd
        Channels               % row vector of channel indices to gather
    end

    methods
        function layer = salsunLocalStateExtraction2dLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'NumberOfNeighborBlocks',[3 3])
            addParameter(p,'Channels',[])
            parse(p,varargin{:})

            layer.Name = p.Results.Name;
            layer.NumberOfNeighborBlocks = p.Results.NumberOfNeighborBlocks;
            layer.Channels = p.Results.Channels;
            if any(mod(layer.NumberOfNeighborBlocks,2)==0)
                throw(MException('SaLsunLayer:InvalidNumberOfNeighborBlocks',...
                    '[%d %d] : NumberOfNeighborBlocks must both be odd.',...
                    layer.NumberOfNeighborBlocks(1),layer.NumberOfNeighborBlocks(2)))
            end
            layer.Description = "SA-LSUN local state extraction " ...
                + "(nv,nh) = (" ...
                + layer.NumberOfNeighborBlocks(1) + "," ...
                + layer.NumberOfNeighborBlocks(2) + "), " ...
                + "#channels = " + numel(layer.Channels);
        end

        function Z = predict(layer, X)
            nv = layer.NumberOfNeighborBlocks(1);
            nh = layer.NumberOfNeighborBlocks(2);
            if isdlarray(X)
                X = stripdims(X);
            end
            Xc = X(layer.Channels,:,:,:);
            Z = [];
            for vshift = fix(nv/2):-1:-fix(nv/2)
                for hshift = fix(nh/2):-1:-fix(nh/2)
                    Z = cat(1,Z,circshift(Xc,[0,vshift,hshift,0]));
                end
            end
        end

    end

end
