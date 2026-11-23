classdef salsunLocalStateExtraction2dLayer < nnet.layer.Layer
    %SALSUNLOCALSTATEEXTRACTION2DLAYER
    %
    %   Input  'in'  : nChs x nRows x nCols x nSamples
    %   Output 'out' : (numel(Channels)*nv*nh) x nRows x nCols x nSamples
    %
    %   The states of the nv x nh neighbor blocks of every block. Padding
    %   'circular' (default) takes the neighbors beyond the boundary from
    %   the opposite side; 'zero' takes zeros instead (for the boundary
    %   operation 'Termination' of fcn_createsalsunlgraph2d).
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
        Padding                % 'circular' or 'zero'
    end

    methods
        function layer = salsunLocalStateExtraction2dLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'NumberOfNeighborBlocks',[3 3])
            addParameter(p,'Channels',[])
            addParameter(p,'Padding','circular')
            parse(p,varargin{:})

            layer.Name = p.Results.Name;
            layer.NumberOfNeighborBlocks = p.Results.NumberOfNeighborBlocks;
            layer.Channels = p.Results.Channels;
            layer.Padding = validatestring(p.Results.Padding,{'circular','zero'});
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
            if strcmp(layer.Padding,'zero')
                layer.Description = layer.Description + ", zero padding";
            end
        end

        function Z = predict(layer, X)
            nv = layer.NumberOfNeighborBlocks(1);
            nh = layer.NumberOfNeighborBlocks(2);
            if isdlarray(X)
                X = stripdims(X);
            end
            Xc = X(layer.Channels,:,:,:);
            isZero = strcmp(layer.Padding,'zero');
            Z = [];
            for vshift = fix(nv/2):-1:-fix(nv/2)
                for hshift = fix(nh/2):-1:-fix(nh/2)
                    if isZero
                        Z = cat(1,Z,zeroshift(Xc,vshift,hshift));
                    else
                        Z = cat(1,Z,circshift(Xc,[0,vshift,hshift,0]));
                    end
                end
            end
        end

    end

end

function Y = zeroshift(X,vshift,hshift)
% Shift of the block grid by [vshift hshift] with zeros coming in
[~,nRows,nCols,~] = size(X);
Y = zeros(size(X),'like',X);
rd = max(1,1+vshift):min(nRows,nRows+vshift);
cd = max(1,1+hshift):min(nCols,nCols+hshift);
Y(:,rd,cd,:) = X(:,rd-vshift,cd-hshift,:);
end
