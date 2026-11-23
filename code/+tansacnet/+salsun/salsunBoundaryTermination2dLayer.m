classdef salsunBoundaryTermination2dLayer < nnet.layer.Layer
    %SALSUNBOUNDARYTERMINATION2DLAYER Termination of the boundary blocks
    %
    %   Input 'angles' (default):
    %     Input  'in'  : nAngles x (nRows*nCols) x nSamples, the angles of
    %                    an intermediate rotation (blocks in column-major
    %                    order)
    %     Output 'out' : the same angles, with those of the boundary
    %                    blocks set to zero
    %   Input 'state':
    %     Input  'in'  : nChs x nRows x nCols x nSamples, the state that
    %                    the estimator of the rotation watches
    %     Output 'out' : the same state, zero at the boundary blocks
    %
    %   The boundary operation 'Termination' of
    %   tansacnet.salsun.fcn_createsalsunlgraph2d, as BoundaryOperation
    %   'Termination' of the NSOLT of SaivDr: the atom extension of a
    %   stage with the target channels 'Difference' shifts half of the
    %   coefficients circularly by one block in Direction, so that the
    %   blocks at one boundary (the first column for 'Right', the last
    %   for 'Left', the first row for 'Down', the last for 'Up') receive
    %   the coefficients of the opposite boundary. With all angles zero
    %   the rotation of these blocks is diag(Mus) = -I (Mus = -1), and the
    %   atom extension of the next stage (target channels 'Sum', the
    %   opposite direction) brings those coefficients back, so that no
    %   basis function wraps around the boundary. Until then the boundary
    %   blocks hold coefficients of the opposite boundary, so their state
    %   is also set to zero for the estimator (Input 'state'): its
    %   neighbors would otherwise see across the boundary.
    %
    % Requirements: MATLAB R2022a
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

    properties
        NumberOfBlocks  % [nRows nCols]
        Direction       % direction of the preceding atom extension
        Input           % 'angles' or 'state'
    end

    methods
        function layer = salsunBoundaryTermination2dLayer(varargin)
            p = inputParser;
            addParameter(p,'Name','')
            addParameter(p,'NumberOfBlocks',[1 1])
            addParameter(p,'Direction','Right')
            addParameter(p,'Input','angles')
            parse(p,varargin{:})

            layer.Name = p.Results.Name;
            layer.NumberOfBlocks = p.Results.NumberOfBlocks;
            layer.Direction = validatestring(p.Results.Direction,...
                {'Right','Left','Down','Up'});
            layer.Input = validatestring(p.Results.Input,{'angles','state'});
            layer.Description = "SA-LSUN boundary termination of the " ...
                + layer.Input + " (" + layer.Direction + ")";
        end

        function Z = predict(layer, X)
            mask = tansacnet.salsun.salsunBoundaryTermination2dLayer.blockMask(...
                layer.NumberOfBlocks,layer.Direction);
            if strcmp(layer.Input,'state')
                mask = reshape(mask,[1 layer.NumberOfBlocks]);
            end
            Z = X.*cast(mask,'like',X);
        end
    end

    methods (Static)
        function mask = blockMask(nBlocks,direction)
            % 1 x (nRows*nCols), zero at the blocks that receive the
            % coefficients of the opposite boundary (column-major order)
            m = ones(nBlocks(1),nBlocks(2));
            switch char(direction)
                case 'Right', m(:,1) = 0;
                case 'Left',  m(:,end) = 0;
                case 'Down',  m(1,:) = 0;
                case 'Up',    m(end,:) = 0;
            end
            mask = reshape(m,1,[]);
        end
    end

end
