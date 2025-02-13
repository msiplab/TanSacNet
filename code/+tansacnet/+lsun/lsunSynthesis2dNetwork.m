classdef lsunSynthesis2dNetwork < dlnetwork
    %LSUNSYNTHESIS2dNETWORK
    %
    % Requirements: MATLAB R2024a
    %
    % Copyright (c) 2025, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %

    properties
        % See the property settings for fcn_createlsunlgraph2d
        TensorSize
        Stride
        OverlappingFactor
        NumberOfLevels
        NumberOfVanishingMoments
        NoDcLeakage
        Prefix
        AppendInOutLayers
        Device
        DType
    end

    properties (Access = private)
        % Private members
        PrivateInputSize
        PrivateNumberOfComponents
    end


    methods
        function obj = lsunSynthesis2dNetwork(varargin)
            % See the property settings for fcn_createlsunlgraph2d
            p = inputParser;
            addParameter(p,'InputSize',[32 32]);
            addParameter(p,'NumberOfComponents',1);
            addParameter(p,'NumberOfVanishingMoments',[1 1]);
            addParameter(p,'Stride',[2 2]);
            addParameter(p,'OverlappingFactor',[1 1]);
            addParameter(p,'NumberOfLevels',1);
            addParameter(p,'NoDcLeakage',false);
            addParameter(p,'Prefix','');
            addParameter(p,'AppendInOutLayers',true);
            addParameter(p,'Device','cuda');
            addParameter(p,'DType','double');
            parse(p,varargin{:});

            % Parameters
            obj.PrivateInputSize = p.Results.InputSize;
            obj.PrivateNumberOfComponents = p.Results.NumberOfComponents;
            obj.TensorSize = [obj.PrivateInputSize obj.PrivateNumberOfComponents];  
            obj.Stride = p.Results.Stride;
            obj.OverlappingFactor = p.Results.OverlappingFactor;
            obj.NumberOfVanishingMoments = p.Results.NumberOfVanishingMoments;
            obj.NumberOfLevels = p.Results.NumberOfLevels;
            obj.NoDcLeakage = p.Results.NoDcLeakage;
            obj.Prefix = p.Results.Prefix;
            obj.AppendInOutLayers = p.Results.AppendInOutLayers;
            obj.Device = p.Results.Device;
            obj.DType = p.Results.DType;
            
        end

        function analyzerFactory = getAnalyzerFactory(obj)
            %       
            analyzerFactory = [];
        end

        function adj = transpose(obj)
            % See fcn_cpparamssys2ana      
            adj = [];
        end
    end

    methods (Hidden=false)
        function dlnet = dlnetwork(obj)
            % See the property settings for fcn_createlsunlgraph2d
            dlnet = [];

            lsunSdlnet = dlnetwork;

            nComponents = obj.PrivateNumberOfComponents;
            inputSize = [obj.PrivateInputSize nComponents];
            stride = obj.Stride;
            ovlpFactor = obj.OverlappingFactor;
            nLevels = obj.NumberOfLevels;
            isapndinout = obj.AppendInOutLayers;
            %
            device = obj.Device;
            dtype = obj.DType;
            %
            noDcLeakage = obj.NumberOfVanishingMoments;

            if isscalar(noDcLeakage)
                noDcLeakage = [1 1]*noDcLeakage; 
            end

            prefix = obj.Prefix;
            %isapndinout = obj.AppendInOutLayers;

            nDecs = prod(stride);
            nChannels = [ceil(nDecs/2) floor(nDecs/2)];
            
            if nChannels(1) ~= nChannels(2)
                throw(MException('LsunLayer:InvalidStride',...
                    '[%d %d] : The product of stride should be even.',...
                    stride(1),stride(2)))
            end
            if ~all(mod(ovlpFactor,2))
                throw(MException('LsunLayer:InvalidOverlappingFactor',...
                    '[%d %d] : Currently, odd overlapping factors are only supported.',...
                    ovlpFactor(1),ovlpFactor(2)))
            end

            nBlocks = inputSize(1:2)./stride;
            if any(mod(nBlocks,1))
                error('[%d %d] : Input size should be multiple of stride.')
            end

            blockIdctLayers = cell(nLevels);
            synthesisLayers = cell(nLevels,nComponents);

            import tansacnet.lsun.*
            for iLv = 1:nLevels
                strLv = sprintf('Lv%0d_',iLv);

                % Initial blocks
                % blockDctLayers{iLv} = lsunBlockDct2dLayer('Name',[prefix strLv 'E0'],...
                %     'Stride',stride,...
                %     'NumberOfComponents',nComponents);
                % for iCmp = 1:nComponents
                %     strCmp = sprintf('Cmp%0d_',iCmp);
                %     analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                %         lsunInitialRotation2dLayer('Name',[prefix strLv strCmp 'V0'],...
                %         'NumberOfBlocks',nBlocks,'Stride',stride,...
                %         'NoDcLeakage',noDcLeakage(1),...
                %         'Device',device,...
                %         'DType',dtype)
                %         ];
                % end


                % Final blocks
                blockIdctLayers{iLv} = lsunBlockIdct2dLayer('Name',[prefix strLv 'E0~'],...
                    'Stride',stride,...
                    'NumberOfComponents',nComponents);
                for iCmp = 1:nComponents
                    strCmp = sprintf('Cmp%0d_',iCmp);
                    synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                        lsunFinalRotation2dLayer('Name',[prefix strLv strCmp 'V0~'],...
                        'NumberOfBlocks',nBlocks,'Stride',stride,...
                        'NoDcLeakage',noDcLeakage(2),...
                            'Device',device,...
                            'DType',dtype)
                        ];
                end

                % NOTE on the modification of parameter Mus
                %
                % Modified default parameter settings so that the default setting becomes a block DCT.
                % Note that in addition to the change in default design values, design parameters are no longer compatible with previous releases.
                % Redesign is strongly recommended!
                %
                % Reference:
                % S. Muramatsu, T. Kobayashi, M. Hiki and H. Kikuchi, "Boundary Operation of 2-D Nonseparable Linear-Phase Paraunitary Filter Banks," in IEEE Transactions on Image Processing, vol. 21, no. 4, pp. 2314-2318, April 2012, doi: 10.1109/TIP.2011.2181527.
                %
                % Date: 6 Setp. 2024
                %
                for iCmp = 1:nComponents
                    strCmp = sprintf('Cmp%0d_',iCmp);
                    % Atom extension in horizontal
                    for iOrderH = 2:2:ovlpFactor(2)-1
                        % analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                        %     lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qh' num2str(iOrderH-1) 'rd'],...
                        %     'Stride',stride,'Direction','Right','TargetChannels','Difference')
                        %     lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vh' num2str(iOrderH-1)],...
                        %     'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',-1,...
                        %     'Device',device,'DType',dtype)
                        %     lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qh' num2str(iOrderH) 'ls'],...
                        %     'Stride',stride,'Direction','Left','TargetChannels','Sum')
                        %     lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vh' num2str(iOrderH) ],...
                        %     'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',-1,...
                        %     'Device',device,...
                        %     'DType',dtype) % Revised default Mus to -1 on 6 Sept. 2024
                        %     ];
                        synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                            lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qh' num2str(iOrderH-1) 'rd~'],...
                            'Stride',stride,'Direction','Left','TargetChannels','Difference')
                            lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vh' num2str(iOrderH-1) '~'],...
                            'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',-1,...
                                'Device',device,...
                                'DType',dtype)
                            lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qh' num2str(iOrderH) 'ls~'],...
                            'Stride',stride,'Direction','Right','TargetChannels','Sum')
                            lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vh' num2str(iOrderH) '~'],...
                            'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',-1,...
                                'Device',device,...
                                'DType',dtype) % Revised default Mus to -1 on 6 Sept. 2024
                            ];
                    end
                    % Atom extension in vertical
                    for iOrderV = 2:2:ovlpFactor(1)-1
                        % analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                        %     lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qv' num2str(iOrderV-1) 'dd'],...
                        %     'Stride',stride,'Direction','Down','TargetChannels','Difference')
                        %     lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vv' num2str(iOrderV-1)],...
                        %     'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',-1,...
                        %     'Device',device,...
                        %     'DType',dtype)
                        %     lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qv' num2str(iOrderV) 'us'],...
                        %     'Stride',stride,'Direction','Up','TargetChannels','Sum')
                        %     lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vv' num2str(iOrderV)],...
                        %     'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',-1,...
                        %     'Device',device,...
                        %     'DType',dtype) % Revised default Mus to -1 on 6 Sept. 2024
                        %     ];
                        synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                            lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qv' num2str(iOrderV-1) 'dd~'],...
                            'Stride',stride,'Direction','Up','TargetChannels','Difference')
                            lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vv' num2str(iOrderV-1) '~'],...
                            'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',-1,...
                                'Device',device,...
                                'DType',dtype)
                            lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qv' num2str(iOrderV) 'us~'],...
                            'Stride',stride,'Direction','Down','TargetChannels','Sum')
                            lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vv' num2str(iOrderV) '~'],...
                            'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',-1,...
                                'Device',device,...
                                'DType',dtype) % Revised default Mus to -1 on 6 Sept. 2024
                            ];
                    end

                    % Channel separation and concatenation
                    % analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                    %     lsunChannelSeparation2dLayer('Name',[prefix strLv strCmp 'Sp'])
                    %     ];


                    synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                        lsunChannelConcatenation2dLayer('Name',[prefix strLv strCmp 'Cn'])
                        ];
                end

            end

            iLv = nLevels;
            strLv = sprintf('Lv%0d_',iLv);
            lsunSdlnet = lsunSdlnet.addLayers(...
                lsunComponentSeparation2dLayer(nComponents,'Name',[prefix strLv 'DcIn']));
            lsunSdlnet = lsunSdlnet.addLayers(...
                lsunComponentSeparation2dLayer(nComponents,'Name',[prefix strLv 'AcIn']));
            if nComponents > 1
                for iCmp = 1:nComponents
                    strCmp = sprintf('Cmp%0d_',iCmp);
                    lsunSdlnet = lsunSdlnet.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
                    lsunSdlnet = lsunSdlnet.connectLayers(...
                        [prefix strLv 'AcIn/out' num2str(iCmp) ], [prefix strLv strCmp 'Cn/ac']);
                    lsunSdlnet = lsunSdlnet.connectLayers(...
                        [prefix strLv 'DcIn/out' num2str(iCmp) ], [prefix strLv strCmp 'Cn/dc']);
                end
            else
                strCmp = 'Cmp1_';
                lsunSdlnet = lsunSdlnet.addLayers(synthesisLayers{iLv,1}(end:-1:1));
                lsunSdlnet = lsunSdlnet.connectLayers(...
                    [prefix strLv 'AcIn/out' ], [prefix strLv strCmp 'Cn/ac']);
                lsunSdlnet = lsunSdlnet.connectLayers(...
                    [prefix strLv 'DcIn/out' ], [prefix strLv strCmp 'Cn/dc']);
            end
            lsunSdlnet = lsunSdlnet.addLayers([
                blockIdctLayers{iLv},...
                lsunIdentityLayer('Name',[prefix strLv 'Out'])
                ]);
            for iCmp = 1:nComponents
                strCmp = sprintf('Cmp%0d_',iCmp);
                if nComponents > 1
                    lsunSdlnet = lsunSdlnet.connectLayers(...
                        [prefix strLv strCmp 'V0~'],[prefix strLv 'E0~/in' num2str(iCmp)]);
                else
                    lsunSdlnet = lsunSdlnet.connectLayers(...
                        [prefix strLv strCmp 'V0~'],[prefix strLv 'E0~']);
                end
            end

            % Level n < N
            for iLv = nLevels-1:-1:1
                strLv = sprintf('Lv%0d_',iLv);
                strLvPre = sprintf('Lv%0d_',iLv+1);
                if nComponents > 1
                    lsunSdlnet = lsunSdlnet.addLayers(...
                        lsunComponentSeparation2dLayer(nComponents,'Name',[prefix strLv 'DcIn']));
                    lsunSdlnet = lsunSdlnet.addLayers(...
                        lsunComponentSeparation2dLayer(nComponents,'Name',[prefix strLv 'AcIn']));
                else
                    lsunSdlnet = lsunSdlnet.addLayers(...
                        lsunIdentityLayer('Name',[prefix strLv 'DcIn']));
                    lsunSdlnet = lsunSdlnet.addLayers(...
                        lsunIdentityLayer('Name',[prefix strLv 'AcIn']));
                end
                lsunSdlnet = lsunSdlnet.connectLayers([prefix strLvPre 'Out'],[prefix strLv 'DcIn']);
                if nComponents > 1
                    for iCmp = 1:nComponents
                        strCmp = sprintf('Cmp%0d_',iCmp);
                        lsunSdlnet = lsunSdlnet.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
                        lsunSdlnet = lsunSdlnet.connectLayers(...
                            [prefix strLv 'AcIn/out' num2str(iCmp) ], [prefix strLv strCmp 'Cn/ac']);
                        lsunSdlnet = lsunSdlnet.connectLayers(...
                            [prefix strLv 'DcIn/out' num2str(iCmp) ], [prefix strLv strCmp 'Cn/dc']);
                    end
                else
                    strCmp = 'Cmp1_';
                    lsunSdlnet = lsunSdlnet.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
                    lsunSdlnet = lsunSdlnet.connectLayers(...
                        [prefix strLv 'AcIn/out' ], [prefix strLv strCmp 'Cn/ac']);
                    lsunSdlnet = lsunSdlnet.connectLayers(...
                        [prefix strLv 'DcIn/out'  ], [prefix strLv strCmp 'Cn/dc']);
                end
                lsunSdlnet = lsunSdlnet.addLayers([
                    blockIdctLayers{iLv},...
                    lsunIdentityLayer('Name',[prefix strLv 'Out'])
                    ]);
                for iCmp = 1:nComponents
                    strCmp = sprintf('Cmp%0d_',iCmp);
                    if nComponents > 1
                        lsunSdlnet = lsunSdlnet.connectLayers(...
                            [prefix strLv strCmp 'V0~'],[prefix strLv 'E0~/in' num2str(iCmp)]);
                    else
                        lsunSdlnet = lsunSdlnet.connectLayers(...
                            [prefix strLv strCmp 'V0~'], [prefix strLv 'E0~']);
                    end
                end
            end

            
            for iLv = 1:nLevels
                strLv = sprintf('Lv%0d_',iLv);
                inputSubSize(1:2) = inputSize(1:2)./(stride.^iLv);
                inputSubSize(3) = nComponents*(sum(nChannels)-1);
                if isapndinout
                    lsunSdlnet = lsunSdlnet.addLayers(...
                        imageInputLayer(inputSubSize,...
                        'Name',[prefix  strLv 'Ac feature input'],...
                        'Normalization','none'));
                    lsunSdlnet = lsunSdlnet.connectLayers(...
                        [prefix strLv 'Ac feature input'],[prefix strLv 'AcIn'] );
                end
            end
            strLv = sprintf('Lv%0d_',nLevels);
            inputSubSize(1:2) = inputSize(1:2)./(stride.^nLevels);
            inputSubSize(3) = nComponents;
            if isapndinout
                lsunSdlnet = lsunSdlnet.addLayers(...
                    imageInputLayer(inputSubSize,...
                    'Name',[prefix  strLv 'Dc feature input'],...
                    'Normalization','none'));
                if iLv == nLevels
                    lsunSdlnet = lsunSdlnet.connectLayers(...
                        [prefix strLv 'Dc feature input'],[prefix strLv 'DcIn']);
                end
            end


            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % lsunSdlnet = addLayers(lsunSdlnet, imageInputLayer(inputSize, ...
            %     'Name', [prefix 'Image input_Dc'], ...
            %     'Normalization', 'none'));
            % lsunSdlnet = addLayers(lsunSdlnet, imageInputLayer(inputSize, ...
            %     'Name', [prefix 'Image input_Ac'], ...
            %     'Normalization', 'none'));
            % lsunSdlnet = connectLayers(lsunSdlnet, [prefix 'Image input_Dc'], [prefix 'Lv1_DcIn']);
            % lsunSdlnet = connectLayers(lsunSdlnet, [prefix 'Image input_Ac'], [prefix 'Lv1_AcIn']);
            % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            dlnet = lsunSdlnet;
        end
    end

end