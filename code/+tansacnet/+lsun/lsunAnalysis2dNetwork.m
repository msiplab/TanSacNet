classdef lsunAnalysis2dNetwork < dlnetwork
    %LSUNANALYSIS2dNETWORK
    %
    % Requirements: MATLAB R2024a
    %
    % Copyright (c) 2025, Shogo MURAMATSU, Yasas GODAGE
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %


    properties
        % fcn_createlsunlgraph2d 参照
        TensorSize
        Stride
        OverlappingFactor
        NumberOfVanishingMoments
        NumberOfLevels
        NoDcLeakage
        Prefix
        Device
        DType
    end

    properties (Access = private)
        % Private members
        PrivateInputSize
        PrivateNumberOfComponents
    end

    methods
        function obj = lsunAnalysis2dNetwork(varargin)
            %UNTITLED このクラスのインスタンスを作成
            %   詳細説明をここに記述
            
            % fcn_createlsunlgraph2d のプロパティ設定を参照
            p = inputParser;
            addParameter(p,'InputSize',[32 32]);
            addParameter(p,'NumberOfComponents',1);
            addParameter(p,'NumberOfVanishingMoments',[1 1]);
            addParameter(p,'Stride',[2 2]);
            addParameter(p,'OverlappingFactor',[1 1]);
            addParameter(p,'NumberOfLevels',1);
            addParameter(p,'NoDcLeakage',false);
            addParameter(p,'Prefix','');
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
            obj.Device = p.Results.Device;
            obj.DType = p.Results.DType;
        end

        function analyzerFactory = getAnalyzerFactory(obj)
            % ???      
            analyzerFactory = [];
        end

        function adj = transpose(obj)
            %METHOD1 このメソッドの概要をここに記述
            %   詳細説明をここに記述

            % fcn_cpparamsana2syn を参照

            % import tansacnet.lsun.*
            % %
            % expanalyzer = '^Lv\d+_Cmp\d+_V(\w\d|0)+$';
            % nLayers = height(analysisdlnet.Layers);
            % for iLayer = 1:nLayers
            %     alayer = analysisdlnet.Layers(iLayer);
            %     alayerName = alayer.Name;
            %     if ~isempty(regexp(alayerName,expanalyzer,'once'))
            %         slayer = synthesislgraph.Layers({synthesislgraph.Layers.Name} == alayerName + "~");
            %         alayer.Angles = slayer.Angles;
            %         alayer.Mus = slayer.Mus;
            %         if isa(alayer,'tansacnet.lsun.lsunInitialRotation2dLayer')
            %             alayer.NoDcLeakage = slayer.NoDcLeakage;
            %         end
            %         analysisdlnet = analysisdlnet.replaceLayer(alayerName,alayer);
            %         disp("Copy angles from " + slayer.Name + " to " + alayerName)
            %     end
            % end
        end
    end

    methods (Hidden=false)
        function dlnet = dlnetwork(obj)
            
            % lsundlnet = [];
            % if isempty(lsundlnet)
            %     lsundlnet = dlnetwork;
            % end
            lsundlnet = dlnetwork;

            nComponents = obj.PrivateNumberOfComponents;
            inputSize = [obj.PrivateInputSize nComponents];
            stride = obj.Stride;
            ovlpFactor = obj.OverlappingFactor;
            nLevels = obj.NumberOfLevels;
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

            blockDctLayers = cell(nLevels);
            analysisLayers = cell(nLevels,nComponents);
            %blockIdctLayers = cell(nLevels);
            %synthesisLayers = cell(nLevels,nComponents);
            import tansacnet.lsun.*
            for iLv = 1:nLevels
                strLv = sprintf('Lv%0d_',iLv);
                
                % Initial blocks
                blockDctLayers{iLv} = lsunBlockDct2dLayer('Name',[prefix strLv 'E0'],...
                    'Stride',stride,...
                    'NumberOfComponents',nComponents);
                for iCmp = 1:nComponents
                    strCmp = sprintf('Cmp%0d_',iCmp);
                    analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                        lsunInitialRotation2dLayer('Name',[prefix strLv strCmp 'V0'],...
                        'NumberOfBlocks',nBlocks,'Stride',stride,...
                        'NoDcLeakage',noDcLeakage(1),...
                         'Device',device,...
                         'DType',dtype)
                        ];
                end

                
                % Final blocks
                % blockIdctLayers{iLv} = lsunBlockIdct2dLayer('Name',[prefix strLv 'E0~'],...
                %     'Stride',stride,...
                %     'NumberOfComponents',nComponents);
                % for iCmp = 1:nComponents
                %     strCmp = sprintf('Cmp%0d_',iCmp);
                %     synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                %         lsunFinalRotation2dLayer('Name',[prefix strLv strCmp 'V0~'],...
                %         'NumberOfBlocks',nBlocks,'Stride',stride,...
                %         'NoDcLeakage',noDcLeakage(2),...
                %             'Device',device,...
                %             'DType',dtype)
                %         ];
                % end
                
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
                        analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                            lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qh' num2str(iOrderH-1) 'rd'],...
                            'Stride',stride,'Direction','Right','TargetChannels','Difference')
                            lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vh' num2str(iOrderH-1)],...
                            'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',-1,...
                                'Device',device,'DType',dtype)
                            lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qh' num2str(iOrderH) 'ls'],...
                            'Stride',stride,'Direction','Left','TargetChannels','Sum')
                            lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vh' num2str(iOrderH) ],...
                            'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',-1,...
                                'Device',device,...
                                'DType',dtype) % Revised default Mus to -1 on 6 Sept. 2024
                            ];
                        % synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                        %     lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qh' num2str(iOrderH-1) 'rd~'],...
                        %     'Stride',stride,'Direction','Left','TargetChannels','Difference')
                        %     lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vh' num2str(iOrderH-1) '~'],...
                        %     'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',-1,...
                        %         'Device',device,...
                        %         'DType',dtype)
                        %     lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qh' num2str(iOrderH) 'ls~'],...
                        %     'Stride',stride,'Direction','Right','TargetChannels','Sum')
                        %     lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vh' num2str(iOrderH) '~'],...
                        %     'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',-1,...
                        %         'Device',device,...
                        %         'DType',dtype) % Revised default Mus to -1 on 6 Sept. 2024
                        %     ];
                    end
                    % Atom extension in vertical
                    for iOrderV = 2:2:ovlpFactor(1)-1
                        analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                            lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qv' num2str(iOrderV-1) 'dd'],...
                            'Stride',stride,'Direction','Down','TargetChannels','Difference')
                            lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vv' num2str(iOrderV-1)],...
                            'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',-1,...
                                'Device',device,...
                                'DType',dtype)
                            lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qv' num2str(iOrderV) 'us'],...
                            'Stride',stride,'Direction','Up','TargetChannels','Sum')
                            lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vv' num2str(iOrderV)],...
                            'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',-1,...
                                'Device',device,...
                                'DType',dtype) % Revised default Mus to -1 on 6 Sept. 2024
                            ];
                        % synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                        %     lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qv' num2str(iOrderV-1) 'dd~'],...
                        %     'Stride',stride,'Direction','Up','TargetChannels','Difference')
                        %     lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vv' num2str(iOrderV-1) '~'],...
                        %     'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',-1,...
                        %         'Device',device,...
                        %         'DType',dtype)
                        %     lsunAtomExtension2dLayer('Name',[prefix strLv strCmp 'Qv' num2str(iOrderV) 'us~'],...
                        %     'Stride',stride,'Direction','Down','TargetChannels','Sum')
                        %     lsunIntermediateRotation2dLayer('Name',[prefix strLv strCmp 'Vv' num2str(iOrderV) '~'],...
                        %     'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',-1,...
                        %         'Device',device,...
                        %         'DType',dtype) % Revised default Mus to -1 on 6 Sept. 2024
                        %     ];
                    end
                    
                    % Channel separation and concatenation
                    analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                      lsunChannelSeparation2dLayer('Name',[prefix strLv strCmp 'Sp'])
                        ];


                    % synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                    %     lsunChannelConcatenation2dLayer('Name',[prefix strLv strCmp 'Cn'])
                    %     ];               
                end
                
            end
            
            iLv = 1;
            strLv = sprintf('Lv%0d_',iLv);

            lsundlnet = addLayers(lsundlnet,[imageInputLayer(inputSize,'Name',[prefix 'Image input'],'Normalization','none'), ...
            lsunIdentityLayer('Name',[prefix strLv 'In']), blockDctLayers{iLv}]);

            %lsundlnet = addLayers(lsundlnet,[lsunIdentityLayer('Name',[prefix strLv 'In']), blockDctLayers{iLv}]);

            for iCmp = 1:nComponents
                strCmp = sprintf('Cmp%0d_',iCmp);
                lsundlnet = addLayers(lsundlnet,analysisLayers{iLv,iCmp});
                if nComponents > 1
                    lsundlnet = connectLayers(lsundlnet,...
                        [prefix strLv 'E0/out' num2str(iCmp)], [prefix strLv strCmp 'V0']);
                else
                    lsundlnet = connectLayers(lsundlnet,...
                        [prefix strLv 'E0'], [prefix strLv strCmp 'V0']);
                end
            end

            if nComponents > 1
                lsundlnet = addLayers(lsundlnet,...
                    depthConcatenationLayer(nComponents,'Name',[prefix strLv 'AcOut']));
                lsundlnet = addLayers(lsundlnet,...
                    depthConcatenationLayer(nComponents,'Name',[prefix strLv 'DcOut']));
            else
                lsundlnet = addLayers(lsundlnet,...
                    lsunIdentityLayer('Name',[prefix strLv 'AcOut']));
                lsundlnet = addLayers(lsundlnet,...
                    lsunIdentityLayer('Name',[prefix strLv 'DcOut']));
            end

            if nComponents > 1
                for iCmp = 1:nComponents
                    strCmp = sprintf('Cmp%0d_',iCmp);
                    lsundlnet = connectLayers(lsundlnet,...
                        [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' num2str(iCmp)]);
                    lsundlnet = connectLayers(lsundlnet,...
                        [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' num2str(iCmp)]);
                end
            else
                strCmp = 'Cmp1_';
                lsundlnet = connectLayers(lsundlnet,...
                    [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' ]);
                lsundlnet = connectLayers(lsundlnet,...
                    [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' ]);
            end

            % Level n > 1
            for iLv = 2:nLevels
                strLv = sprintf('Lv%0d_',iLv);
                strLvPre = sprintf('Lv%0d_',iLv-1);
                lsundlnet = addLayers(lsundlnet,[
                    lsunIdentityLayer('Name',[prefix strLv 'In']),...
                    blockDctLayers{iLv}]);
                lsundlnet = connectLayers(lsundlnet,[prefix strLvPre 'DcOut'],[prefix strLv 'In']);
                for iCmp = 1:nComponents
                    strCmp = sprintf('Cmp%0d_',iCmp);
                    lsundlnet = addLayers(lsundlnet,analysisLayers{iLv,iCmp});
                    if nComponents > 1
                        lsundlnet = connectLayers(lsundlnet,...
                            [prefix strLv 'E0/out' num2str(iCmp)], [prefix strLv strCmp 'V0']);
                    else
                        lsundlnet = connectLayers(lsundlnet,...
                            [prefix strLv 'E0'], [prefix strLv strCmp 'V0']);
                    end
                end
                % Output
                if nComponents > 1
                    lsundlnet = addLayers(lsundlnet,...
                        depthConcatenationLayer(nComponents,'Name',[prefix strLv 'AcOut']));
                    lsundlnet = addLayers(lsundlnet,...
                        depthConcatenationLayer(nComponents,'Name',[prefix strLv 'DcOut']));
                else
                    lsundlnet = addLayers(lsundlnet,...
                        lsunIdentityLayer('Name',[prefix strLv 'AcOut']));
                    lsundlnet = addLayers(lsundlnet,...
                        lsunIdentityLayer('Name',[prefix strLv 'DcOut']));
                end
                if nComponents > 1
                    for iCmp = 1:nComponents
                        strCmp = sprintf('Cmp%0d_',iCmp);
                        lsundlnet = connectLayers(lsundlnet,...
                            [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' num2str(iCmp)]);
                        lsundlnet = connectLayers(lsundlnet,...
                            [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' num2str(iCmp)]);
                    end
                else
                    strCmp = 'Cmp1_';
                    lsundlnet = connectLayers(lsundlnet,...
                        [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' ]);
                    lsundlnet = connectLayers(lsundlnet,...
                        [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' ]);
                end
            end

            dlnet = lsundlnet;

           % ???
            %dlnet = [];
        end
    end

end
