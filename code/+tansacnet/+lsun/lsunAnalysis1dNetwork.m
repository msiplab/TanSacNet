classdef lsunAnalysis1dNetwork < dlnetwork
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
        %NumberOfVanishingMoments
        NumberOfLevels
        NoDcLeakage
        Prefix
        AppendInOutLayers
        Device
        DType
    end

    properties (Access = private)
        % Private members
        InputSize
        NumberOfComponents
    end

     methods
        function obj = lsunAnalysis1dNetwork(varargin)
            %UNTITLED このクラスのインスタンスを作成
            %   詳細説明をここに記述
            
            % fcn_createlsunlgraph1d のプロパティ設定を参照
            p = inputParser;
            addParameter(p,'InputSize',32);
            addParameter(p,'NumberOfComponents',1);
            %addParameter(p,'NumberOfVanishingMoments',[1 1]);
            addParameter(p,'Stride',1);
            addParameter(p,'OverlappingFactor',1);
            addParameter(p,'NumberOfLevels',1);
            %addParameter(p,'NoDcLeakage',false);
            addParameter(p,'Prefix','');
            addParameter(p,'AppendInOutLayers',true);
            addParameter(p,'Device','cuda');
            addParameter(p,'DType','double');
            parse(p,varargin{:});

            % Parameters
            obj.InputSize = p.Results.InputSize;
            obj.NumberOfComponents = p.Results.NumberOfComponents;
            obj.TensorSize = [1 obj.InputSize obj.NumberOfComponents];  
            obj.Stride = p.Results.Stride;
            obj.OverlappingFactor = p.Results.OverlappingFactor;
            %obj.NumberOfVanishingMoments = p.Results.NumberOfVanishingMoments;
            obj.NumberOfLevels = p.Results.NumberOfLevels;
            %obj.NoDcLeakage = p.Results.NoDcLeakage;
            obj.Prefix = p.Results.Prefix;
            obj.AppendInOutLayers = p.Results.AppendInOutLayers;
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

        end
     end

       methods (Hidden=false)
           function dlnet = dlnetwork(obj)
               
               lsundlnet = dlnetwork;

               nComponents = obj.NumberOfComponents;
               inputSize = [1 obj.InputSize nComponents];
               stride = obj.Stride;
               ovlpFactor = obj.OverlappingFactor;
               nLevels = obj.NumberOfLevels;
               %
               device = obj.Device;
               dtype = obj.DType;
               %
               %noDcLeakage = obj.NumberOfVanishingMoments;

               nDecs = prod(stride);
               nChannels = [ceil(nDecs/2) floor(nDecs/2)];

               isapndinout = obj.AppendInOutLayers;
               prefix = obj.Prefix;

               % if nChannels(1) ~= nChannels(2)
               %     throw(MException('LsunLayer:InvalidStride',...
               %         '[%d %d] : The product of stride should be even.',...
               %         stride(1),stride(2)))
               % end
               if ~all(mod(ovlpFactor,2))
                   throw(MException('LsunLayer:InvalidOverlappingFactor',...
                       '%d : Currently, odd overlapping factors are only supported.',...
                       ovlpFactor))
               end

               %%
               nBlocks = inputSize(2)/stride;
               if mod(nBlocks,1)
                   error('%d : Input size should be multiple of stride.')
               end

               blockDctLayers = cell(nLevels);
               analysisLayers = cell(nLevels,nComponents);

               %blockIdctLayers = cell(nLevels);
               %synthesisLayers = cell(nLevels,nComponents);
               import tansacnet.lsun.*
               for iLv = 1:nLevels
                   strLv = sprintf('Lv%0d_',iLv);

                   % Initial blocks
                   blockDctLayers{iLv} = lsunBlockDct1dLayer('Name',[prefix strLv 'E0'],...
                       'Stride',stride,...
                       'NumberOfComponents',nComponents);
                   for iCmp = 1:nComponents
                       strCmp = sprintf('Cmp%0d_',iCmp);
                       analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                           lsunInitialFullRotation1dLayer('Name',[prefix strLv strCmp 'V0'],...
                           'NumberOfBlocks',nBlocks,'Stride',stride,...
                           'Device',device,...
                           'DType',dtype) %,...
                           ...'NoDcLeakage',noDcLeakage(1))
                           ];
                   end
                   % Final blocks
                   % blockIdctLayers{iLv} = lsunBlockIdct1dLayer('Name',[prefix strLv 'E0~'],...
                   %     'Stride',stride,...
                   %     'NumberOfComponents',nComponents);
                   % for iCmp = 1:nComponents
                   %     strCmp = sprintf('Cmp%0d_',iCmp);
                   %     synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                   %         lsunFinalFullRotation1dLayer('Name',[prefix strLv strCmp 'V0~'],...
                   %         'NumberOfBlocks',nBlocks,'Stride',stride,...
                   %         'Device',device,...
                   %         'DType',dtype)%,...
                   %         ...'NoDcLeakage',noDcLeakage(2))
                   %         ];
                   % end

                   for iCmp = 1:nComponents
                       strCmp = sprintf('Cmp%0d_',iCmp);

                       % Atom extension
                       mus_ = cat(1,ones(stride/2,nBlocks),-ones(stride/2,nBlocks)); % 初期化で非重複変換に縮退させるため
                       for iOrderV = 2:2:ovlpFactor-1
                           analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                               lsunCSAtomExtension1dLayer('Name',[prefix strLv strCmp 'Qx' num2str(iOrderV-1) 'rb'],...
                               'Stride',stride,'NumberOfBlocks',nBlocks,'Direction','Right','TargetChannels','Bottom','Mode','Analysis')
                               lsunIntermediateFullRotation1dLayer('Name',[prefix strLv strCmp 'Vx' num2str(iOrderV-1)],...
                               'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',mus_,...
                               'Device',device,...
                               'DType',dtype)
                               lsunCSAtomExtension1dLayer('Name',[prefix strLv strCmp 'Qx' num2str(iOrderV) 'lt'],...
                               'Stride',stride,'NumberOfBlocks',nBlocks,'Direction','Left','TargetChannels','Top','Mode','Analysis')
                               lsunIntermediateFullRotation1dLayer('Name',[prefix strLv strCmp 'Vx' num2str(iOrderV)],...
                               'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',mus_,...
                               'Device',device,...
                               'DType',dtype)
                               ];
                           % synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                           %     lsunCSAtomExtension1dLayer('Name',[prefix strLv strCmp 'Qx' num2str(iOrderV-1) 'rb~'],...
                           %     'Stride',stride,'NumberOfBlocks',nBlocks,'Direction','Left','TargetChannels','Bottom','Mode','Synthesis')
                           %     lsunIntermediateFullRotation1dLayer('Name',[prefix strLv strCmp 'Vx' num2str(iOrderV-1) '~'],...
                           %     'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',mus_,...
                           %     'Device',device,...
                           %     'DType',dtype)
                           %     lsunCSAtomExtension1dLayer('Name',[prefix strLv strCmp 'Qx' num2str(iOrderV) 'lt~'],...
                           %     'Stride',stride,'NumberOfBlocks',nBlocks,'Direction','Right','TargetChannels','Top','Mode','Synthesis')
                           %     lsunIntermediateFullRotation1dLayer('Name',[prefix strLv strCmp 'Vx' num2str(iOrderV) '~'],...
                           %     'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',mus_,...
                           %     'Device',device,...
                           %     'DType',dtype)
                           %     ];
                       end

                       % Channel separation and concatenation
                       analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                           lsunChannelSeparation1dLayer('Name',[prefix strLv strCmp 'Sp'])
                           ];
                       % synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                       %     lsunChannelConcatenation1dLayer('Name',[prefix strLv strCmp 'Cn'])
                       %     ];
                   end

               end

               % Level 1
               iLv = 1;
               strLv = sprintf('Lv%0d_',iLv);
               if isapndinout
                   lsundlnet = lsundlnet.addLayers(...
                       [ imageInputLayer(inputSize,...
                       'Name',[prefix 'Seq. input'],...
                       'Normalization','none'),...
                       lsunIdentityLayer(...
                       'Name',[prefix strLv 'In']),...
                       blockDctLayers{iLv}
                       ]);
               else
                   lsundlnet = lsundlnet.addLayers(...
                       [lsunIdentityLayer(...
                       'Name',[prefix strLv 'In']),...
                       blockDctLayers{iLv}]);
               end
               for iCmp = 1:nComponents
                   strCmp = sprintf('Cmp%0d_',iCmp);
                   lsundlnet = lsundlnet.addLayers(analysisLayers{iLv,iCmp});
                   if nComponents > 1
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv 'E0/out' num2str(iCmp)], [prefix strLv strCmp 'V0']);
                   else
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv 'E0'], [prefix strLv strCmp 'V0']);
                   end
               end
               % Output
               if nComponents > 1
                   lsundlnet = lsundlnet.addLayers(...
                       depthConcatenationLayer(nComponents,'Name',[prefix strLv 'AcOut']));
                   lsundlnet = lsundlnet.addLayers(...
                       depthConcatenationLayer(nComponents,'Name',[prefix strLv 'DcOut']));
               else
                   lsundlnet = lsundlnet.addLayers(...
                       lsunIdentityLayer('Name',[prefix strLv 'AcOut']));
                   lsundlnet = lsundlnet.addLayers(...
                       lsunIdentityLayer('Name',[prefix strLv 'DcOut']));
               end
               if nComponents > 1
                   for iCmp = 1:nComponents
                       strCmp = sprintf('Cmp%0d_',iCmp);
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' num2str(iCmp)]);
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' num2str(iCmp)]);
                   end
               else
                   strCmp = 'Cmp1_';
                   lsundlnet = lsundlnet.connectLayers(...
                       [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' ]);
                   lsundlnet = lsundlnet.connectLayers(...
                       [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' ]);
               end
               % Level n > 1
               for iLv = 2:nLevels
                   strLv = sprintf('Lv%0d_',iLv);
                   strLvPre = sprintf('Lv%0d_',iLv-1);
                   lsundlnet = lsundlnet.addLayers([
                       lsunIdentityLayer('Name',[prefix strLv 'In']),...
                       blockDctLayers{iLv}]);
                   lsundlnet = lsundlnet.connectLayers([prefix strLvPre 'DcOut'],[prefix strLv 'In']);
                   for iCmp = 1:nComponents
                       strCmp = sprintf('Cmp%0d_',iCmp);
                       lsundlnet = lsundlnet.addLayers(analysisLayers{iLv,iCmp});
                       if nComponents > 1
                           lsundlnet = lsundlnet.connectLayers(...
                               [prefix strLv 'E0/out' num2str(iCmp)], [prefix strLv strCmp 'V0']);
                       else
                           lsundlnet = lsundlnet.connectLayers(...
                               [prefix strLv 'E0'], [prefix strLv strCmp 'V0']);
                       end
                   end
                   % Output
                   if nComponents > 1
                       lsundlnet = lsundlnet.addLayers(...
                           depthConcatenationLayer(nComponents,'Name',[prefix strLv 'AcOut']));
                       lsundlnet = lsundlnet.addLayers(...
                           depthConcatenationLayer(nComponents,'Name',[prefix strLv 'DcOut']));
                   else
                       lsundlnet = lsundlnet.addLayers(...
                           lsunIdentityLayer('Name',[prefix strLv 'AcOut']));
                       lsundlnet = lsundlnet.addLayers(...
                           lsunIdentityLayer('Name',[prefix strLv 'DcOut']));
                   end
                   if nComponents > 1
                       for iCmp = 1:nComponents
                           strCmp = sprintf('Cmp%0d_',iCmp);
                           lsundlnet = lsundlnet.connectLayers(...
                               [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' num2str(iCmp)]);
                           lsundlnet = lsundlnet.connectLayers(...
                               [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' num2str(iCmp)]);
                       end
                   else
                       strCmp = 'Cmp1_';
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv strCmp 'Sp/ac'], [prefix strLv 'AcOut/in' ]);
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv strCmp 'Sp/dc'], [prefix strLv 'DcOut/in' ]);
                   end
               end

               dlnet = lsundlnet;
           end
       end
end