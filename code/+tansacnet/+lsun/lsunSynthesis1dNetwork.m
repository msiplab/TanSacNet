classdef lsunSynthesis1dNetwork < dlnetwork
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
         function obj = lsunSynthesis1dNetwork(varargin)
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
               % blockDctLayers = cell(nLevels);
               % analysisLayers = cell(nLevels,nComponents);

               blockIdctLayers = cell(nLevels);
               synthesisLayers = cell(nLevels,nComponents);

               import tansacnet.lsun.*
               for iLv = 1:nLevels
                   strLv = sprintf('Lv%0d_',iLv);

                   % Initial blocks
                   % blockDctLayers{iLv} = lsunBlockDct1dLayer('Name',[prefix strLv 'E0'],...
                   %     'Stride',stride,...
                   %     'NumberOfComponents',nComponents);
                   % for iCmp = 1:nComponents
                   %     strCmp = sprintf('Cmp%0d_',iCmp);
                   %     analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                   %         lsunInitialFullRotation1dLayer('Name',[prefix strLv strCmp 'V0'],...
                   %         'NumberOfBlocks',nBlocks,'Stride',stride,...
                   %         'Device',device,...
                   %         'DType',dtype) %,...
                   %         ...'NoDcLeakage',noDcLeakage(1))
                   %         ];
                   % end
                   % Final blocks
                   blockIdctLayers{iLv} = lsunBlockIdct1dLayer('Name',[prefix strLv 'E0~'],...
                       'Stride',stride,...
                       'NumberOfComponents',nComponents);
                   for iCmp = 1:nComponents
                       strCmp = sprintf('Cmp%0d_',iCmp);
                       synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                           lsunFinalFullRotation1dLayer('Name',[prefix strLv strCmp 'V0~'],...
                           'NumberOfBlocks',nBlocks,'Stride',stride,...
                           'Device',device,...
                           'DType',dtype)%,...
                           ...'NoDcLeakage',noDcLeakage(2))
                           ];
                   end

                   for iCmp = 1:nComponents
                       strCmp = sprintf('Cmp%0d_',iCmp);

                       % Atom extension
                       mus_ = cat(1,ones(stride/2,nBlocks),-ones(stride/2,nBlocks)); % 初期化で非重複変換に縮退させるため
                       for iOrderV = 2:2:ovlpFactor-1
                           % analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                           %     lsunCSAtomExtension1dLayer('Name',[prefix strLv strCmp 'Qx' num2str(iOrderV-1) 'rb'],...
                           %     'Stride',stride,'NumberOfBlocks',nBlocks,'Direction','Right','TargetChannels','Bottom','Mode','Analysis')
                           %     lsunIntermediateFullRotation1dLayer('Name',[prefix strLv strCmp 'Vx' num2str(iOrderV-1)],...
                           %     'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',mus_,...
                           %     'Device',device,...
                           %     'DType',dtype)
                           %     lsunCSAtomExtension1dLayer('Name',[prefix strLv strCmp 'Qx' num2str(iOrderV) 'lt'],...
                           %     'Stride',stride,'NumberOfBlocks',nBlocks,'Direction','Left','TargetChannels','Top','Mode','Analysis')
                           %     lsunIntermediateFullRotation1dLayer('Name',[prefix strLv strCmp 'Vx' num2str(iOrderV)],...
                           %     'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Analysis','Mus',mus_,...
                           %     'Device',device,...
                           %     'DType',dtype)
                           %     ];
                           synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                               lsunCSAtomExtension1dLayer('Name',[prefix strLv strCmp 'Qx' num2str(iOrderV-1) 'rb~'],...
                               'Stride',stride,'NumberOfBlocks',nBlocks,'Direction','Left','TargetChannels','Bottom','Mode','Synthesis')
                               lsunIntermediateFullRotation1dLayer('Name',[prefix strLv strCmp 'Vx' num2str(iOrderV-1) '~'],...
                               'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',mus_,...
                               'Device',device,...
                               'DType',dtype)
                               lsunCSAtomExtension1dLayer('Name',[prefix strLv strCmp 'Qx' num2str(iOrderV) 'lt~'],...
                               'Stride',stride,'NumberOfBlocks',nBlocks,'Direction','Right','TargetChannels','Top','Mode','Synthesis')
                               lsunIntermediateFullRotation1dLayer('Name',[prefix strLv strCmp 'Vx' num2str(iOrderV) '~'],...
                               'Stride',stride,'NumberOfBlocks',nBlocks,'Mode','Synthesis','Mus',mus_,...
                               'Device',device,...
                               'DType',dtype)
                               ];
                       end

                       % Channel separation and concatenation
                       % analysisLayers{iLv,iCmp} = [ analysisLayers{iLv,iCmp}
                       %     lsunChannelSeparation1dLayer('Name',[prefix strLv strCmp 'Sp'])
                       %     ];
                       synthesisLayers{iLv,iCmp} = [ synthesisLayers{iLv,iCmp}
                           lsunChannelConcatenation1dLayer('Name',[prefix strLv strCmp 'Cn'])
                           ];
                   end

               end

               % Level N
               iLv = nLevels;
               strLv = sprintf('Lv%0d_',iLv);
               lsundlnet = lsundlnet.addLayers(...
                   lsunComponentSeparation1dLayer(nComponents,'Name',[prefix strLv 'DcIn']));
               lsundlnet = lsundlnet.addLayers(...
                   lsunComponentSeparation1dLayer(nComponents,'Name',[prefix strLv 'AcIn']));
               if nComponents > 1
                   for iCmp = 1:nComponents
                       strCmp = sprintf('Cmp%0d_',iCmp);
                       lsundlnet = lsundlnet.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv 'AcIn/out' num2str(iCmp) ], [prefix strLv strCmp 'Cn/ac']);
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv 'DcIn/out' num2str(iCmp) ], [prefix strLv strCmp 'Cn/dc']);
                   end
               else
                   strCmp = 'Cmp1_';
                   lsundlnet = lsundlnet.addLayers(synthesisLayers{iLv,1}(end:-1:1));
                   lsundlnet = lsundlnet.connectLayers(...
                       [prefix strLv 'AcIn/out' ], [prefix strLv strCmp 'Cn/ac']);
                   lsundlnet = lsundlnet.connectLayers(...
                       [prefix strLv 'DcIn/out' ], [prefix strLv strCmp 'Cn/dc']);
               end
               lsundlnet = lsundlnet.addLayers([
                   blockIdctLayers{iLv},...
                   lsunIdentityLayer('Name',[prefix strLv 'Out'])
                   ]);
               for iCmp = 1:nComponents
                   strCmp = sprintf('Cmp%0d_',iCmp);
                   if nComponents > 1
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv strCmp 'V0~'],[prefix strLv 'E0~/in' num2str(iCmp)]);
                   else
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv strCmp 'V0~'],[prefix strLv 'E0~']);
                   end
               end

               % Level n < N
               for iLv = nLevels-1:-1:1
                   strLv = sprintf('Lv%0d_',iLv);
                   strLvPre = sprintf('Lv%0d_',iLv+1);
                   if nComponents > 1
                       lsundlnet = lsundlnet.addLayers(...
                           lsunComponentSeparation1dLayer(nComponents,'Name',[prefix strLv 'DcIn']));
                       lsundlnet = lsundlnet.addLayers(...
                           lsunComponentSeparation1dLayer(nComponents,'Name',[prefix strLv 'AcIn']));
                   else
                       lsundlnet = lsundlnet.addLayers(...
                           lsunIdentityLayer('Name',[prefix strLv 'DcIn']));
                       lsundlnet = lsundlnet.addLayers(...
                           lsunIdentityLayer('Name',[prefix strLv 'AcIn']));
                   end
                   lsundlnet = lsundlnet.connectLayers([prefix strLvPre 'Out'],[prefix strLv 'DcIn']);
                   if nComponents > 1
                       for iCmp = 1:nComponents
                           strCmp = sprintf('Cmp%0d_',iCmp);
                           lsundlnet = lsundlnet.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
                           lsundlnet = lsundlnet.connectLayers(...
                               [prefix strLv 'AcIn/out' num2str(iCmp) ], [prefix strLv strCmp 'Cn/ac']);
                           lsundlnet = lsundlnet.connectLayers(...
                               [prefix strLv 'DcIn/out' num2str(iCmp) ], [prefix strLv strCmp 'Cn/dc']);
                       end
                   else
                       strCmp = 'Cmp1_';
                       lsundlnet = lsundlnet.addLayers(synthesisLayers{iLv,iCmp}(end:-1:1));
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv 'AcIn/out' ], [prefix strLv strCmp 'Cn/ac']);
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv 'DcIn/out'  ], [prefix strLv strCmp 'Cn/dc']);
                   end
                   lsundlnet = lsundlnet.addLayers([
                       blockIdctLayers{iLv},...
                       lsunIdentityLayer('Name',[prefix strLv 'Out'])
                       ]);
                   for iCmp = 1:nComponents
                       strCmp = sprintf('Cmp%0d_',iCmp);
                       if nComponents > 1
                           lsundlnet = lsundlnet.connectLayers(...
                               [prefix strLv strCmp 'V0~'],[prefix strLv 'E0~/in' num2str(iCmp)]);
                       else
                           lsundlnet = lsundlnet.connectLayers(...
                               [prefix strLv strCmp 'V0~'], [prefix strLv 'E0~']);
                       end
                   end
               end

               for iLv = 1:nLevels
                   strLv = sprintf('Lv%0d_',iLv);
                   inputSubSize(1) = sum(nChannels)-1;
                   inputSubSize(2) = 1;
                   inputSubSize(3) = inputSize(2)/(stride.^iLv);
                   if isapndinout
                       lsundlnet = lsundlnet.addLayers(...
                           imageInputLayer(inputSubSize,...
                           'Name',[prefix  strLv 'Ac feature input'],...
                           'Normalization','none'));
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv 'Ac feature input'],[prefix strLv 'AcIn'] );
                   end
               end
               strLv = sprintf('Lv%0d_',nLevels);
               inputSubSize(1) = 1;
               inputSubSize(2) = 1;
               inputSubSize(3) = inputSize(2)/(stride.^nLevels);
               if isapndinout
                   lsundlnet = lsundlnet.addLayers(...
                       imageInputLayer(inputSubSize,...
                       'Name',[prefix  strLv 'Dc feature input'],...
                       'Normalization','none'));
                   if iLv == nLevels
                       lsundlnet = lsundlnet.connectLayers(...
                           [prefix strLv 'Dc feature input'],[prefix strLv 'DcIn']);
                   end
               end

               dlnet = lsundlnet;
           end
       end
end