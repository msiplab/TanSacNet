classdef lsunAnalys2dNetwork < dlnetwork
    %UNTITLED このクラスの概要をここに記述
    %   詳細説明をここに記述

    properties
        % fcn_createlsunlgraph2d 参照
        TensorSize
        Stride
        OverlappingFactor
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
        function obj = lsunAnalys2dNetwork(varargin)
            %UNTITLED このクラスのインスタンスを作成
            %   詳細説明をここに記述
            
            % fcn_createlsunlgraph2d のプロパティ設定を参照
            p = inputParser;
            addParameter(p,'InputSize',[32 32]);
            addParameter(p,'NumberOfComponents',1);
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
        end
    end

    methods (Hidden=false)
        function dlnet = dlnetwork(obj)
           % ???
            dlnet = [];
        end
    end

end
