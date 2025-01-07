classdef lsunAnalys2dNetwork < dlnetwork
    %UNTITLED このクラスの概要をここに記述
    %   詳細説明をここに記述

    properties
        % fcn_createlsunlgraph2d 参照
    end

    methods
        function obj = lsunAnalysis2dNetwork(varargin)
            %UNTITLED このクラスのインスタンスを作成
            %   詳細説明をここに記述
            
            % fcn_createlsunlgraph2d のプロパティ設定を参照
        end

        function analyzerFactory = getAnalyzerFactory(net)
            % ???
            analyzerFactory = [];
        end

        function adjnet = transpose(obj)
            %METHOD1 このメソッドの概要をここに記述
            %   詳細説明をここに記述
            
            % fcn_cpparamsana2syn を参照
        end
    end

    methods (Hidden=false)
        function dlnet = dlnetwork(net)
           % ???
            dlnet = [];
        end
    end

end
