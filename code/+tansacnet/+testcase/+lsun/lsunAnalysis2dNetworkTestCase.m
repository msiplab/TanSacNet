classdef lsunAnalysis2dNetworkTestCase < matlab.unittest.TestCase

    methods (TestClassSetup)
        % テスト クラス全体の共有セットアップ
    end

    methods (TestMethodSetup)
        % 各テストのセットアップ
    end

    methods (Test)
        % テスト メソッド

        function unimplementedTest(testCase)
            testCase.verifyFail("Unimplemented test");
        end
    end

end