function testRes = mytest(testCaseStr,isProfiling)
%MYTEST Script of unit testing for TanSacNet package
%
% This test script works with unit testing framework 
% See the following site:
%
% http://www.mathworks.co.jp/jp/help/matlab/matlab-unit-test-framework.html
%
% Requirements: MATLAB R2022a
%
% Copyright (c) 2022, Shogo MURAMATSU
%
% All rights reserved.
%
% Contact address: Shogo MURAMATSU,
%    Faculty of Engineering, Niigata University,
%    8050 2-no-cho Ikarashi, Nishi-ku,
%    Niigata, 950-2181, JAPAN
%
% http://msiplab.eng.niigata-u.ac.jp/
%

%%
import matlab.unittest.TestSuite

if nargin > 0
    isTestCase = true;
else
    clear classes %#ok
    isTestCase = false;
end

if nargin < 2
    isProfiling = false;
end

%% Package list
packageList = { ...
    'tansacnet.testcase.lsun',...
    'tansacnet.testcase.utility'
    };

%% Set path
setpath

%% Run test cases
if isProfiling
    profile on
end
if isTestCase
    testCase = eval(testCaseStr);
    testRes = run(testCase);
else
    testRes = cell(length(packageList),2);
    for idx = 1:length(packageList)
        if verLessThan('matlab','8.2.0.701') && ...
                strcmp(packageList{idx},'saivdr.testcase.embedded')
            disp('Package +embedded is available for R2013b or later.')
        else
            packageSuite = TestSuite.fromPackage(packageList{idx});
            testRes{idx,1} = packageList{idx};
            testRes{idx,2} = run(packageSuite);
        end
    end
end
if isProfiling
    profile off
    profile viewer
end

%% License check
license('inuse')

