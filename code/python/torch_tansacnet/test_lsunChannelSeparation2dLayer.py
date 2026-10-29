import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
from torch_tansacnet.lsunChannelSeparation2dLayer import LsunChannelSeparation2dLayer

nchs = [ [3, 3], [4, 4], [32, 32] ]
datatype = [ torch.float32, torch.float64 ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]
batch = [ 1, 8 ]

class LsunChannelSeparation2dLayerTestCase(unittest.TestCase):
    """
    LSUNCHANNELSEPARATION2DLAYERTESTCASE

        １コンポーネント入力(nComponents=1のみサポート):
            nSamples x nRows x nCols x nChsTotal 

        ２コンポーネント出力(nComponents=2のみサポート):
            nSamples x nRows x nCols x (nChsTotal-1) 
            nSamples x nRows x nCols
    
    Requirements: Python 3.10/11.x, PyTorch 2.3.x

    Copyright (c) 2024, Shogo MURAMATSU

    All rights reserved.

    Contact address: Shogo MURAMATSU,
                    Faculty of Engineering, Niigata University,
                    8050 2-no-cho Ikarashi, Nishi-ku,
                    Niigata, 950-2181, JAPAN

    https://www.eng.niigata-u.ac.jp/~msiplab/
    """

    def testConstructor(self):        
        # Expected values
        expctdName = 'Sp'
        expctdDescription = "Channel separation"
            
        # Instantiation of target class
        layer = LsunChannelSeparation2dLayer(
                name=expctdName
            )
            
        # Actual values
        actualName = layer.name
        actualDescription = layer.description
            
        # Evaluation
        self.assertTrue(isinstance(layer, nn.Module))                
        self.assertEqual(actualName,expctdName)    
        self.assertEqual(actualDescription,expctdDescription)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,batch,datatype))
    )
    def testPredict(self,
        nchs,nrows,ncols,batch,datatype):
        rtol,atol=1e-5,1e-8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Parameters
        nSamples = batch
        nChsTotal = sum(nchs)

        # nSamplex s nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        # nSamples x nRows x nCols x (nChsTotal-1)
        expctdZac = X[:,:,:,1:]
        # nSamples x nRows x nCols
        expctdZdc = X[:,:,:,0]

        # Instantiation of target class
        layer = LsunChannelSeparation2dLayer(
            name='Sp'
        )

        # Actual values
        with torch.no_grad():
            actualZac, actualZdc = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZac.dtype,datatype)
        self.assertEqual(actualZdc.dtype,datatype)
        self.assertTrue(torch.allclose(actualZac,expctdZac,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualZdc,expctdZdc,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,batch,datatype))
    )
    def testBackward(self,
                     nchs,nrows,ncols,batch,datatype):
        rtol,atol = 1e-5,1e-8
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Parameters
        nSamples = batch
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nChsTotal
        X = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype,device=device,requires_grad=True)
        # nSamples x nRows x nCols x (nChsTotal-1)
        dLdZac = torch.randn(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype)
        dLdZac = dLdZac.to(device)
        # nSamples x nRows x nCols
        dLdZdc = torch.randn(nSamples,nrows,ncols,dtype=datatype)
        dLdZdc = dLdZdc.to(device)

        # Expected values
        # nSamples  x nRows x nCols x nChsTotal
        expctedLdX = torch.cat((dLdZdc.unsqueeze(dim=3),dLdZac),dim=3)

        # Instantiation of target class
        layer = LsunChannelSeparation2dLayer(
            name='Sp'
        )

        # Actual values
        Zac,Zdc = layer.forward(X)
        Zac.backward(dLdZac,retain_graph=True)
        Zdc.backward(dLdZdc,retain_graph=False)
        actualdLdX = X.grad

        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctedLdX,rtol=rtol,atol=atol))
        self.assertTrue(Zac.requires_grad)
        self.assertTrue(Zdc.requires_grad)

if __name__ == '__main__':
    unittest.main()

"""
classdef lsunChannelSeparation2dLayerTestCase < matlab.unittest.TestCase
    %NSOLTCHANNELSEPARATION2DLAYERTESTCASE
    %
    %   １コンポーネント入力(nComponents=1のみサポート):
    %      nChsTotal x nRows x nCols x nSamples
    %
    %   ２コンポーネント出力(nComponents=2のみサポート):
    %      nRows x nCols x 1 x nSamples
    %      nRows x nCols x (nChsTotal-1) x nSamples    
    %
    % Requirements: MATLAB R2020b
    %
    % Copyright (c) 2020-2021, Shogo MURAMATSU
    %
    % All rights reserved.
    %
    % Contact address: Shogo MURAMATSU,
    %                Faculty of Engineering, Niigata University,
    %                8050 2-no-cho Ikarashi, Nishi-ku,
    %                Niigata, 950-2181, JAPAN
    %
    % http://msiplab.eng.niigata-u.ac.jp/
    
    properties (TestParameter)
        nchs = { [3 3], [4 4], [32 32] };
        datatype = { 'single', 'double' };
        nrows = struct('small', 1,'medium', 4, 'large', 16);
        ncols = struct('small', 1,'medium', 4, 'large', 16);
        batch = { 1, 8 };
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunChannelSeparation2dLayer();
            fprintf("\n --- Check layer for 2-D images ---\n");
            checkLayer(layer,[6 8 8],...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase)
            
            % Expected values
            expctdName = 'Sp';
            expctdDescription = "Channel separation";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelSeparation2dLayer('Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);    
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        function testPredict(testCase,nchs,nrows,ncols,batch,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = batch;
            nChsTotal = sum(nchs);
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            
            % Expected values
            % nRows x nCols x (nChsTotal-1) x nSamples 
            %expctdZ2 = X(:,:,2:end,:);
            expctdZac = permute(X(2:end,:,:,:),[2 3 1 4]);
            % nRows x nCols x 1 x nSamples
            %expctdZ1 = X(:,:,1,:);
            expctdZdc = permute(X(1,:,:,:),[2 3 1 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelSeparation2dLayer('Name','Sp');
            
            % Actual values
            [actualZac,actualZdc] = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZdc,datatype);
            testCase.verifyInstanceOf(actualZac,datatype);            
            testCase.verifyThat(actualZdc,...
                IsEqualTo(expctdZdc,'Within',tolObj));
            testCase.verifyThat(actualZac,...
                IsEqualTo(expctdZac,'Within',tolObj));            
            
        end

        function testBackward(testCase,nchs,nrows,ncols,batch,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = batch;
            nChsTotal = sum(nchs);
            % (nChsTotal-1) x nRows x nCols x nSamples 
            dLdZac = randn(nrows,ncols,nChsTotal-1,nSamples,datatype);
            % 1 x nRows x nCols x nSamples
            dLdZdc = randn(nrows,ncols,1,nSamples,datatype);
            
            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            %expctddLdX = cat(3,dLdZ1,dLdZ2);
            expctddLdX = ipermute(cat(3,dLdZdc,dLdZac),[2 3 1 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelSeparation2dLayer('Name','Sp');
            
            % Actual values
            actualdLdX = layer.backward([],[],[],dLdZac,dLdZdc,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end
    end
    
end
"""