import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
from torch_tansacnet.lsunChannelConcatenation2dLayer import LsunChannelConcatenation2dLayer

nchs = [ [3, 3], [4, 4], [32, 32] ]
datatype = [ torch.float32, torch.float64 ]
nrows = [ 4, 8, 16 ]
ncols = [ 4, 8, 16 ]
batch = [ 1, 8 ]

class LsunChannelConcatenation2dLayerTestCase(unittest.TestCase):
    """
    LSUNCHANNELCONCATENATION2DLAYERTESTCASE

        ２コンポーネント入力(nComponents=2のみサポート):
            nSamples x nRows x nCols x (nChsTotal-1)    
            nSamples x nRows x nCols

        １コンポーネント出力(nComponents=1のみサポート):
        nSamples x nRows x nCols x nChsTotal

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
        expctdName = 'Cn'
        expctdDescription = "Channel concatenation"
            
        # Instantiation of target class
        layer = LsunChannelConcatenation2dLayer(
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
        rtol,atol = 1e-5,1e-6
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Parameters        
        nSamples = batch
        nChsTotal = sum(nchs)

        # nSamples x nRows x nCols x (nChsTotal-1) 
        Xac = torch.randn(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype,device=device,requires_grad=True)
        # nSamples x nRows x nCols
        Xdc = torch.randn(nSamples,nrows,ncols,dtype=datatype,device=device,requires_grad=True)
            
        # Expected values
        # nSamples x nRows x nCols x nChsTotal
        expctdZ = torch.cat((Xdc.unsqueeze(dim=3),Xac),dim=3)
            
        # Instantiation of target class
        layer = LsunChannelConcatenation2dLayer(
            name='Cn'
            )
        
        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(Xac=Xac,Xdc=Xdc)
            
        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol)) 
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(nchs,nrows,ncols,batch,datatype))
    )
    def testBackward(self,
        nchs,nrows,ncols,batch,datatype):
        rtol,atol = 1e-5,1e-6
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Parameters
        nSamples = batch
        nChsTotal = sum(nchs)
        # nSamples x nRows x nCols x nChsTotalx
        Xac = torch.randn(nSamples,nrows,ncols,nChsTotal-1,dtype=datatype,device=device,requires_grad=True)
        # nSamples x nRows x nCols
        Xdc = torch.randn(nSamples,nrows,ncols,dtype=datatype,device=device,requires_grad=True)
        # nSamples x nRows x nCols x nChsTotal
        dLdZ = torch.randn(nSamples,nrows,ncols,nChsTotal,dtype=datatype)
        dLdZ = dLdZ.to(device)
            
        # Expected values
        # nSamples  x nRows x nCols x (nChsTotal-1) 
        expctddLdXac = dLdZ[:,:,:,1:]
        # nSamples x nRows x nCols
        expctddLdXdc = dLdZ[:,:,:,0]
            
        # Instantiation of target class
        layer = LsunChannelConcatenation2dLayer(
            name='Cn'
            )
            
        # Actual values
        Z = layer.forward(Xac=Xac,Xdc=Xdc)
        Z.backward(dLdZ)
        actualdLdXac = Xac.grad
        actualdLdXdc = Xdc.grad
            
        # Evaluation
        self.assertEqual(actualdLdXdc.dtype,datatype)
        self.assertEqual(actualdLdXac.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdXdc,expctddLdXdc,rtol=rtol,atol=atol)) 
        self.assertTrue(torch.allclose(actualdLdXac,expctddLdXac,rtol=rtol,atol=atol)) 
        self.assertTrue(Z.requires_grad)

if __name__ == '__main__':
    unittest.main()

"""
classdef lsunChannelConcatenation2dLayerTestCase < matlab.unittest.TestCase
    %NSOLTCHANNELCONCATENATION2DLAYERTESTCASE
    %
    %   ２コンポーネント入力(nComponents=2のみサポート):
    %      nRows x nCols x (nChsTotal-1) x nSamples    
    %      nRows x nCols x 1 x nSamples
    %
    %   １コンポーネント出力(nComponents=1のみサポート):
    %       nChsTotal x nRows x nCols xnSamples
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
            layer = lsunChannelConcatenation2dLayer();
            fprintf("\n --- Check layer for 2-D images ---\n");
            checkLayer(layer,{[8 8 5], [8 8 1]},...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase)
            
            % Expected values
            expctdName = 'Cn';
            expctdDescription = "Channel concatenation";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelConcatenation2dLayer('Name',expctdName);
            
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
            % nRows x nCols x (nChsTotal-1) x nSamples 
            Xac = randn(nrows,ncols,nChsTotal-1,nSamples,datatype);
            % nRows x nCols x 1 x nSamples
            Xdc = randn(nrows,ncols,1,nSamples,datatype);

            
            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            expctdZ = permute(cat(3,Xdc,Xac),[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelConcatenation2dLayer('Name','Cn');
            
            % Actual values
            actualZ = layer.predict(Xac,Xdc);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
                
        function testBackward(testCase,nchs,nrows,ncols,batch,datatype)
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = batch;
            nChsTotal = sum(nchs);
            % nChsTotal x nRows x nCols x nSamples
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            
            % Expected values
            % nRows x nCols x (nChsTotal-1) x nSamples 
            expctddLdXac = permute(dLdZ(2:end,:,:,:),[2 3 1 4]);
            % nRows x nCols x 1 x nSamples
            expctddLdXdc = permute(dLdZ(1,:,:,:),[2 3 1 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunChannelConcatenation2dLayer('Name','Cn');
            
            % Actual values
            [actualdLdXac,actualdLdXdc] = layer.backward([],[],[],dLdZ,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdXdc,datatype);
            testCase.verifyInstanceOf(actualdLdXac,datatype);            
            testCase.verifyThat(actualdLdXdc,...
                IsEqualTo(expctddLdXdc,'Within',tolObj));
            testCase.verifyThat(actualdLdXac,...
                IsEqualTo(expctddLdXac,'Within',tolObj));            
            
        end
        
    end
    
end

"""