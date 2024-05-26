import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
#import torch_dct as dct_2d
import scipy.fftpack as fftpack

import math
from lsunBlockDct2dLayer import LsunBlockDct2dLayer
from lsunUtility import Direction

stride = [ [1, 1], [2, 2], [2, 4], [4, 1], [4, 4] ]
datatype = [ torch.float32, torch.float64 ]
height = [ 8, 16, 32 ]
width = [ 8, 16, 32 ]
isdevicetest = True

class LsunBlockDct2dLayerTestCase(unittest.TestCase):
    """
    LSUNBLOCKDCT2DLAYERTESTCASE
    
        ベクトル配列をブロック配列を入力:
          nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols) 
    
        コンポーネント別に出力:
          nSamples x nRows x nCols x nDecs
    
     Requirements: Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
     Copyright (c) 2024, Shogo MURAMATSU
    
     All rights reserved.
    
     Contact address: Shogo MURAMATSU,
                    Faculty of Engineering, Niigata University,
                    8050 2-no-cho Ikarashi, Nishi-ku,
                    Niigata, 950-2181, JAPAN
    
     http://www.eng.niigata-u.ac.jp/~msiplab/
    """

    @parameterized.expand(
        list(itertools.product(stride))
    )
    def testConstructor(self,stride):
        # Expected values
        expctdName = 'E0'
        expctdDescription = "Block DCT of size " \
            + str(stride[Direction.VERTICAL]) + "x" \
            + str(stride[Direction.HORIZONTAL])

        # Instantiation of target class
        layer = LsunBlockDct2dLayer(
                decimation_factor=stride,
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
        list(itertools.product(stride,height,width,datatype))
    )
    def testPredictGrayScale(self,
            stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")      

        # Parameters
        nSamples = 8
        nComponents = 1
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        ndecs =  stride[0]*stride[1] # math.prod(stride)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        #Y = dct.dct_2d(X.view(arrayshape),norm='ortho')
        Y = torch.tensor(fftpack.dct(fftpack.dct(X.cpu().view(arrayshape).detach().numpy(),axis=2,type=2,norm='ortho'),axis=1,type=2,norm='ortho'),dtype=datatype)
        Y = Y.to(device)
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        expctdZ = A.view(nSamples,nrows,ncols,ndecs)

        # Instantiation of target class
        layer = LsunBlockDct2dLayer(
                decimation_factor=stride,
                name='E0'
            )
            
        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)         
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertFalse(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype))
    )
    def testForwardGrayScale(self,
        stride, height, width, datatype):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")               
            
        # Parameters
        nSamples = 8
        nComponents = 1
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        ndecs = stride[0]*stride[1] # math.prod(stride)
        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        #Y = dct.dct_2d(X.view(arrayshape),norm='ortho')
        Y = torch.tensor(fftpack.dct(fftpack.dct(X.cpu().view(arrayshape).detach().numpy(),axis=2,type=2,norm='ortho'),axis=1,type=2,norm='ortho'),dtype=datatype)
        Y = Y.to(device)
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        expctdZ = A.view(nSamples,nrows,ncols,ndecs)

        # Instantiation of target class
        layer = LsunBlockDct2dLayer(
                decimation_factor=stride,
                name='E0'
            )
            
        # Actual values
        actualZ = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZ.dtype,datatype)         
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        self.assertTrue(actualZ.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype))
    )
    def testPredictRgbColor(self,
        stride, height, width, datatype):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")    

        # Parameters
        nSamples = 8
        nComponents = 3 # RGB
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)

        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        ndecs = stride[0]*stride[1] # math.prod(stride)

        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        #Y = dct.dct_2d(X.view(arrayshape),norm='ortho')
        Y = torch.tensor(fftpack.dct(fftpack.dct(X.cpu().view(arrayshape).detach().numpy(),axis=2,type=2,norm='ortho'),axis=1,type=2,norm='ortho'),dtype=datatype)
        Y = Y.to(device)
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        Z = A.view(nSamples,nComponents,nrows,ncols,ndecs)
        expctdZr = Z[:,0,:,:,:]
        expctdZg = Z[:,1,:,:,:]
        expctdZb = Z[:,2,:,:,:]

        # Instantiation of target class
        layer = LsunBlockDct2dLayer(
                decimation_factor=stride,
                number_of_components=nComponents,
                name='E0'
            )
            
        # Actual values
        with torch.no_grad():        
            actualZr,actualZg,actualZb = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZr.dtype,datatype)         
        self.assertEqual(actualZg.dtype,datatype)         
        self.assertEqual(actualZb.dtype,datatype)                 
        self.assertTrue(torch.allclose(actualZr,expctdZr,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualZg,expctdZg,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualZb,expctdZb,rtol=rtol,atol=atol))
        self.assertFalse(actualZr.requires_grad)
        self.assertFalse(actualZg.requires_grad)
        self.assertFalse(actualZb.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype))
    )
    def testForwardRgbColor(self,
        stride, height, width, datatype):
        rtol,atol=1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")      

        # Parameters
        nSamples = 8
        nComponents = 3 # RGB
        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)
        
        # Expected values
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        ndecs = stride[0]*stride[1] # math.prod(stride)

        # Block DCT (nSamples x nComponents x nrows x ncols) x decV x decH
        arrayshape = stride.copy()
        arrayshape.insert(0,-1)
        #Y = dct.dct_2d(X.view(arrayshape),norm='ortho')
        Y = torch.tensor(fftpack.dct(fftpack.dct(X.cpu().view(arrayshape).detach().numpy(),axis=2,type=2,norm='ortho'),axis=1,type=2,norm='ortho'),dtype=datatype)
        Y = Y.to(device)
        # Rearrange the DCT Coefs. (nSamples x nComponents x nrows x ncols) x (decV x decH)
        A = permuteDctCoefs_(Y)
        Z = A.view(nSamples,nComponents,nrows,ncols,ndecs)
        expctdZr = Z[:,0,:,:,:]
        expctdZg = Z[:,1,:,:,:]
        expctdZb = Z[:,2,:,:,:]

        # Instantiation of target class
        layer = LsunBlockDct2dLayer(
                decimation_factor=stride,
                number_of_components=nComponents,
                name='E0'
            )
            
        # Actual values
        actualZr,actualZg,actualZb = layer.forward(X)

        # Evaluation
        self.assertEqual(actualZr.dtype,datatype)         
        self.assertEqual(actualZg.dtype,datatype)         
        self.assertEqual(actualZb.dtype,datatype)                 
        self.assertTrue(torch.allclose(actualZr,expctdZr,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualZg,expctdZg,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualZb,expctdZb,rtol=rtol,atol=atol))
        self.assertTrue(actualZr.requires_grad)
        self.assertTrue(actualZg.requires_grad)
        self.assertTrue(actualZb.requires_grad)    

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype))
    )
    def testBackwardGrayScale(self,
        stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")               

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nComponents = 1

        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)      
        # nSamples x nRows x nCols x nDecs
        dLdZ = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype)
        dLdZ = dLdZ.to(device)

        # Expected values
        A = permuteIdctCoefs_(dLdZ,stride)
        #Y = dct.idct_2d(A,norm='ortho')
        Y = torch.tensor(fftpack.idct(fftpack.idct(A.detach().numpy(),axis=1,type=2,norm='ortho'),axis=2,type=2,norm='ortho'),dtype=datatype)
        Y = Y.to(device)
        expctddLdX = Y.reshape(nSamples,nComponents,height,width)
        
        # Instantiation of target class
        layer = LsunBlockDct2dLayer(
                decimation_factor=stride,
                name='E0'
            )
            
        # Actual values
        Z = layer.forward(X)
        Z.backward(dLdZ)
        actualdLdX = X.grad

        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(Z.requires_grad)

    @parameterized.expand(
        list(itertools.product(stride,height,width,datatype))
    )
    def testBackwardRgbColor(self,
        stride, height, width, datatype):
        rtol,atol = 1e-3,1e-6
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
        else:
            device = torch.device("cpu")              

        # Parameters
        nSamples = 8
        nrows = int(math.ceil(height/stride[Direction.VERTICAL])) #.astype(int)
        ncols = int(math.ceil(width/stride[Direction.HORIZONTAL])) #.astype(int)
        nDecs = stride[0]*stride[1] # math.prod(stride)
        nComponents = 3 # RGB

        # Source (nSamples x nComponents x (Stride[0]xnRows) x (Stride[1]xnCols))
        X = torch.rand(nSamples,nComponents,height,width,dtype=datatype,device=device,requires_grad=True)     
        # nSamples x nRows x nCols x nDecs
        dLdZr = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype)
        dLdZr = dLdZr.to(device)
        dLdZg = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype)
        dLdZg = dLdZg.to(device)
        dLdZb = torch.rand(nSamples,nrows,ncols,nDecs,dtype=datatype) 
        dLdZb = dLdZb.to(device)
    
        # Expected values
        Ar = permuteIdctCoefs_(dLdZr,stride)
        Ag = permuteIdctCoefs_(dLdZg,stride)        
        Ab = permuteIdctCoefs_(dLdZb,stride)                
        #Yr = dct.idct_2d(Ar,norm='ortho')
        Yr = torch.tensor(fftpack.idct(fftpack.idct(Ar.detach().numpy(),axis=1,type=2,norm='ortho'),axis=2,type=2,norm='ortho'),dtype=datatype)
        Yr = Yr.to(device)
        #Yg = dct.idct_2d(Ag,norm='ortho')
        Yg = torch.tensor(fftpack.idct(fftpack.idct(Ag.detach().numpy(),axis=1,type=2,norm='ortho'),axis=2,type=2,norm='ortho'),dtype=datatype)
        Yg = Yg.to(device)
        #Yb = dct.idct_2d(Ab,norm='ortho')
        Yb = torch.tensor(fftpack.idct(fftpack.idct(Ab.detach().numpy(),axis=1,type=2,norm='ortho'),axis=2,type=2,norm='ortho'),dtype=datatype)
        Yb = Yb.to(device)
        
        expctddLdX = torch.cat((
            Yr.reshape(nSamples,1,height,width),
            Yg.reshape(nSamples,1,height,width),
            Yb.reshape(nSamples,1,height,width)),dim=1)
        
        # Instantiation of target class
        layer = LsunBlockDct2dLayer(
                decimation_factor=stride,
                number_of_components=nComponents,                
                name='E0'
            )
            
        # Actual values
        Zr,Zg,Zb = layer.forward(X)
        Zr.backward(dLdZr,retain_graph=True)
        Zg.backward(dLdZg,retain_graph=True)
        Zb.backward(dLdZb,retain_graph=False)
        actualdLdX = X.grad

        # Evaluation
        self.assertEqual(actualdLdX.dtype,datatype)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(Zr.requires_grad)
        self.assertTrue(Zg.requires_grad)
        self.assertTrue(Zb.requires_grad)

def permuteDctCoefs_(x):
    cee = x[:,0::2,0::2].reshape(x.size(0),-1)
    coo = x[:,1::2,1::2].reshape(x.size(0),-1)
    coe = x[:,1::2,0::2].reshape(x.size(0),-1)
    ceo = x[:,0::2,1::2].reshape(x.size(0),-1)
    return torch.cat((cee,coo,coe,ceo),dim=-1)

def permuteIdctCoefs_(x,block_size):
    coefs = x.view(-1,block_size[Direction.VERTICAL]*block_size[Direction.HORIZONTAL]) # x.view(-1,math.prod(block_size)) 
    decY_ = block_size[Direction.VERTICAL]
    decX_ = block_size[Direction.HORIZONTAL]
    chDecY = int(math.ceil(decY_/2.)) #.astype(int)
    chDecX = int(math.ceil(decX_/2.)) #.astype(int)
    fhDecY = int(math.floor(decY_/2.)) #.astype(int)
    fhDecX = int(math.floor(decX_/2.)) #.astype(int)
    nQDecsee = chDecY*chDecX
    nQDecsoo = fhDecY*fhDecX
    nQDecsoe = fhDecY*chDecX
    cee = coefs[:,:nQDecsee]
    coo = coefs[:,nQDecsee:nQDecsee+nQDecsoo]
    coe = coefs[:,nQDecsee+nQDecsoo:nQDecsee+nQDecsoo+nQDecsoe]
    ceo = coefs[:,nQDecsee+nQDecsoo+nQDecsoe:]
    nBlocks = coefs.size(0)
    value = torch.zeros(nBlocks,decY_,decX_,dtype=x.dtype)
    value[:,0::2,0::2] = cee.view(nBlocks,chDecY,chDecX)
    value[:,1::2,1::2] = coo.view(nBlocks,fhDecY,fhDecX)
    value[:,1::2,0::2] = coe.view(nBlocks,fhDecY,chDecX)
    value[:,0::2,1::2] = ceo.view(nBlocks,chDecY,fhDecX)
    return value

if __name__ == '__main__':
    unittest.main()

"""
    properties (TestParameter)
        stride = { [1 1], [2 2], [2 4], [4 1], [4 4], [8 8] };
        datatype = { 'single', 'double' };
        height = struct('small', 8,'medium', 16, 'large', 32);
        width = struct('small', 8,'medium', 16, 'large', 32);
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            fprintf("\n --- Check layer for 2-D images ---\n");
            % Grayscale
            layer = lsunBlockDct2dLayer(...
                'Stride',[2 2]);
            checkLayer(layer,[8 8 1],'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)
            % RGB color
            layer = lsunBlockDct2dLayer(...
                'NumberOfComponents',3,...
                'Stride',[2 2]);            
            checkLayer(layer,[8 8 3],'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, stride)
            
            % Expected values
            expctdName = 'E0';
            expctdDescription = "Block DCT of size " ...
                + stride(1) + "x" + stride(2);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct2dLayer(...
                'Stride',stride,...
                'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        function testPredictGrayScale(testCase, ...
                stride, height, width, datatype)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nComponents = 1;
            X = rand(height,width,nComponents,nSamples, datatype);
            
            % Expected values
            nrows = ceil(height/stride(Direction.VERTICAL));
            ncols = ceil(width/stride(Direction.HORIZONTAL));
            ndecs = prod(stride);
            %expctdZ = zeros(nrows,ncols,ndecs,nSamples,datatype);
            expctdZ = zeros(ndecs,nrows,ncols,nSamples,datatype);
            for iSample = 1:nSamples
                % Block DCT
                Y = blockproc(X(:,:,nComponents,iSample),...
                    stride,@(x) dct2(x.data));
                % Rearrange the DCT Coefs.
                A = blockproc(Y,...
                    stride,@testCase.permuteDctCoefs_);
                expctdZ(:,:,:,iSample) = ...
                    ...permute(reshape(A,ndecs,nrows,ncols),[2 3 1]);
                    reshape(A,ndecs,nrows,ncols);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct2dLayer(...
                'Stride',stride,...
                'Name','E0');
            
            % Actual values
            actualZ = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
                
        function testForwardGrayScale(testCase, ...
                stride, height, width, datatype)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nComponents = 1;
            X = rand(height,width,nComponents,nSamples, datatype);
            
            % Expected values
            nrows = ceil(height/stride(Direction.VERTICAL));
            ncols = ceil(width/stride(Direction.HORIZONTAL));
            ndecs = prod(stride);
            %expctdZ = zeros(nrows,ncols,ndecs,nSamples,datatype);
            expctdZ = zeros(ndecs,nrows,ncols,nSamples,datatype);
            for iSample = 1:nSamples
                % Block DCT
                Y = blockproc(X(:,:,nComponents,iSample),...
                    stride,@(x) dct2(x.data));
                % Rearrange the DCT Coefs.
                A = blockproc(Y,...
                    stride,@testCase.permuteDctCoefs_);
                expctdZ(:,:,:,iSample) = ...
                    ...permute(reshape(A,ndecs,nrows,ncols),[2 3 1]);
                    reshape(A,ndecs,nrows,ncols);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct2dLayer(...
                'Stride',stride,...
                'Name','E0');
            
            % Actual values
            actualZ = layer.forward(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        
        function testPredictRgbColor(testCase, ...
                stride, height, width, datatype)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nComponents = 3; % RGB
            X = rand(height,width,nComponents,nSamples, datatype);
            
            % Expected values
            nrows = height/stride(Direction.VERTICAL);
            ncols = width/stride(Direction.HORIZONTAL);
            ndecs = prod(stride);
            %expctdZr = zeros(nrows,ncols,ndecs,nSamples,datatype);
            %expctdZg = zeros(nrows,ncols,ndecs,nSamples,datatype);
            %expctdZb = zeros(nrows,ncols,ndecs,nSamples,datatype);
            expctdZr = zeros(ndecs,nrows,ncols,nSamples,datatype);
            expctdZg = zeros(ndecs,nrows,ncols,nSamples,datatype);
            expctdZb = zeros(ndecs,nrows,ncols,nSamples,datatype);            
            for iSample = 1:nSamples
                % Block DCT
                Yr = blockproc(X(:,:,1,iSample),...
                    stride,@(x) dct2(x.data));
                Yg = blockproc(X(:,:,2,iSample),...
                    stride,@(x) dct2(x.data));
                Yb = blockproc(X(:,:,3,iSample),...
                    stride,@(x) dct2(x.data));
                % Rearrange the DCT Coefs.
                Ar = blockproc(Yr,...
                    stride,@testCase.permuteDctCoefs_);
                Ag = blockproc(Yg,...
                    stride,@testCase.permuteDctCoefs_);
                Ab = blockproc(Yb,...
                    stride,@testCase.permuteDctCoefs_);
                expctdZr(:,:,:,iSample) = ...
                    ...permute(reshape(Ar,ndecs,nrows,ncols),[2 3 1]);
                    reshape(Ar,ndecs,nrows,ncols);
                expctdZg(:,:,:,iSample) = ...
                    ...permute(reshape(Ag,ndecs,nrows,ncols),[2 3 1]);
                    reshape(Ag,ndecs,nrows,ncols);
                expctdZb(:,:,:,iSample) = ...
                    ...permute(reshape(Ab,ndecs,nrows,ncols),[2 3 1]);
                    reshape(Ab,ndecs,nrows,ncols);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct2dLayer(...
                'Stride',stride,...
                'NumberOfComponents',nComponents,...
                'Name','E0');
            
            % Actual values
            [actualZr,actualZg,actualZb] = layer.predict(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZr,datatype);
            testCase.verifyInstanceOf(actualZg,datatype);
            testCase.verifyInstanceOf(actualZb,datatype);
            testCase.verifyThat(actualZr,...
                IsEqualTo(expctdZr,'Within',tolObj));
            testCase.verifyThat(actualZg,...
                IsEqualTo(expctdZg,'Within',tolObj));
            testCase.verifyThat(actualZb,...
                IsEqualTo(expctdZb,'Within',tolObj));
            
        end
        
        function testForwardRgbColor(testCase, ...
                stride, height, width, datatype)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nComponents = 3; % RGB
            X = rand(height,width,nComponents,nSamples, datatype);
            
            % Expected values
            nrows = height/stride(Direction.VERTICAL);
            ncols = width/stride(Direction.HORIZONTAL);
            ndecs = prod(stride);
            %expctdZr = zeros(nrows,ncols,ndecs,nSamples,datatype);
            %expctdZg = zeros(nrows,ncols,ndecs,nSamples,datatype);
            %expctdZb = zeros(nrows,ncols,ndecs,nSamples,datatype);
            expctdZr = zeros(ndecs,nrows,ncols,nSamples,datatype);
            expctdZg = zeros(ndecs,nrows,ncols,nSamples,datatype);
            expctdZb = zeros(ndecs,nrows,ncols,nSamples,datatype);            
            for iSample = 1:nSamples
                % Block DCT
                Yr = blockproc(X(:,:,1,iSample),...
                    stride,@(x) dct2(x.data));
                Yg = blockproc(X(:,:,2,iSample),...
                    stride,@(x) dct2(x.data));
                Yb = blockproc(X(:,:,3,iSample),...
                    stride,@(x) dct2(x.data));
                % Rearrange the DCT Coefs.
                Ar = blockproc(Yr,...
                    stride,@testCase.permuteDctCoefs_);
                Ag = blockproc(Yg,...
                    stride,@testCase.permuteDctCoefs_);
                Ab = blockproc(Yb,...
                    stride,@testCase.permuteDctCoefs_);
                expctdZr(:,:,:,iSample) = ...
                    ...permute(reshape(Ar,ndecs,nrows,ncols),[2 3 1]);
                    reshape(Ar,ndecs,nrows,ncols);
                expctdZg(:,:,:,iSample) = ...
                    ...permute(reshape(Ag,ndecs,nrows,ncols),[2 3 1]);
                    reshape(Ag,ndecs,nrows,ncols);
                expctdZb(:,:,:,iSample) = ...
                    ...permute(reshape(Ab,ndecs,nrows,ncols),[2 3 1]);
                    reshape(Ab,ndecs,nrows,ncols);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct2dLayer(...
                'Stride',stride,...
                'NumberOfComponents',nComponents,...
                'Name','E0');
            
            % Actual values
            [actualZr,actualZg,actualZb] = layer.forward(X);
            
            % Evaluation
            testCase.verifyInstanceOf(actualZr,datatype);
            testCase.verifyInstanceOf(actualZg,datatype);
            testCase.verifyInstanceOf(actualZb,datatype);
            testCase.verifyThat(actualZr,...
                IsEqualTo(expctdZr,'Within',tolObj));
            testCase.verifyThat(actualZg,...
                IsEqualTo(expctdZg,'Within',tolObj));
            testCase.verifyThat(actualZb,...
                IsEqualTo(expctdZb,'Within',tolObj));
            
        end
        
        
        function testBackwardGrayScale(testCase, ...
                stride, height, width, datatype)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nrows = height/stride(Direction.VERTICAL);
            ncols = width/stride(Direction.HORIZONTAL);
            nDecs = prod(stride);
            nComponents = 1;
            %dLdZ = rand(nrows,ncols,nDecs,nSamples,datatype);
            dLdZ = rand(nDecs,nrows,ncols,nSamples,datatype);
            
            % Expected values
            expctddLdX = zeros(height,width,datatype);
            for iSample = 1:nSamples
                %A = reshape(permute(dLdZ(:,:,:,iSample),[3 1 2]),...
                %    nDecs*nrows,ncols);
                A = reshape(dLdZ(:,:,:,iSample),nDecs*nrows,ncols);                
                Y = blockproc(A,[nDecs 1],...
                    @(x) testCase.permuteIdctCoefs_(x.data,stride));
                expctddLdX(:,:,nComponents,iSample) = ...
                    blockproc(Y,...
                    stride,...
                    @(x) idct2(x.data));
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct2dLayer(...
                'Stride',stride,...
                'Name','E0');
            
            % Actual values
            actualdLdX = layer.backward([],[],dLdZ,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end
        
        function testBackwardRgbColor(testCase, ...
                stride, height, width, datatype)
            import tansacnet.utility.Direction
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-5,single(1e-5));
            
            % Parameters
            nSamples = 8;
            nrows = ceil(height/stride(Direction.VERTICAL));
            ncols = ceil(width/stride(Direction.HORIZONTAL));
            nDecs = prod(stride);
            nComponents = 3; % RGB
            %dLdZr = rand(nrows,ncols,nDecs,nSamples,datatype);
            %dLdZg = rand(nrows,ncols,nDecs,nSamples,datatype);
            %dLdZb = rand(nrows,ncols,nDecs,nSamples,datatype);
            dLdZr = rand(nDecs,nrows,ncols,nSamples,datatype);
            dLdZg = rand(nDecs,nrows,ncols,nSamples,datatype);
            dLdZb = rand(nDecs,nrows,ncols,nSamples,datatype);            
            
            % Expected values
            expctddLdX = zeros(height,width,nComponents,datatype);
            for iSample = 1:nSamples
                %Ar = reshape(permute(dLdZr(:,:,:,iSample),[3 1 2]),...
                %    nDecs*nrows,ncols);
                %Ag = reshape(permute(dLdZg(:,:,:,iSample),[3 1 2]),...
                %    nDecs*nrows,ncols);
                %Ab = reshape(permute(dLdZb(:,:,:,iSample),[3 1 2]),...
                %    nDecs*nrows,ncols);
                Ar = reshape(dLdZr(:,:,:,iSample),nDecs*nrows,ncols);
                Ag = reshape(dLdZg(:,:,:,iSample),nDecs*nrows,ncols);
                Ab = reshape(dLdZb(:,:,:,iSample),nDecs*nrows,ncols);                
                Yr = blockproc(Ar,[nDecs 1],...
                    @(x) testCase.permuteIdctCoefs_(x.data,stride));
                Yg = blockproc(Ag,[nDecs 1],...
                    @(x) testCase.permuteIdctCoefs_(x.data,stride));
                Yb = blockproc(Ab,[nDecs 1],...
                    @(x) testCase.permuteIdctCoefs_(x.data,stride));
                expctddLdX(:,:,1,iSample) = ...
                    blockproc(Yr,...
                    stride,...
                    @(x) idct2(x.data));
                expctddLdX(:,:,2,iSample) = ...
                    blockproc(Yg,...
                    stride,...
                    @(x) idct2(x.data));
                expctddLdX(:,:,3,iSample) = ...
                    blockproc(Yb,...
                    stride,...
                    @(x) idct2(x.data));
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunBlockDct2dLayer(...
                'Stride',stride,...
                'NumberOfComponents',nComponents,...
                'Name','E0');
            
            % Actual values
            actualdLdX = layer.backward([],[],[],[],dLdZr,dLdZg,dLdZb,[]);
            
            % Evaluation
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            
        end
        
    end
    
    methods (Static, Access = private)
        
        function value = permuteDctCoefs_(x)
            coefs = x.data;
            cee = coefs(1:2:end,1:2:end);
            coo = coefs(2:2:end,2:2:end);
            coe = coefs(2:2:end,1:2:end);
            ceo = coefs(1:2:end,2:2:end);
            value = [ cee(:) ; coo(:) ; coe(:) ; ceo(:) ];
        end
        function value = permuteIdctCoefs_(x,blockSize)
            import tansacnet.utility.Direction
            coefs = x;
            decY_ = blockSize(Direction.VERTICAL);
            decX_ = blockSize(Direction.HORIZONTAL);
            nQDecsee = ceil(decY_/2)*ceil(decX_/2);
            nQDecsoo = floor(decY_/2)*floor(decX_/2);
            nQDecsoe = floor(decY_/2)*ceil(decX_/2);
            cee = coefs(         1:  nQDecsee);
            coo = coefs(nQDecsee+1:nQDecsee+nQDecsoo);
            coe = coefs(nQDecsee+nQDecsoo+1:nQDecsee+nQDecsoo+nQDecsoe);
            ceo = coefs(nQDecsee+nQDecsoo+nQDecsoe+1:end);
            value = zeros(decY_,decX_,'like',x);
            value(1:2:decY_,1:2:decX_) = reshape(cee,ceil(decY_/2),ceil(decX_/2));
            value(2:2:decY_,2:2:decX_) = reshape(coo,floor(decY_/2),floor(decX_/2));
            value(2:2:decY_,1:2:decX_) = reshape(coe,floor(decY_/2),ceil(decX_/2));
            value(1:2:decY_,2:2:decX_) = reshape(ceo,ceil(decY_/2),floor(decX_/2));
        end
    end
    
end
"""
