import itertools
import unittest
from parameterized import parameterized
import math
import torch
from lsunIntermediateRotation2dLayer import LsunIntermediateRotation2dLayer
from lsunUtility import Direction

stride = [ [2, 2], [4, 4] ]
mus = [ 1, -1 ]
datatype = [ torch.float32, torch.float64 ]
nrows = [ 2, 4, 8 ]
ncols = [ 2, 4, 8 ]
usegpu = [ True, False ]

class LsunIntermediateRotation2dLayerTestCase(unittest.TestCase):
    """
    LSUNINTERMEDIATEROTATION2DLAYERTESTCASE 
    
        コンポーネント別に入力(nComponents=1のみサポート):
            nSamples x nRows x nCols x nChs

        コンポーネント別に出力(nComponents=1のみサポート):
            nSamples x nRows x nCols x nChs

        Requirements: Python 3.10/11.x, PyTorch 2.3.x

        Copyright (c) 2024, Shogo MURAMATSU

        All rights reserved.

        Contact address: Shogo MURAMATSU,
                        Faculty of Engineering, Niigata University,
                        8050 2-no-cho Ikarashi, Nishi-ku,
                        Niigata, 950-2181, JAPAN

        https://www.eng.niigata-u.ac.jp/~msiplab/
"""

    @parameterized.expand(
        itertools.product(stride)
        ) 
    def testConstructor(self, stride):
        # Expected values
        expctdName = 'Vn~'
        expctdMode = 'Synthesis'
        expctdDescription = "Synthesis LSUN intermediate rotation " \
            + "(ps,pa) = (" \
            + str(math.ceil(math.prod(stride)/2)) + "," + str(math.floor(math.prod(stride)/2)) + ")"
        
        # Instantiation of target class
        layer = LsunIntermediateRotation2dLayer(
            stride=stride,
            name=expctdName)
        
        # Actual values
        actualName = layer.name
        actualMode = layer.mode
        actualDescription = layer.description
                  
        # Evaluation
        self.assertEqual(actualName,expctdName)
        self.assertEqual(actualMode,expctdMode)
        self.assertEqual(actualDescription,expctdDescription)

    @parameterized.expand(
        itertools.product(usegpu,stride,nrows,ncols,mus,datatype)
        )
    def testForwardGrayscale(self, usegpu, stride, nrows, ncols, mus, datatype):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('No GPU device was detected.')
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-5, 1e-6

        # Parameters
        nSamples = 8
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]

        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,nDecs,device=device,dtype=datatype)

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        UnT = mus*torch.eye(pa,device=device,dtype=datatype).repeat(nrows*ncols,1,1)
        expctdZ = torch.zeros_like(X)
        for iSample in range(nSamples):
            Xi = X[iSample,:,:,:].clone()
            Ys = Xi[:,:,:ps].view(-1,ps)            
            Ya = Xi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ya[iblk,:] = UnT[iblk,:,:] @ Ya[iblk,:]
            Yi = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            expctdZ[iSample,:,:,:] = Yi

        # Instantiation of target class
        layer = LsunIntermediateRotation2dLayer(
            dtype=datatype,
            device=device,
            stride=stride,
            number_of_blocks=[nrows,ncols],
            name='Vn~')
        
        # Actual values
        with torch.no_grad():
            layer.mus = mus
            actualZ = layer.forward(X)

        # Evaluation
        self.assertIsInstance(actualZ,torch.Tensor)
        self.assertEqual(actualZ.shape,expctdZ.shape)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

"""
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);                
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        function testPredictGrayscaleWithRandomAngles(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            UnT = permute(genU.step(angles,mus),[2 1 3]);
            Y = X; %permute(X,[3 1 2 4]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            Za = zeros(size(Ya),'like',Ya);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Za(:,iblk,iSample) = UnT(:,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~');
            
            % Actual values
            layer.Mus = mus;
            layer.Angles = angles;
            actualZ = layer.predict(X);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));

        end
        
        function testPredictGrayscaleAnalysisMode(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem();

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            Un = genU.step(angles,mus);
            Y = X; % permute(X,[3 1 2 4]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            Za = zeros(size(Ya),'like',Ya);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Za(:,iblk,iSample) = Un(:,:,iblk)*Ya(:,iblk,iSample);
                end
            end            
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            expctdDescription = "Analysis LSUN intermediate rotation " ...
                + "(ps,pa) = (" ...
                + ps + "," + pa + ")";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn',...
                'Mode','Analysis');
            
            % Actual values
            layer.Mus = mus;
            layer.Angles = angles;
            actualZ = layer.predict(X);
            actualDescription = layer.Description;
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            testCase.verifyEqual(actualDescription,expctdDescription);

        end

        function testBackwardGrayscale(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = zeros(nAngles,nrows*ncols,datatype);
            
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);            
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);            
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            Un = genU.step(angles,mus,0);
            adLd_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols)
                    cdLd_low(:,iblk,iSample) = Un(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(ps+1:ps+pa,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nSamples);
            expctddLdX = adLd_; %ipermute(adLd_,[3 1 2 4]);           
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nrows*ncols,datatype);
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = 1:nAngles
                dUn_T = permute(genU.step(angles,mus,iAngle),[2 1 3]);
                for iblk = 1:(nrows*ncols)
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn_T(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~');
            layer.Mus = mus;
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));            
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));                        
        end

        function testBackwardGrayscaleWithRandomAngles(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)
    
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols);
                 
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);            
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);            
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);            
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            Un = genU.step(angles,mus,0);
            adLd_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols)
                    cdLd_low(:,iblk,iSample) = Un(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(ps+1:ps+pa,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nSamples);
            expctddLdX = adLd_; %ipermute(adLd_,[3 1 2 4]);           
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nrows*ncols,datatype);
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = 1:nAngles
                dUn_T = permute(genU.step(angles,mus,iAngle),[2 1 3]);
                for iblk = 1:(nrows*ncols)
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn_T(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~');
            layer.Mus = mus;
            layer.Angles = angles;
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));            
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));                             
        end
        
        function testBackwardGrayscaleAnalysisMode(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols);
            
            % nChsTotal x nRows x nCols xnSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            UnT = permute(genU.step(angles,mus,0),[2 1 3]);
            adLd_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols)
                    cdLd_low(:,iblk,iSample) = UnT(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(ps+1:ps+pa,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nSamples);
            expctddLdX = adLd_; %ipermute(adLd_,[3 1 2 4]);           
            
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nrows*ncols,datatype);
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = 1:nAngles
                dUn = genU.step(angles,mus,iAngle);
                for iblk = 1:(nrows*ncols)
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn',...
                'Mode','Analysis');
            layer.Mus = mus;
            layer.Angles = angles;
            %expctdZ = layer.predict(X);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);
            end
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));
        end

    end

    
end
"""

if __name__ == '__main__':
    unittest.main()

"""
classdef lsunIntermediateRotation2dLayerTestCase < matlab.unittest.TestCase
    %LSUNINTERMEDIATEROTATION2DLAYERTESTCASE 
    %   
    %   コンポーネント別に入力(nComponents)
    %      nChs x nRows x nCols x nSamples
    %
    %   コンポーネント別に出力(nComponents):
    %      nChs x nRows x nCols x nSamples
    %
    % Requirements: MATLAB R2022a
    %
    % Copyright (c) 2022, Shogo MURAMATSU
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
        stride = { [2 2], [4 4] };
        datatype = { 'single', 'double' };
        mus = { -1, 1 };
        nrows = struct('small', 2,'medium', 4, 'large', 8);
        ncols = struct('small', 2,'medium', 4, 'large', 8);
        usegpu = struct( 'true', true, 'false', false);        
    end

    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',[2 2],...
                'NumberOfBlocks',[8 8]);
            fprintf("\n --- Check layer for 2-D images ---\n");
            checkLayer(layer,[4 8 8],...
                'ObservationDimension',4,...
                'CheckCodegenCompatibility',true)      
        end
    end
    
    methods (Test)
        
        function testConstructor(testCase, stride)
            
            % Expected values
            expctdName = 'Vn~';
            expctdMode = 'Synthesis';
            expctdDescription = "Synthesis LSUN intermediate rotation " ...
                + "(ps,pa) = (" ...
                + ceil(prod(stride)/2) + "," + floor(prod(stride)/2) + ")";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualMode = layer.Mode;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualMode,expctdMode);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
        
        function testPredictGrayscale(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end

            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));        
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            % nChsTotal x nRows x nCols xnSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end
            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            UnT = repmat(mus*eye(pa,datatype),[1 1 nrows*ncols]);
            Y = X; %permute(X,[3 1 2 4]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            Za = zeros(size(Ya),'like',Ya);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Za(:,iblk,iSample) = UnT(:,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~');
            
            % Actual values
            layer.Mus = mus;
            actualZ = layer.predict(X);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);                
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            
        end
        
        function testPredictGrayscaleWithRandomAngles(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            UnT = permute(genU.step(angles,mus),[2 1 3]);
            Y = X; %permute(X,[3 1 2 4]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            Za = zeros(size(Ya),'like',Ya);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Za(:,iblk,iSample) = UnT(:,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~');
            
            % Actual values
            layer.Mus = mus;
            layer.Angles = angles;
            actualZ = layer.predict(X);

            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));

        end
        
        function testPredictGrayscaleAnalysisMode(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem();

            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            Un = genU.step(angles,mus);
            Y = X; % permute(X,[3 1 2 4]);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            Za = zeros(size(Ya),'like',Ya);
            for iSample=1:nSamples
                for iblk = 1:(nrows*ncols)
                    Za(:,iblk,iSample) = Un(:,:,iblk)*Ya(:,iblk,iSample);
                end
            end            
            Y(ps+1:ps+pa,:,:,:) = reshape(Za,pa,nrows,ncols,nSamples);
            expctdZ = Y; %ipermute(Y,[3 1 2 4]);
            expctdDescription = "Analysis LSUN intermediate rotation " ...
                + "(ps,pa) = (" ...
                + ps + "," + pa + ")";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn',...
                'Mode','Analysis');
            
            % Actual values
            layer.Mus = mus;
            layer.Angles = angles;
            actualZ = layer.predict(X);
            actualDescription = layer.Description;
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualZ,'gpuArray')
                actualZ = gather(actualZ);
                expctdZ = gather(expctdZ);
            end
            testCase.verifyInstanceOf(actualZ,datatype);
            testCase.verifyThat(actualZ,...
                IsEqualTo(expctdZ,'Within',tolObj));
            testCase.verifyEqual(actualDescription,expctdDescription);

        end

        function testBackwardGrayscale(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = zeros(nAngles,nrows*ncols,datatype);
            
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);            
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);            
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            Un = genU.step(angles,mus,0);
            adLd_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols)
                    cdLd_low(:,iblk,iSample) = Un(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(ps+1:ps+pa,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nSamples);
            expctddLdX = adLd_; %ipermute(adLd_,[3 1 2 4]);           
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nrows*ncols,datatype);
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = 1:nAngles
                dUn_T = permute(genU.step(angles,mus,iAngle),[2 1 3]);
                for iblk = 1:(nrows*ncols)
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn_T(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~');
            layer.Mus = mus;
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));            
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));                        
        end

        function testBackwardGrayscaleWithRandomAngles(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)
    
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols);
                 
            % nChsTotal x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);            
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);            
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);            
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            Un = genU.step(angles,mus,0);
            adLd_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols)
                    cdLd_low(:,iblk,iSample) = Un(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(ps+1:ps+pa,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nSamples);
            expctddLdX = adLd_; %ipermute(adLd_,[3 1 2 4]);           
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nrows*ncols,datatype);
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = 1:nAngles
                dUn_T = permute(genU.step(angles,mus,iAngle),[2 1 3]);
                for iblk = 1:(nrows*ncols)
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn_T(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn~');
            layer.Mus = mus;
            layer.Angles = angles;
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);                
            end
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);            
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));            
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));                             
        end
        
        function testBackwardGrayscaleAnalysisMode(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nChsTotal = prod(stride);
            nAngles = (nChsTotal-2)*nChsTotal/8;
            angles = randn((nChsTotal-2)*nChsTotal/8,nrows*ncols);
            
            % nChsTotal x nRows x nCols xnSamples
            %X = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,nChsTotal,nSamples,datatype);
            X = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nChsTotal,nrows,ncols,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                dLdZ = gpuArray(dLdZ);
                angles = gpuArray(angles);
            end

            % Expected values
            % nChsTotal x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            UnT = permute(genU.step(angles,mus,0),[2 1 3]);
            adLd_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            cdLd_low = reshape(adLd_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols)
                    cdLd_low(:,iblk,iSample) = UnT(:,:,iblk)*cdLd_low(:,iblk,iSample);
                end
            end
            adLd_(ps+1:ps+pa,:,:,:) = reshape(cdLd_low,pa,nrows,ncols,nSamples);
            expctddLdX = adLd_; %ipermute(adLd_,[3 1 2 4]);           
            
            
            % dLdWi = <dLdZ,(dVdWi)X>
            expctddLdW = zeros(nAngles,nrows*ncols,datatype);
            c_low = reshape(X(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            dldz_low = reshape(dLdZ(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = 1:nAngles
                dUn = genU.step(angles,mus,iAngle);
                for iblk = 1:(nrows*ncols)
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    c_low_iblk = dUn(:,:,iblk)*c_low_iblk;
                    dldz_iblk = squeeze(dldz_low(:,iblk,:));
                    expctddLdW(iAngle,iblk) = sum(dldz_iblk.*c_low_iblk,'all');
                end
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunIntermediateRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','Vn',...
                'Mode','Analysis');
            layer.Mus = mus;
            layer.Angles = angles;
            %expctdZ = layer.predict(X);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                testCase.verifyClass(actualdLdW,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                actualdLdW = gather(actualdLdW);
                expctddLdW = gather(expctddLdW);
            end
            testCase.verifyInstanceOf(actualdLdX,datatype);
            testCase.verifyInstanceOf(actualdLdW,datatype);
            testCase.verifyThat(actualdLdX,...
                IsEqualTo(expctddLdX,'Within',tolObj));
            testCase.verifyThat(actualdLdW,...
                IsEqualTo(expctddLdW,'Within',tolObj));
        end

    end

    
end
"""