import itertools
import unittest
from parameterized import parameterized
import math
import torch
#import torch.nn as nn
from torch_tansacnet.lsunInitialRotation2dLayer import LsunInitialRotation2dLayer
from torch_tansacnet.lsunUtility import Direction,OrthonormalMatrixGenerationSystem

stride = [ [2, 2], [4, 4] ]
mus = [ 1, -1 ]
datatype = [ torch.float32, torch.float64 ]
nrows = [ 2, 4, 8 ]
ncols = [ 2, 4, 8 ]
usegpu = [ True, False ]

class LsunInitialRotation2dLayerTestCase(unittest.TestCase):
    """
    LSUNINITIALROTATION2DLAYERTESTCASE

        コンポーネント別に入力(nComponents=1のみサポート):
            nSamples x nRows x nCols x nChs

        コンポーネント別に出力(nComponents=1のみサポート):
            nSamples x nRows x nCols x nChs

        Requirements: Python 3.10-12.x, PyTorch 2.3/4.x

        Copyright (c) 2024, Shogo MURAMATSU, Yasas GODAGE

        All rights reserved.

        Contact address: Shogo MURAMATSU,
                        Faculty of Engineering, Niigata University,
                        8050 2-no-cho Ikarashi, Nishi-ku,
                        Niigata, 950-2181, JAPAN

        https://www.eng.niigata-u.ac.jp/~msiplab/
    """

    @parameterized.expand(
            itertools.product(stride,nrows,ncols)
            )
    def testConstructor(self, stride,nrows,ncols):
        # Expected values
        expctdName = 'V0'
        expctdDescription = "LSUN initial rotation " \
            + "(ps,pa) = (" \
            + str(math.ceil(stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]/2)) + "," \
            + str(math.floor(stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]/2)) + "), "  \
            + "(mv,mh) = (" \
            + str(stride[Direction.VERTICAL]) + "," + str(stride[Direction.HORIZONTAL]) + ")"

        # Instantiation of target class
        layer = LsunInitialRotation2dLayer(
            stride=stride,
            number_of_blocks=[nrows,ncols],
            name=expctdName)

        # Actual values
        actualName = layer.name
        actualDescription = layer.description

        # Evaluation
        self.assertEqual(actualName,expctdName)
        self.assertEqual(actualDescription,expctdDescription)

    @parameterized.expand(
            itertools.product(usegpu,stride,nrows,ncols,mus,datatype)
            )
    def testForwardGrayscale(self, usegpu, stride, nrows, ncols, mus, datatype):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")     
        rtol, atol = 1e-5, 1e-6

        # Parameters
        nSamples = 8
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]
        
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device)

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        W0 = mus*torch.eye(ps,dtype=datatype,device=device).repeat(nrows*ncols,1,1)
        U0 = mus*torch.eye(pa,dtype=datatype,device=device).repeat(nrows*ncols,1,1)
        expctdZ = torch.zeros_like(X)
        for iSample in range(nSamples):
            Xi = X[iSample,:,:,:].clone()
            Ys = Xi[:,:,:ps].view(-1,ps)
            Ya = Xi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0[iblk,:,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0[iblk,:,:] @ Ya[iblk,:]
            Yi = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            expctdZ[iSample,:,:,:] = Yi

        # Instantiation of target class
        layer = LsunInitialRotation2dLayer(
            dtype=datatype,
            device=device,
            stride=stride,
            number_of_blocks=[nrows,ncols],
            name='V0')
        
        # Actual values
        with torch.no_grad():
            layer.mus = mus
            actualZ = layer.forward(X)

        # Evaluation
        self.assertIsInstance(actualZ,torch.Tensor)
        self.assertEqual(actualZ.shape,expctdZ.shape)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
    
    @parameterized.expand(
            itertools.product(usegpu,stride,nrows,ncols,mus,datatype)
            )
    def testForwardGrayscaleMus(self, usegpu, stride, nrows, ncols, mus, datatype):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")     
        rtol, atol = 1e-5, 1e-6

        # Parameters
        nSamples = 8
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]
        
        # nSamples x nRows x nCols x nChs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device)

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        W0 = mus*torch.eye(ps,dtype=datatype,device=device).repeat(nrows*ncols,1,1)
        U0 = mus*torch.eye(pa,dtype=datatype,device=device).repeat(nrows*ncols,1,1)
        expctdZ = torch.zeros_like(X)
        for iSample in range(nSamples):
            Xi = X[iSample,:,:,:].clone()
            Ys = Xi[:,:,:ps].view(-1,ps)
            Ya = Xi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0[iblk,:,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0[iblk,:,:] @ Ya[iblk,:]
            Yi = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            expctdZ[iSample,:,:,:] = Yi

        # Instantiation of target class
        layer = LsunInitialRotation2dLayer(
            dtype=datatype,
            device=device,
            stride=stride,
            number_of_blocks=[nrows,ncols],
            mus=mus,
            name='V0')
        
        # Actual values
        with torch.no_grad():
            actualZ = layer.forward(X)

        # Evaluation
        self.assertIsInstance(actualZ,torch.Tensor)
        self.assertEqual(actualZ.shape,expctdZ.shape)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))    
    
    @parameterized.expand(
            itertools.product(usegpu,stride,nrows,ncols,mus,datatype)
            )
    def testForwardGrayscaleWithRandomAngles(self, usegpu, stride, nrows, ncols, mus, datatype):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")
                return
        else:
            device = torch.device("cpu")        
        rtol, atol = 1e-5, 1e-6

        genW = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)
        genU = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)

        # Parameters
        nSamples = 8
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]

        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device)
        nAngles = (nDecs-2)*nDecs//4
        angles = torch.randn(nrows*ncols,nAngles,dtype=datatype,device=device) 

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        nAnglesH = nAngles//2
        W0 = mus*genW(angles=angles[:,:nAnglesH]) 
        U0 = mus*genU(angles=angles[:,nAnglesH:])

        expctdZ = torch.zeros_like(X)
        for iSample in range(nSamples):
            Xi = X[iSample,:,:,:].clone()
            Ys = Xi[:,:,:ps].view(-1,ps)
            Ya = Xi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0[iblk,:,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0[iblk,:,:] @ Ya[iblk,:]
            Yi = torch.cat((Ys,Ya),1).view(nrows,ncols,nDecs)
            expctdZ[iSample,:,:,:] = Yi

        # Instantiation of target class
        layer = LsunInitialRotation2dLayer(
            dtype=datatype,
            device=device,
            stride=stride,
            number_of_blocks=[nrows,ncols],
            name='V0')
        
        # Actual values
        with torch.no_grad():
            layer.mus = mus
            layer.angles = angles
            actualZ = layer.forward(X)

        # Evaluation
        self.assertIsInstance(actualZ,torch.Tensor)
        self.assertEqual(actualZ.shape,expctdZ.shape)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
    
    @parameterized.expand(
            itertools.product(usegpu,stride,nrows,ncols,mus,datatype)
            )
    def testForwardGrayscaleWithRandomAnglesNoDcLeackage(self, usegpu, stride, nrows, ncols, mus, datatype):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")
                return
        else:
            device = torch.device("cpu")        
        rtol, atol = 1e-5, 1e-6

        genW = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)
        genU = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype)

        # Parameters
        nSamples = 8
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]

        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device)
        nAngles = (nDecs-2)*nDecs//4
        angles = torch.randn(nrows*ncols,nAngles,dtype=datatype,device=device) 

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        nAnglesH = nAngles//2
        anglesNoDc = angles.clone()
        anglesNoDc[:,:(ps-1)] = 0
        
        musW = mus*torch.ones(nrows*ncols,ps,dtype=datatype,device=device)
        musW[:,0] = 1
        musU = mus*torch.ones(nrows*ncols,pa,dtype=datatype,device=device)
        W0 = genW(angles=anglesNoDc[:,:nAnglesH],mus=musW)
        U0 = genU(angles=anglesNoDc[:,nAnglesH:],mus=musU)

        expctdZ = torch.zeros_like(X)
        for iSample in range(nSamples):
            Xi = X[iSample,:,:,:].clone()
            Ys = Xi[:,:,:ps].view(-1,ps)
            Ya = Xi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0[iblk,:,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0[iblk,:,:] @ Ya[iblk,:]
            Yi = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            expctdZ[iSample,:,:,:] = Yi

        # Instantiation of target class
        layer = LsunInitialRotation2dLayer(
            dtype=datatype,
            device=device,
            stride=stride,
            number_of_blocks=[nrows,ncols],
            no_dc_leakage=True,
            name='V0')
        
        # Actual values
        with torch.no_grad():
            layer.mus = mus
            layer.angles = angles
            actualZ = layer.forward(X)

        # Evaluation
        self.assertIsInstance(actualZ,torch.Tensor)
        self.assertEqual(actualZ.shape,expctdZ.shape)
        message = 'usegpu=%s, stride=%s, nrows=%d, ncols=%d, mus=%f, datatype=%s' % (usegpu,stride,nrows,ncols,mus,datatype)
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol),msg=message)
    
    @parameterized.expand(
            itertools.product(usegpu,stride,nrows,ncols,mus,datatype)
            )
    def testBackwardGrayscale(self, usegpu, stride, nrows, ncols, mus, datatype):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-5

        genW = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype,partial_difference=True,mode='normal')
        genU = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype,partial_difference=True,mode='normal')

        # Parameters
        nSamples = 8
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]
        nAnglesH = (nDecs-2)*nDecs//8
        anglesW = torch.zeros(nrows*ncols,nAnglesH,dtype=datatype,device=device)
        anglesU = torch.zeros(nrows*ncols,nAnglesH,dtype=datatype,device=device)

        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)        
        dLdZ = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device)

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        W0T = genW(angles=anglesW,mus=mus,index_pd_angle=None).transpose(1,2)
        U0T = genU(angles=anglesU,mus=mus,index_pd_angle=None).transpose(1,2)
        Y = dLdZ.clone()
        expctddLdX = torch.empty_like(X) 
        for iSample in range(nSamples):
            Yi = Y[iSample,:,:,:]
            Ys = Yi[:,:,:ps].view(-1,ps)
            Ya = Yi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0T[iblk,:,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0T[iblk,:,:] @ Ya[iblk,:]
            Zsai = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            expctddLdX[iSample] = Zsai
        
        # dLdWi = <dLdZ,(dVdWi)X>
        nblks = nrows*ncols
        dldw_ = torch.empty(nblks,2*nAnglesH,dtype=datatype,device=device)
        dldz_ = dLdZ.clone()
        dldz_upp = dldz_[:,:,:,:ps].view(nSamples,nblks,ps)
        dldz_low = dldz_[:,:,:,ps:].view(nSamples,nblks,pa)
        a_ = X.clone()
        c_upp = a_[:,:,:,:ps].view(nSamples,nblks,ps)
        c_low = a_[:,:,:,ps:].view(nSamples,nblks,pa)
        for iAngle in range(nAnglesH):
            dW0 = genW(angles=anglesW,mus=mus,index_pd_angle=iAngle)
            dU0 = genU(angles=anglesU,mus=mus,index_pd_angle=iAngle)
            for iblk in range(nblks):
                dldz_upp_iblk = dldz_upp[:,iblk,:] # nSamples x ps
                dldz_low_iblk = dldz_low[:,iblk,:] # nSamples x pa
                c_upp_iblk = c_upp[:,iblk,:] # nSamples x ps
                c_low_iblk = c_low[:,iblk,:] # nSamples x pa
                d_upp_iblk = torch.empty_like(c_upp_iblk)
                d_low_iblk = torch.empty_like(c_low_iblk)
                for iSample in range(nSamples):
                    d_upp_iblk[iSample,:] = dW0[iblk,:,:ps] @ c_upp_iblk[iSample,:]
                    d_low_iblk[iSample,:] = dU0[iblk,:,:pa] @ c_low_iblk[iSample,:]
                dldw_[iblk,iAngle] = torch.sum(dldz_upp_iblk * d_upp_iblk)
                dldw_[iblk,nAnglesH+iAngle] = torch.sum(dldz_low_iblk * d_low_iblk)
        expctddLdW = dldw_
        
        # Instantiation of target class
        layer = LsunInitialRotation2dLayer(
            dtype=datatype,
            device=device,
            stride=stride,
            number_of_blocks=[nrows,ncols],
            name='V0')
        layer.angles = torch.cat((anglesW,anglesU),dim=1)
        layer.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True) 
        Z = layer(X)
        layer.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = torch.cat((layer.orthTransW0.orthonormalTransforms.angles.grad, layer.orthTransU0.orthonormalTransforms.angles.grad),dim=1) 

        # Evaluation
        self.assertIsInstance(actualdLdX,torch.Tensor)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertIsInstance(actualdLdW[iblk],torch.Tensor)
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))
    
    @parameterized.expand(
            itertools.product(usegpu,stride,nrows,ncols,datatype)
            )
    def testBackwardGrayscaleWithRandomAngles(self, usegpu, stride, nrows, ncols, datatype):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-5

        genW = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype,partial_difference=True,mode='normal')
        genU = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype,partial_difference=True,mode='normal')

        # Parameters
        nSamples = 8
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]
        nAnglesH = int((nDecs-2)*nDecs/8)
        anglesW = torch.randn(nrows*ncols,nAnglesH,dtype=datatype,device=device)
        anglesU = torch.randn(nrows*ncols,nAnglesH,dtype=datatype,device=device)
        mus = 1

        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device)

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        W0T = genW(angles=anglesW,mus=mus,index_pd_angle=None).transpose(1,2)
        U0T = genU(angles=anglesU,mus=mus,index_pd_angle=None).transpose(1,2)
        Y = dLdZ.clone()
        expctddLdX = torch.empty_like(X)
        for iSample in range(nSamples):
            Yi = Y[iSample,:,:,:]
            Ys = Yi[:,:,:ps].view(-1,ps)
            Ya = Yi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0T[iblk,:,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0T[iblk,:,:] @ Ya[iblk,:]
            Zsai = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            expctddLdX[iSample] = Zsai

        # dLdWi = <dLdZ,(dVdWi)X>
        nblks = nrows*ncols
        dldw_ = torch.empty(nblks,2*nAnglesH,dtype=datatype,device=device)  
        dldz_= dLdZ.clone()
        dldz_upp = dldz_[:,:,:,:ps].view(nSamples,nblks,ps)
        dldz_low = dldz_[:,:,:,ps:].view(nSamples,nblks,pa)
        a_ = X.clone()
        c_upp = a_[:,:,:,:ps].view(nSamples,nblks,ps)
        c_low = a_[:,:,:,ps:].view(nSamples,nblks,pa)
        for iAngle in range(nAnglesH):
            dW0 = genW(angles=anglesW,mus=mus,index_pd_angle=iAngle)
            dU0 = genU(angles=anglesU,mus=mus,index_pd_angle=iAngle)
            for iblk in range(nblks):
                dldz_upp_iblk = dldz_upp[:,iblk,:] # nSamples x ps 
                dldz_low_iblk = dldz_low[:,iblk,:] # nSamples x pa
                c_upp_iblk = c_upp[:,iblk,:] # nSamples x ps
                c_low_iblk = c_low[:,iblk,:] # nSamples x pa
                d_upp_iblk = torch.empty_like(c_upp_iblk)
                d_low_iblk = torch.empty_like(c_low_iblk)
                for iSample in range(nSamples):
                    d_upp_iblk[iSample,:] = dW0[iblk,:,:ps] @ c_upp_iblk[iSample,:]
                    d_low_iblk[iSample,:] = dU0[iblk,:,:pa] @ c_low_iblk[iSample,:]
                dldw_[iblk,iAngle] = torch.sum(dldz_upp_iblk * d_upp_iblk)
                dldw_[iblk,nAnglesH+iAngle] = torch.sum(dldz_low_iblk * d_low_iblk)
        expctddLdW = dldw_

        # Instantiation of target class
        layer = LsunInitialRotation2dLayer(
            dtype=datatype,
            device=device,
            stride=stride,
            number_of_blocks=[nrows,ncols],
            name='V0')
        layer.angles = torch.cat((anglesW,anglesU),dim=1)
        layer.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = layer(X)
        layer.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW =  torch.cat((layer.orthTransW0.orthonormalTransforms.angles.grad, layer.orthTransU0.orthonormalTransforms.angles.grad),1)  
        
        # Evaluation
        self.assertIsInstance(actualdLdX,torch.Tensor)
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertIsInstance(actualdLdW[iblk],torch.Tensor)
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))

    @parameterized.expand(
            itertools.product(usegpu,stride,nrows,ncols,mus,datatype)
            )
    def testBackwardGrayscaleWithRandomAnglesNoDcLeackage(self, usegpu, stride, nrows, ncols, mus, datatype):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-5

        genW = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype,partial_difference=True,mode='normal')
        genU = OrthonormalMatrixGenerationSystem(device=device,dtype=datatype,partial_difference=True,mode='normal')

        # Parameters
        nSamples = 8
        nDecs = stride[Direction.VERTICAL]*stride[Direction.HORIZONTAL]
        nAnglesH = (nDecs-2)*nDecs//8
        anglesW = torch.randn(nrows*ncols,nAnglesH,dtype=datatype,device=device)
        anglesU = torch.randn(nrows*ncols,nAnglesH,dtype=datatype,device=device)
        
        # nSamples x nRows x nCols x nDecs
        X = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nSamples,nrows,ncols,nDecs,dtype=datatype,device=device)

        # Expected values
        ps = math.ceil(nDecs/2)
        pa = math.floor(nDecs/2)
        anglesW_NoDc = anglesW.clone()
        anglesW_NoDc[:,:(ps-1)] = 0
        musW = mus*torch.ones(nrows*ncols,ps,dtype=datatype,device=device)
        musW[:,0] = 1
        musU = mus*torch.ones(nrows*ncols,pa,dtype=datatype,device=device)
        W0T = genW(angles=anglesW_NoDc,mus=musW,index_pd_angle=None).transpose(1,2)
        U0T = genU(angles=anglesU,mus=musU,index_pd_angle=None).transpose(1,2)
        Y = dLdZ.clone()
        exceptdLdX = torch.empty_like(X)
        for iSample in range(nSamples):
            Yi = Y[iSample,:,:,:]
            Ys = Yi[:,:,:ps].view(-1,ps)
            Ya = Yi[:,:,ps:].view(-1,pa)
            for iblk in range(nrows*ncols):
                Ys[iblk,:] = W0T[iblk,:,:] @ Ys[iblk,:]
                Ya[iblk,:] = U0T[iblk,:,:] @ Ya[iblk,:]
            Zsai = torch.cat((Ys,Ya),dim=1).view(nrows,ncols,nDecs)
            exceptdLdX[iSample] = Zsai

        # dLdWi = <dLdZ,(dVdWi)X>
        nblks = nrows*ncols
        dldw_ = torch.empty(nblks,2*nAnglesH,dtype=datatype,device=device)
        dldz_= dLdZ.clone()
        dldz_upp = dldz_[:,:,:,:ps].view(nSamples,nblks,ps)
        dldz_low = dldz_[:,:,:,ps:].view(nSamples,nblks,pa)
        a_ = X.clone()
        c_upp = a_[:,:,:,:ps].view(nSamples,nblks,ps)
        c_low = a_[:,:,:,ps:].view(nSamples,nblks,pa)
        for iAngle in range(nAnglesH):
            dW0 = genW(angles=anglesW_NoDc,mus=musW,index_pd_angle=iAngle)
            dU0 = genU(angles=anglesU,mus=musU,index_pd_angle=iAngle)
            for iblk in range(nblks):
                dldz_upp_iblk = dldz_upp[:,iblk,:] # nSamples x ps
                dldz_low_iblk = dldz_low[:,iblk,:] # nSamples x pa
                c_upp_iblk = c_upp[:,iblk,:] # nSamples x ps
                c_low_iblk = c_low[:,iblk,:] # nSamples x pa
                d_upp_iblk = torch.empty_like(c_upp_iblk)
                d_low_iblk = torch.empty_like(c_low_iblk)
                for iSample in range(nSamples):
                    d_upp_iblk[iSample,:] = dW0[iblk,:,:ps] @ c_upp_iblk[iSample,:]
                    d_low_iblk[iSample,:] = dU0[iblk,:,:pa] @ c_low_iblk[iSample,:]
                dldw_[iblk,iAngle] = torch.sum(dldz_upp_iblk * d_upp_iblk)
                dldw_[iblk,nAnglesH+iAngle] = torch.sum(dldz_low_iblk * d_low_iblk)
        expctddLdW = dldw_

        # Instantiation of target class
        layer = LsunInitialRotation2dLayer(
            dtype=datatype,
            device=device,
            stride=stride,
            number_of_blocks=[nrows,ncols],
            no_dc_leakage=True,
            name='V0')
        layer.angles = torch.cat((anglesW,anglesU),dim=1)
        layer.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = layer(X)
        layer.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = torch.cat((layer.orthTransW0.orthonormalTransforms.angles.grad, layer.orthTransU0.orthonormalTransforms.angles.grad),dim=1)
        
        # Evaluation
        self.assertIsInstance(actualdLdX,torch.Tensor)
        self.assertTrue(torch.allclose(actualdLdX,exceptdLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertIsInstance(actualdLdW[iblk],torch.Tensor)
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))
 
if __name__ == '__main__':
    unittest.main()

    """
    # Run specific test cases
    suite = unittest.TestSuite()
    suite.addTest(LsunInitialRotation2dLayerTestCase('testForwardGrayscaleWithRandomAngles_053'))

    runner = unittest.TextTestRunner()
    runner.run(suite)
    """
    
    """
    # TestSuite に特定のテストケースを追加
    suite = unittest.TestSuite()
    suite.addTest(LsunInitialRotation2dLayerTestCase('testBackwardGrayscale_142'))

    # TestRunner でスイートを実行
    runner = unittest.TextTestRunner()
    runner.run(suite)
    """

"""
classdef lsunInitialRotation2dLayerTestCase < matlab.unittest.TestCase
    %LSUNINITIALROTATION2DLAYERTESTCASE
    %
    %   コンポーネント別に入力(nComponents=1のみサポート):
    %      nChs x nRows x nCols x nSamples
    %
    %   コンポーネント別に出力(nComponents=1のみサポート):
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
        mus = { -1, 1 };
        datatype = { 'single', 'double' };
        nrows = struct('small', 2,'medium', 4, 'large', 8);
        ncols = struct('small', 2,'medium', 4, 'large', 8);
        usegpu = struct( 'true', true, 'false', false);           
    end
    
    methods (TestClassTeardown)
        function finalCheck(~)
            import tansacnet.lsun.*
            layer = lsunInitialRotation2dLayer(...
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
            expctdName = 'V0';
            expctdDescription = "LSUN initial rotation " ...
                + "(ps,pa) = (" ...
                + ceil(prod(stride)/2) + "," ...
                + floor(prod(stride)/2) + "), "  ...
                + "(mv,mh) = (" ...
                + stride(1) + "," + stride(2) + ")";
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation2dLayer(...
                'Stride',stride,...
                'Name',expctdName);
            
            % Actual values
            actualName = layer.Name;
            actualDescription = layer.Description;
            
            % Evaluation
            testCase.verifyEqual(actualName,expctdName);
            testCase.verifyEqual(actualDescription,expctdDescription);
        end
  
        function testPredictGrayscale(testCase, ...
                usegpu, stride, nrows, ncols, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
            end
            
            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0 = repmat(eye(ps,datatype),[1 1 nrows*ncols]);
            U0 = repmat(eye(pa,datatype),[1 1 nrows*ncols]);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y(1:ps,:,:) = reshape(Ys,ps,nrows,ncols);
                Y(ps+1:ps+pa,:,:) = reshape(Ya,pa,nrows,ncols);                
                expctdZ(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0');
            
            % Actual values
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
                usegpu, stride, nrows, ncols, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end            
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem();
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,nrows*ncols);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end            

            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            W0 = genW.step(angles(1:size(angles,1)/2,:),1);
            U0 = genU.step(angles(size(angles,1)/2+1:end,:),1);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nDecs,nrows,ncols);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y(1:ps,:,:) = reshape(Ys,ps,nrows,ncols);
                Y(ps+1:ps+pa,:,:) = reshape(Ya,pa,nrows,ncols);
                expctdZ(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0');
            
            % Actual values
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
       
        function testPredictGrayscaleWithRandomAnglesNoDcLeackage(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end    
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-6,single(1e-6));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem();
            genU = OrthonormalMatrixGenerationSystem();
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            % nDecs x nRows x nCols x nDecs x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            angles = randn((nChsTotal-2)*nChsTotal/4,nrows*ncols);
            if usegpu
                X = gpuArray(X);
                angles = gpuArray(angles);
            end   

            % Expected values
            % nChs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            anglesNoDc = angles;
            anglesNoDc(1:ps-1,:)=zeros(ps-1,nrows*ncols);
            musW = mus*ones(ps,nrows*ncols);
            musW(1,1:end) = 1;
            musU = mus*ones(pa,nrows*ncols);
            W0 = genW.step(anglesNoDc(1:size(angles,1)/2,:),musW);
            U0 = genU.step(anglesNoDc(size(angles,1)/2+1:end,:),musU);
            %expctdZ = zeros(nrows,ncols,nChsTotal,nSamples,datatype);
            expctdZ = zeros(nChsTotal,nrows,ncols,nSamples,datatype);
            Y  = zeros(nChsTotal,nrows,ncols,datatype);
            for iSample=1:nSamples
                % Perumation in each block
                Ai = X(:,:,:,iSample); %permute(X(:,:,:,iSample),[3 1 2]);
                Yi = reshape(Ai,nDecs,nrows*ncols);
                %
                Ys = Yi(1:ps,:);
                Ya = Yi(ps+1:end,:);
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk) = W0(:,1:ps,iblk)*Ys(:,iblk);
                    Ya(:,iblk) = U0(:,1:pa,iblk)*Ya(:,iblk);
                end
                Y(1:ps,:,:) = reshape(Ys,ps,nrows,ncols);
                Y(ps+1:end,:,:) = reshape(Ya,pa,nrows,ncols);                
                expctdZ(:,:,:,iSample) = Y; %ipermute(Y,[3 1 2]);
            end
            
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'NoDcLeakage',true,...
                'Name','V0');
            
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

        function testBackwardGrayscale(testCase, ...
                usegpu, stride, nrows, ncols, datatype)
            
            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-4));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = zeros(nAnglesH,nrows*ncols,datatype);
            anglesU = zeros(nAnglesH,nrows*ncols,datatype);
            mus_ = 1;
            
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,sum(stride),nSamples,datatype);            
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nSamples,datatype);                        
            if usegpu
                X = gpuArray(X);
                anglesW = gpuArray(anglesW);
                anglesU = gpuArray(anglesU);
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values
            % nDecs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            W0T = permute(genW.step(anglesW,mus_,0),[2 1 3]);
            U0T = permute(genU.step(anglesU,mus_,0),[2 1 3]);
            Y = dLdZ; % permute(dLdZ,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample); 
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctddLdX = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctddLdX = reshape(Zsa,nDecs,nrows,ncols,nSamples);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(2*nAnglesH,nrows*ncols,1,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            dldz_low = reshape(dldz_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(a_(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            c_low = reshape(a_(ps+1:nDecs,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = 1:nAnglesH
                dW0 = genW.step(anglesW,mus_,iAngle);
                dU0 = genU.step(anglesU,mus_,iAngle);
                for iblk = 1:(nrows*ncols)                
                    dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                    dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                    c_upp_iblk = squeeze(c_upp(:,iblk,:));
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                    d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                    for iSample = 1:nSamples
                        d_upp_iblk(:,iSample) = dW0(:,1:ps,iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0(:,1:pa,iblk)*c_low_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_upp_iblk.*d_upp_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_low_iblk.*d_low_iblk,'all');
                end
            end
            expctddLdW = dldw_;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0');
            layer.Mus = mus_;
            %expctdZ = layer.predict(X);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                testCase.verifyClass(actualdLdW,'gpuArray')
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
                usegpu, stride, nrows, ncols, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-3));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = randn(nAnglesH,nrows*ncols,datatype);
            anglesU = randn(nAnglesH,nrows*ncols,datatype);
            mus_ = 1;
            
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,sum(stride),nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nSamples,datatype);
            if usegpu
                X = gpuArray(X);
                anglesW = gpuArray(anglesW);
                anglesU = gpuArray(anglesU);
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values
            % nDecs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            W0T = permute(genW.step(anglesW,mus_,0),[2 1 3]);
            U0T = permute(genU.step(anglesU,mus_,0),[2 1 3]);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample); 
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctddLdX = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctddLdX = reshape(Zsa,nDecs,nrows,ncols,nSamples);
                        
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(2*nAnglesH,nrows*ncols,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            dldz_low = reshape(dldz_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(a_(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            c_low = reshape(a_(ps+1:nDecs,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = 1:nAnglesH
                dW0 = genW.step(anglesW,mus_,iAngle);
                dU0 = genU.step(anglesU,mus_,iAngle);
                for iblk = 1:(nrows*ncols)
                    dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                    dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                    c_upp_iblk = squeeze(c_upp(:,iblk,:));
                    c_low_iblk = squeeze(c_low(:,iblk,:));
                    d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                    d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                    for iSample = 1:nSamples
                        d_upp_iblk(:,iSample) = dW0(:,1:ps,iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0(:,1:pa,iblk)*c_low_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_upp_iblk.*d_upp_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_low_iblk.*d_low_iblk,'all');
                end
            end
            expctddLdW = dldw_;

            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'Name','V0');
            layer.Mus = mus_;
            layer.Angles = [anglesW; anglesU];
            %expctdZ = layer.predict(X);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                testCase.verifyClass(actualdLdW,'gpuArray')
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

        function testBackwardGrayscaleWithRandomAnglesNoDcLeackage(testCase, ...
                usegpu, stride, nrows, ncols, mus, datatype)

            if usegpu && gpuDeviceCount == 0
                warning('No GPU device was detected.')
                return;
            end
            import matlab.unittest.constraints.IsEqualTo
            import matlab.unittest.constraints.AbsoluteTolerance
            tolObj = AbsoluteTolerance(1e-4,single(1e-3));
            import tansacnet.utility.*
            genW = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            genU = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','on');
            
            % Parameters
            nSamples = 8;
            nDecs = prod(stride);
            nChsTotal = nDecs;
            nAnglesH = (nChsTotal-2)*nChsTotal/8;
            anglesW = randn(nAnglesH,nrows*ncols,datatype);
            anglesU = randn(nAnglesH,nrows*ncols,datatype);
            
            % nDecs x nRows x nCols x nSamples
            %X = randn(nrows,ncols,nDecs,nSamples,datatype);
            %dLdZ = randn(nrows,ncols,sum(stride),nSamples,datatype);
            X = randn(nDecs,nrows,ncols,nSamples,datatype);
            dLdZ = randn(nDecs,nrows,ncols,nSamples,datatype);            
            if usegpu
                X = gpuArray(X);
                anglesW = gpuArray(anglesW);
                anglesU = gpuArray(anglesU);
                dLdZ = gpuArray(dLdZ);
            end

            % Expected values
            % nDecs x nRows x nCols x nSamples
            ps = ceil(nChsTotal/2);
            pa = floor(nChsTotal/2);
            
            % dLdX = dZdX x dLdZ
            anglesW_NoDc = anglesW;
            anglesW_NoDc(1:ps-1,:)=zeros(ps-1,nrows*ncols);
            musW = mus*ones(ps,nrows*ncols);
            musW(1,:) = ones(1,nrows*ncols);
            musU = mus*ones(pa,nrows*ncols);            
            W0T = permute(genW.step(anglesW_NoDc,musW,0),[2 1 3]);
            U0T = permute(genU.step(anglesU,musU,0),[2 1 3]);
            Y = dLdZ; %permute(dLdZ,[3 1 2 4]);
            Ys = reshape(Y(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            Ya = reshape(Y(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            for iSample = 1:nSamples
                for iblk = 1:(nrows*ncols)
                    Ys(:,iblk,iSample) = W0T(1:ps,:,iblk)*Ys(:,iblk,iSample);
                    Ya(:,iblk,iSample) = U0T(1:pa,:,iblk)*Ya(:,iblk,iSample);
                end
            end
            Zsa = cat(1,Ys,Ya);
            %expctddLdX = ipermute(reshape(Zsa,nDecs,nrows,ncols,nSamples),...
            %    [3 1 2 4]);
            expctddLdX = reshape(Zsa,nDecs,nrows,ncols,nSamples);
            
            % dLdWi = <dLdZ,(dVdWi)X>
            dldw_ = zeros(2*nAnglesH,nrows*ncols,datatype);
            dldz_ = dLdZ; %permute(dLdZ,[3 1 2 4]);
            dldz_upp = reshape(dldz_(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            dldz_low = reshape(dldz_(ps+1:ps+pa,:,:,:),pa,nrows*ncols,nSamples);
            % (dVdWi)X
            a_ = X; %permute(X,[3 1 2 4]);
            c_upp = reshape(a_(1:ps,:,:,:),ps,nrows*ncols,nSamples);
            c_low = reshape(a_(ps+1:nDecs,:,:,:),pa,nrows*ncols,nSamples);
            for iAngle = 1:nAnglesH
                dW0 = genW.step(anglesW_NoDc,musW,iAngle);
                dU0 = genU.step(anglesU,musU,iAngle);
                for iblk = 1:(nrows*ncols)
                    dldz_upp_iblk = squeeze(dldz_upp(:,iblk,:));
                    dldz_low_iblk = squeeze(dldz_low(:,iblk,:));
                    c_upp_iblk = squeeze(c_upp(:,iblk,:));
                    c_low_iblk = squeeze(c_low(:,iblk,:));                    
                    d_upp_iblk = zeros(size(c_upp_iblk),'like',c_upp_iblk);
                    d_low_iblk = zeros(size(c_low_iblk),'like',c_low_iblk);
                    for iSample = 1:nSamples
                        d_upp_iblk(:,iSample) = dW0(:,1:ps,iblk)*c_upp_iblk(:,iSample);
                        d_low_iblk(:,iSample) = dU0(:,1:pa,iblk)*c_low_iblk(:,iSample);
                    end
                    dldw_(iAngle,iblk) = sum(dldz_upp_iblk.*d_upp_iblk,'all');
                    dldw_(nAnglesH+iAngle,iblk) = sum(dldz_low_iblk.*d_low_iblk,'all');
                end
            end
            expctddLdW = dldw_;
        
            % Instantiation of target class
            import tansacnet.lsun.*
            layer = lsunInitialRotation2dLayer(...
                'Stride',stride,...
                'NumberOfBlocks',[nrows ncols],...
                'NoDcLeakage',true,...
                'Name','V0');
            layer.Mus = mus;
            layer.Angles = [anglesW; anglesU];
            %expctdZ = layer.predict(X);
            
            % Actual values
            [actualdLdX,actualdLdW] = layer.backward(X,[],dLdZ,[]);
            
            % Evaluation
            if usegpu
                testCase.verifyClass(actualdLdX,'gpuArray')
                actualdLdX = gather(actualdLdX);
                expctddLdX = gather(expctddLdX);
                testCase.verifyClass(actualdLdW,'gpuArray')
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
