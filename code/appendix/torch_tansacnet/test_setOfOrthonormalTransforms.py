import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import math
from random import *
from torch_tansacnet.orthonormalTransform import SetOfOrthonormalTransforms
from torch_tansacnet.lsunUtility import OrthonormalMatrixGenerationSystem
from torch_tansacnet.lsunLayerExceptions import InvalidMode, InvalidMus

datatype = [ torch.float32, torch.float64 ]
nsamples = [ 1, 2, 4 ]
nblks = [ 1, 2, 4 ]
npoints = [ 1, 2, 3, 4, 5, 6 ]
mode = [ 'Analysis', 'Synthesis' ]
#isdevicetest = True
usegpu = [ True, False ]

class SetOfOrthonormalTransformsTestCase(unittest.TestCase):
    """
    SETOFORTHONORMALTRANSFORMSTESTCASE
    
    Requirements: Python 3.10-12.x, PyTorch 2.3/4.x
    
    Copyright (c) 2021-2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        https://www.eng.niigata-u.ac.jp/~msiplab/
    """

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,usegpu))
    )
    def testConstructor(self,datatype,nblks,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")    
        rtol,atol = 1e-5,1e-8 

        # Expected values
        X = torch.randn(nblks,2,nsamples,dtype=datatype)
        X = X.to(device)

        expctdZ = X
        expctdNTrxs = nblks
        expctdMode = 'Analysis'

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,device=device,dtype=datatype)

        # Actual values
        with torch.no_grad():
            actualZ = target(X)
        #actualNTrxs = len(target.orthonormalTransforms)
        actualMode = target.mode
        
        # Evaluation
        self.assertTrue(isinstance(target,nn.Module))        
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))    
        #self.assertEqual(actualNTrxs,expctdNTrxs)
        self.assertEqual(actualMode,expctdMode)

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,usegpu))
    )
    def testConstructorToDevice(self,datatype,nblks,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")    
        rtol,atol = 1e-5,1e-8    

        # Expected values
        X = torch.randn(nblks,2,nsamples,dtype=datatype)
        X = X.to(device)

        expctdZ = X
        expctdNTrxs = nblks
        expctdMode = 'Analysis'

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,dtype=datatype)
        target = target.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = target(X)
        #actualNTrxs = len(target.orthonormalTransforms)
        actualMode = target.mode
        
        # Evaluation
        self.assertTrue(isinstance(target,nn.Module))        
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))    
        #self.assertEqual(actualNTrxs,expctdNTrxs)
        self.assertEqual(actualMode,expctdMode)

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,usegpu))
    )
    def testCallWithAngles(self,datatype,nblks,nsamples,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")    
        rtol, atol = 1e-4, 1e-7

        # Expected values
        X = torch.randn(nblks,2,nsamples,dtype=datatype)
        X = X.to(device)
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4),  math.cos(math.pi/4) ] ],
            dtype=datatype)
        R = R.to(device)
        expctdZ = torch.empty_like(X)
        for iblk in range(nblks):
            X_iblk = X[iblk,:,:]
            if mode!='Synthesis':
                Z_iblk = R @ X_iblk.view(2,-1)
            else:
                Z_iblk = R.T @ X_iblk.view(2,-1)
            expctdZ[iblk,:,:] = Z_iblk.view(2,nsamples)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.orthonormalTransforms.angles = nn.init.constant_(target.orthonormalTransforms.angles,val=math.pi/4)

        # Actual values
        with torch.no_grad():
            actualZ = target(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,usegpu))
    )
    def testCallWithAnglesToDevice(self,datatype,nblks,nsamples,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-7

        # Expected values
        X = torch.randn(nblks,2,nsamples,dtype=datatype)
        X = X.to(device)
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4),  math.cos(math.pi/4) ] ],
            dtype=datatype) 
        R = R.to(device)
        if mode!='Synthesis':
            expctdZ = R @ X.view(nblks,2,-1)
        else:
            expctdZ = R.T @ X.view(nblks,2,-1)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,mode=mode,dtype=datatype)
        target = target.to(device)
        target.orthonormalTransforms.angles = nn.init.constant_(target.orthonormalTransforms.angles,val=math.pi/4)

        # Actual values
        with torch.no_grad():
            actualZ = target(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,usegpu))
    )
    def testCallWithAnglesAndMus(self,datatype,nblks,nsamples,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-7

        # Expected values
        X = torch.randn(nblks,2,nsamples,dtype=datatype)
        X = X.to(device)
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ -math.sin(math.pi/4), -math.cos(math.pi/4) ] ],
            dtype=datatype)
        R = R.to(device)
        expctdZ = torch.empty_like(X)
        for iblk in range(nblks):
            X_iblk = X[iblk,:,:]
            if mode!='Synthesis':
                Z_iblk = R @ X_iblk.view(2,-1)
            else:
                Z_iblk = R.T @ X_iblk.view(2,-1)
            expctdZ[iblk,:,:] = Z_iblk.view(2,nsamples)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.orthonormalTransforms.angles = nn.init.constant_(target.orthonormalTransforms.angles,val=math.pi/4)
        target.orthonormalTransforms.mus = torch.tensor([1, -1])

        # Actual values
        with torch.no_grad():
            actualZ = target(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))
        

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,usegpu))
    )
    def testCallWithAnglesAndMusToDevice(self,datatype,nblks,nsamples,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-7

        # Expected values
        X = torch.randn(nblks,2,nsamples,dtype=datatype)
        X = X.to(device)
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ -math.sin(math.pi/4), -math.cos(math.pi/4) ] ],
            dtype=datatype)
        R = R.to(device)
        expctdZ = torch.empty_like(X)        
        for iblk in range(nblks):
            X_iblk = X[iblk,:,:]
            if mode!='Synthesis':
                Z_iblk = R @ X_iblk.view(2,-1)
            else:
                Z_iblk = R.T @ X_iblk.view(2,-1)
            expctdZ[iblk,:,:] = Z_iblk.view(2,nsamples)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,mode=mode,dtype=datatype)
        target = target.to(device)
        target.orthonormalTransforms.angles = nn.init.constant_(target.orthonormalTransforms.angles,val=math.pi/4)
        target.orthonormalTransforms.mus = torch.tensor([1, -1])

        # Actual values
        with torch.no_grad():
            actualZ = target(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,usegpu))
    )
    def testSetAngles(self,datatype,nblks,nsamples,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-7

        # Expected values
        X = torch.randn(nblks,2,nsamples,dtype=datatype)
        X = X.to(device)
        R = torch.eye(2,dtype=datatype)
        R = R.to(device)
        expctdZ = torch.empty_like(X)
        for iblk in range(nblks):
            X_iblk = X[iblk,:,:]
            if mode!='Synthesis':
                Z_iblk = R @ X_iblk.view(2,-1)
            else:
                Z_iblk = R.T @ X_iblk.view(2,-1)
            expctdZ[iblk,:,:] = Z_iblk.view(2,nsamples)
        
        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,mode=mode,device=device,dtype=datatype)

        # Actual values
        with torch.no_grad():
            actualZ = target(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

        # Expected values
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4), math.cos(math.pi/4) ] ],
            dtype=datatype)
        R = R.to(device)
        expctdZ = torch.empty_like(X)
        for iblk in range(nblks):
            X_iblk = X[iblk,:,:]
            if mode!='Synthesis':
                Z_iblk = R @ X_iblk.view(2,-1)
            else:
                Z_iblk = R.T @ X_iblk.view(2,-1)
            expctdZ[iblk,:,:] = Z_iblk.view(2,nsamples)

        # Actual values
        target.orthonormalTransforms.angles.data = torch.tensor([math.pi/4],dtype=datatype).to(device)
        with torch.no_grad():
            actualZ = target(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,usegpu))
    )
    def testSetAnglesToDevice(self,datatype,nblks,nsamples,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-7

        # Expected values
        X = torch.randn(nblks,2,nsamples,dtype=datatype)
        X = X.to(device)
        R = torch.eye(2,dtype=datatype)
        R = R.to(device)
        expctdZ = torch.empty_like(X)
        for iblk in range(nblks):
            X_iblk = X[iblk,:,:]
            if mode!='Synthesis':
                Z_iblk = R @ X_iblk.view(2,-1)
            else:
                Z_iblk = R.T @ X_iblk.view(2,-1)
            expctdZ[iblk,:,:] = Z_iblk.view(2,nsamples)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,mode=mode,dtype=datatype)
        target = target.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = target(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

        # Expected values
        R = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4), math.cos(math.pi/4) ] ],
            dtype=datatype)
        R = R.to(device)
        expctdZ = torch.empty_like(X)
        for iblk in range(nblks):
            X_iblk = X[iblk,:,:]
            if mode!='Synthesis':
                Z_iblk = R @ X_iblk.view(2,-1)
            else:
                Z_iblk = R.T @ X_iblk.view(2,-1)
            expctdZ[iblk,:,:] = Z_iblk.view(2,nsamples)

        # Actual values
        target.orthonormalTransforms.angles.data = torch.tensor([math.pi/4],dtype=datatype).to(device)
        with torch.no_grad():
            actualZ = target(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,usegpu))
    )
    def test4x4(self,datatype,nblks,nsamples,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-7

        # Expected values
        expctdNorm = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=4,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.orthonormalTransforms.angles = nn.init.normal_(target.orthonormalTransforms.angles)

        # Actual values
        unitvec = torch.randn(nblks,4,nsamples,dtype=datatype,device=device)
        unitvec /= torch.linalg.vector_norm(unitvec,dim=1).unsqueeze(dim=1)
        with torch.no_grad():
            actualNorm = torch.linalg.vector_norm(target(unitvec),dim=1)

        # Evaluation
        self.assertTrue(torch.allclose(actualNorm,expctdNorm,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,usegpu))
    )
    def test8x8(self,datatype,nblks,nsamples,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-7

        # Expected values
        expctdNorm = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=8,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.orthonormalTransforms.angles = nn.init.normal_(target.orthonormalTransforms.angles)

        # Actual values
        unitvec = torch.randn(nblks,8,nsamples,dtype=datatype,device=device)
        unitvec /= torch.linalg.vector_norm(unitvec,dim=1).unsqueeze(dim=1)
        with torch.no_grad():
            actualNorm = torch.linalg.vector_norm(target(unitvec),dim=1)

        # Evaluation
        message = "actualNorm=%s differs from %s" % ( str(actualNorm), str(expctdNorm) )
        self.assertTrue(torch.allclose(actualNorm,expctdNorm,rtol=rtol,atol=atol),message)

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,npoints,usegpu))
    )
    def testNxN(self,datatype,nblks,nsamples,mode,npoints,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-7

        # Expected values
        expctdNorm = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        #nAngles = int(npoints*(npoints-1)/2)
        target = SetOfOrthonormalTransforms(n=npoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.orthonormalTransforms.angles = nn.init.normal_(target.orthonormalTransforms.angles)

        # Actual values
        unitvec = torch.randn(nblks,npoints,nsamples,dtype=datatype,device=device)
        unitvec /= torch.linalg.vector_norm(unitvec,dim=1).unsqueeze(dim=1)
        with torch.no_grad():
            actualNorm = torch.linalg.vector_norm(target(unitvec),dim=1)

        # Evaluation
        message = "actualNorm=%s differs from %s" % ( str(actualNorm), str(expctdNorm) )
        self.assertTrue(torch.allclose(actualNorm,expctdNorm,rtol=rtol,atol=atol),message)

    @parameterized.expand(
        list(itertools.product(datatype,nblks,mode,usegpu))
    )
    def test4x4red(self,datatype,nblks,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-7

        # Configuration
        nPoints = 4
        #nAngles = int(nPoints*(nPoints-1)/2)

        # Expected values
        expctdLeftTop = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.orthonormalTransforms.angles = nn.init.uniform_(target.orthonormalTransforms.angles,a=0.0,b=2*math.pi)
        target.orthonormalTransforms.angles.data[:,:nPoints-1] = torch.zeros(nPoints-1)

        # Actual values
        X = torch.eye(nPoints,dtype=datatype,device=device).repeat(nblks,1,1)
        with torch.no_grad():
            Z = target(X)
        actualLeftTop = Z[:,0,0]

        # Evaluation
        message = "actualLeftTop=%s differs from %s" % ( str(actualLeftTop), str(expctdLeftTop) )
        self.assertTrue(torch.allclose(actualLeftTop,expctdLeftTop,rtol=rtol,atol=atol),message)

    @parameterized.expand(
        list(itertools.product(datatype,nblks,mode,usegpu))
    )
    def test8x8red(self,datatype,nblks,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-7

        # Configuration
        nPoints = 8
        #nAngles = int(nPoints*(nPoints-1)/2)

        # Expected values
        expctdLeftTop = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.orthonormalTransforms.angles = nn.init.uniform_(target.orthonormalTransforms.angles,a=0.0,b=2*math.pi)
        target.orthonormalTransforms.angles.data[:,:nPoints-1] = torch.zeros(nPoints-1)

        # Actual values
        X = torch.eye(nPoints,dtype=datatype,device=device).repeat(nblks,1,1)
        with torch.no_grad():
            Z = target(X)
        actualLeftTop = Z[:,0,0]

        # Evaluation
        message = "actualLeftTop=%s differs from %s" % ( str(actualLeftTop), str(expctdLeftTop) )
        self.assertTrue(torch.allclose(actualLeftTop,expctdLeftTop,rtol=rtol,atol=atol),message)

    @parameterized.expand(
        list(itertools.product(datatype,nblks,npoints,mode,usegpu))
    )
    def testNxNred(self,datatype,nblks,npoints,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-7

        # Configuration
        #nAngles = int(npoints*(npoints-1)/2)

        # Expected values
        expctdLeftTop = torch.tensor(1.,dtype=datatype)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=npoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.orthonormalTransforms.angles = nn.init.uniform_(target.orthonormalTransforms.angles,a=0.0,b=2*math.pi)
        target.orthonormalTransforms.angles.data[:,:npoints-1] = torch.zeros(npoints-1)

        # Actual values
        X = torch.eye(npoints,dtype=datatype,device=device).repeat(nblks,1,1)
        with torch.no_grad():
            Z = target(X)
        actualLeftTop = Z[:,0,0]

        # Evaluation
        message = "actualLeftTop=%s differs from %s" % ( str(actualLeftTop), str(expctdLeftTop) )
        self.assertTrue(torch.allclose(actualLeftTop,expctdLeftTop,rtol=rtol,atol=atol),message)

    @parameterized.expand(
        list(itertools.product(datatype,nblks,mode,usegpu))
    )
    def testBackward(self,datatype,nblks,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol, atol = 1e-4, 1e-7

        # Configuration2
        nSamples = 1
        nPoints = 2
        nAngles = int(nPoints*(nPoints-1)/2)

        # Expected values
        X = torch.randn(nblks,nPoints,nSamples,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nblks,nPoints,nSamples,dtype=datatype,device=device)
        R = torch.eye(nPoints,dtype=datatype,device=device)
        dRdW = torch.tensor([
            [ 0., -1. ],
            [ 1., 0. ] ],
            dtype=datatype,device=device)
        dRdW = dRdW.to(device)
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.zeros(nblks,nAngles,dtype=datatype,device=device)        
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R.mT @ dLdZ_iblk
                dLdW_iblk = dLdZ_iblk.mT @ dRdW @ X_iblk
            else:
                dLdX_iblk = R @ dLdZ_iblk
                dLdW_iblk = dLdZ_iblk.mT @ dRdW.mT @ X_iblk
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        
        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        #actualdLdW = [ target.orthonormalTransforms[iblk].angles.grad for iblk in range(nblks) ]
        actualdLdW = target.orthonormalTransforms.angles.grad 

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,usegpu))
    )
    def testBackwardMultiSamples(self,datatype,nblks,nsamples,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol = 1e-4,1e-7

        # Configuration
        nPoints = 2
        nAngles = int(nPoints*(nPoints-1)/2)

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)     
        dLdZ = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)
        R = torch.eye(nPoints,dtype=datatype,device=device)
        dRdW = torch.tensor([
            [ 0., -1. ],
            [ 1., 0. ] ],
            dtype=datatype)
        dRdW = dRdW.to(device)
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.zeros(nblks,nAngles,dtype=datatype,device=device)        
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R.T @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW @ X_iblk))
            else:
                dLdX_iblk = R @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW.T @ X_iblk))
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.orthonormalTransforms.angles.grad 

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))
   
    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,usegpu))
    )
    def testBackwardMultiSamplesWithPi4Angles(self,datatype,nblks,nsamples,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol = 1e-4,1e-7

        # Configuration
        nPoints = 2
        nAngles = int(nPoints*(nPoints-1)/2)

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)
        angles = (math.pi/4.)*torch.ones(nblks,nAngles,dtype=datatype,device=device)
        omg = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=True,mode='normal')
        R = omg(angles=angles,index_pd_angle=None).to(device)
        dRdW = omg(angles=angles,index_pd_angle=0).to(device)
        expctdAngles = angles
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.empty(nblks,nAngles,dtype=datatype,device=device)        
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            R_iblk = R[iblk,:,:]
            dRdW_iblk = dRdW[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R_iblk.T @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk @ X_iblk))
            else:
                dLdX_iblk = R_iblk @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk.T @ X_iblk))
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)

        # Actual values
        target.angles = angles
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.orthonormalTransforms.angles.grad 
        actualAngles = target.angles

        # Evaluation
        self.assertTrue(torch.allclose(actualAngles,expctdAngles,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,usegpu))
    )
    def testBackwardMultiSamplesWithRandnAngles(self,datatype,nblks,nsamples,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol = 1e-4,1e-7

        # Configuration
        nPoints = 2
        nAngles = int(nPoints*(nPoints-1)/2)

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)
        angles = (math.pi/6)*torch.randn(nblks,nAngles,dtype=datatype,device=device)
        omg = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=True,mode='normal')
        R = omg(angles=angles,index_pd_angle=None).to(device)
        dRdW = omg(angles=angles,index_pd_angle=0).to(device)
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.empty(nblks,nAngles,dtype=datatype,device=device)        
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            R_iblk = R[iblk,:,:]
            dRdW_iblk = dRdW[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R_iblk.T @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk @ X_iblk))
            else:
                dLdX_iblk = R_iblk @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk.T @ X_iblk))
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)

        # Actual values
        target.angles = angles
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.orthonormalTransforms.angles.grad

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,nsamples,mode,usegpu))
    )
    def testBackwardAngsAndMus(self,datatype,nblks,nsamples,mode,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol = 1e-4,1e-7

        # Configuration
        nPoints = 2
        nAngles = int(nPoints*(nPoints-1)/2)
        mus = torch.tensor([1,-1]).repeat(nblks,1)
    
        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)
        angles = (math.pi/6)*torch.randn(nblks,nAngles,dtype=datatype,device=device)
        omg = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=True,mode='normal')
        R = omg(angles=angles,mus=mus,index_pd_angle=None).to(device)
        dRdW = omg(angles=angles,mus=mus,index_pd_angle=0).to(device)
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.empty(nblks,nAngles,dtype=datatype,device=device)
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            R_iblk = R[iblk,:,:]
            dRdW_iblk = dRdW[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R_iblk.T @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk @ X_iblk))
            else:
                dLdX_iblk = R_iblk @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk.T @ X_iblk))
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)

        # Actual values
        target.angles = angles
        target.mus = mus
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.orthonormalTransforms.angles.grad 

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))

    def testInstantiationWithInvalidMode(self):
        mode = 'Invalid'

        # Instantiation of target class
        with self.assertRaises(InvalidMode):
            target = SetOfOrthonormalTransforms(mode=mode)

  
    def testSetInvalidMode(self):
        mode = 'Invalid'        
        with self.assertRaises(InvalidMode):
            target = SetOfOrthonormalTransforms()
            target.mode = 'InvalidMode'

    @parameterized.expand(
        list(itertools.product(nblks))
    )
    def testSetInvalidMus(self,nblks):        
        mus = [ [2] for _ in range(nblks) ]      
        with self.assertRaises(InvalidMus):
            target = SetOfOrthonormalTransforms(nblks=nblks)
            target.mus = mus

    @parameterized.expand(
        list(itertools.product(datatype,nblks,mode,nsamples,usegpu))
    )
    def testBackwardSetAngles(self,datatype,nblks,mode,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol = 1e-4,1e-7

        # Configuration
        nPoints = 2
        nAngles = int(nPoints*(nPoints-1)/2)

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)
        angles = torch.zeros(nblks,nAngles,dtype=datatype,device=device)
        omg = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=True,mode='normal')
        R = omg(angles=angles,index_pd_angle=None).to(device)
        dRdW = omg(angles=angles,index_pd_angle=0).to(device)
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.zeros(nblks,nAngles,dtype=datatype,device=device)
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            R_iblk = R[iblk,:,:]
            dRdW_iblk = dRdW[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R_iblk.T @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk @ X_iblk))
            else:
                dLdX_iblk = R_iblk @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk.T @ X_iblk))
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.angles = angles

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.orthonormalTransforms.angles.grad

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))

        # Expected values
        X = X.detach()
        X.requires_grad = True
        angles = (math.pi/6.) * torch.randn(nblks,nAngles,dtype=datatype).to(device)
        R = omg(angles=angles,index_pd_angle=None).to(device)
        dRdW = omg(angles=angles,index_pd_angle=0).to(device)
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.zeros(nblks,nAngles,dtype=datatype,device=device)
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            R_iblk = R[iblk,:,:]
            dRdW_iblk = dRdW[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R_iblk.T @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk @ X_iblk))
            else:
                dLdX_iblk = R_iblk @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk.T @ X_iblk))
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)

        # Set angles
        target.angles = angles

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.orthonormalTransforms.angles.grad 

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))


    @parameterized.expand(
        list(itertools.product(datatype,nblks,mode,nsamples,usegpu))
    )
    def testForward4x4RandnAngs(self,datatype,nblks,mode,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol=1e-4,1e-7

        # Configuration
        nPoints = 4
        nAngles = int(nPoints*(nPoints-1)/2)
        angles = (math.pi/6.)*torch.randn(nblks,nAngles,dtype=datatype,device=device)
        mus = torch.tensor([-1,1,-1,1]).repeat(nblks,1)
        omg = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=False,mode='normal')

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)
        R = omg(angles=angles,mus=mus,index_pd_angle=None).to(device)
        expctdZ = torch.empty_like(X)
        for iblk in range(nblks):
            if mode!='Synthesis':
                expctdZ[iblk,:,:] = R[iblk,:,:] @ X[iblk,:,:]
            else:
                expctdZ[iblk,:,:] = R[iblk,:,:].T @ X[iblk,:,:]

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.angles = angles
        target.mus = mus

        # Actual values
        with torch.no_grad():
            actualZ = target(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,mode,nsamples,usegpu))
    )
    def testForward4x4RandnAngsToDevice(self,datatype,nblks,mode,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol=1e-4,1e-6

        # Configuration
        nPoints = 4
        nAngles = int(nPoints*(nPoints-1)/2)
        angles = (math.pi/6)*torch.randn(nblks,nAngles)
        mus = torch.tensor([-1,1,-1,1]).repeat(nblks,1)
        omg = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=False,mode='normal')

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype)
        X = X.to(device)
        R = omg(angles=angles,mus=mus,index_pd_angle=None).to(device)
        expctdZ = torch.empty_like(X)
        for iblk in range(nblks):
            if mode!='Synthesis':
                expctdZ[iblk,:,:] = R[iblk,:,:] @ X[iblk,:,:]
            else:
                expctdZ[iblk,:,:] = R[iblk,:,:].T @ X[iblk,:,:]

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,dtype=datatype)
        target.angles = angles
        target.mus = mus
        target = target.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = target(X)

        # Evaluation
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,mode,nsamples,usegpu))
    )
    def testBackward4x4RandAngPdAng2(self,datatype,nblks,mode,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol=1e-4,1e-6

        # Configuration
        nPoints = 4
        pdAng = 2
        nAngles = int(nPoints*(nPoints-1)/2)
        angles = (math.pi/6)*torch.randn(nblks,nAngles)
        mus = torch.tensor([-1,1,-1,1]).repeat(nblks,1)
        omg = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=True,mode='normal')

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)
        R = omg(angles=angles,mus=mus,index_pd_angle=None).to(device)
        dRdW = omg(angles=angles,mus=mus,index_pd_angle=pdAng).to(device)
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.zeros(nblks,nAngles,dtype=datatype,device=device)
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            R_iblk = R[iblk,:,:]
            dRdW_iblk = dRdW[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R_iblk.T @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk @ X_iblk))
            else:
                dLdX_iblk = R_iblk @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk.T @ X_iblk))
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.angles = angles
        target.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.orthonormalTransforms.angles.grad[:,pdAng]

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))
 
    @parameterized.expand(
        list(itertools.product(datatype,nblks,mode,nsamples,usegpu))
    )
    def testBackward4x4RandAngPdAng5(self,datatype,nblks,mode,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol=1e-4,1e-6

        # Configuration
        nPoints = 4
        pdAng = 5
        nAngles = int(nPoints*(nPoints-1)/2)
        angles = (math.pi/6)*torch.randn(nblks,nAngles)
        mus = torch.tensor([1,1,-1,-1]).repeat(nblks,1)
        omg = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=True,mode='normal')

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)
        R = omg(angles=angles,mus=mus,index_pd_angle=None).to(device)
        dRdW = omg(angles=angles,mus=mus,index_pd_angle=pdAng).to(device)
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.zeros(nblks,nAngles,dtype=datatype,device=device)
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            R_iblk = R[iblk,:,:]
            dRdW_iblk = dRdW[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R_iblk.T @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk @ X_iblk))
            else:
                dLdX_iblk = R_iblk @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk.T @ X_iblk))
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.angles = angles
        target.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.orthonormalTransforms.angles.grad[:,pdAng]

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,mode,nsamples,usegpu))
    )
    def testBackward4x4RandAngPdAng1(self,datatype,nblks,mode,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol=1e-4,1e-7

        # Configuration
        nPoints = 4
        pdAng = 1
        nAngles = int(nPoints*(nPoints-1)/2)
        angles = (math.pi/6)*torch.randn(nblks,nAngles)
        mus = torch.tensor([-1,-1,-1,-1]).repeat(nblks,1)
        omg = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=True,mode='normal')

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)
        R = omg(angles=angles,mus=mus,index_pd_angle=None).to(device)
        dRdW = omg(angles=angles,mus=mus,index_pd_angle=pdAng).to(device)
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.zeros(nblks,nAngles,dtype=datatype,device=device)
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            R_iblk = R[iblk,:,:]
            dRdW_iblk = dRdW[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R_iblk.T @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk @ X_iblk))
            else:
                dLdX_iblk = R_iblk @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk.T @ X_iblk))
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.angles = angles
        target.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.orthonormalTransforms.angles.grad[:,pdAng]
        

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,mode,nsamples,usegpu))
    )
    def testBackward8x8RandnAngPdAng4(self,datatype,nblks,mode,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol=1e-3,1e-6

        # Configuration
        nPoints = 8
        pdAng = 4
        nAngles = int(nPoints*(nPoints-1)/2)
        angles = (math.pi/6)*torch.randn(nblks,nAngles)
        mus = torch.tensor([-1,1,-1,1,-1,1,-1,1]).repeat(nblks,1)
        omg = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=True,mode='normal')

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)
        R = omg(angles=angles,mus=mus,index_pd_angle=None).to(device)
        dRdW = omg(angles=angles,mus=mus,index_pd_angle=pdAng).to(device)
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.zeros(nblks,nAngles,dtype=datatype,device=device)
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            R_iblk = R[iblk,:,:]
            dRdW_iblk = dRdW[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R_iblk.T @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk @ X_iblk))
            else:
                dLdX_iblk = R_iblk @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk.T @ X_iblk))
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.angles = angles
        target.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.orthonormalTransforms.angles.grad[:,pdAng] 

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,mode,nsamples,usegpu))
    )
    def testBackward8x8RandnAngPdAng13(self,datatype,nblks,mode,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol=1e-4,1e-6

        # Configuration
        nPoints = 8
        pdAng = 13
        nAngles = int(nPoints*(nPoints-1)/2)
        angles = (math.pi/6)*torch.randn(nblks,nAngles)
        mus = torch.tensor([1,1,1,1,-1,-1,-1,-1]).repeat(nblks,1)
        omg = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=True,mode='normal')

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)
        R = omg(angles=angles,mus=mus,index_pd_angle=None).to(device)
        dRdW = omg(angles=angles,mus=mus,index_pd_angle=pdAng).to(device)
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.zeros(nblks,nAngles,dtype=datatype,device=device)
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            R_iblk = R[iblk,:,:]
            dRdW_iblk = dRdW[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R_iblk.T @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk @ X_iblk))
            else:
                dLdX_iblk = R_iblk @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk.T @ X_iblk))
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)
            
        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.angles = angles
        target.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.orthonormalTransforms.angles.grad[:,pdAng]

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))


    @parameterized.expand(
        list(itertools.product(datatype,nblks,mode,nsamples,usegpu))
    )
    def testBackward8x8RandnAngPdAng7(self,datatype,nblks,mode,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else: 
                print("No GPU device was detected.")                
                return
        else:
            device = torch.device("cpu")
        rtol,atol=1e-4,1e-6

        # Configuration
        nPoints = 8
        pdAng = 7
        nAngles = int(nPoints*(nPoints-1)/2)
        angles = (math.pi/6.)*torch.randn(nblks,nAngles)
        mus = torch.tensor([1,-1,1,-1,1,-1,1,-1]).repeat(nblks,1)
        omg = OrthonormalMatrixGenerationSystem(dtype=datatype,partial_difference=True,mode='normal')

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)
        R = omg(angles=angles,mus=mus,index_pd_angle=None).to(device)
        dRdW = omg(angles=angles,mus=mus,index_pd_angle=pdAng).to(device)
        expctddLdX = torch.empty_like(X)
        expctddLdW = torch.zeros(nblks,nAngles,dtype=datatype,device=device)
        for iblk in range(nblks):
            dLdZ_iblk = dLdZ[iblk,:,:]
            X_iblk = X[iblk,:,:]
            R_iblk = R[iblk,:,:]
            dRdW_iblk = dRdW[iblk,:,:]
            if mode!='Synthesis':
                dLdX_iblk = R_iblk.T @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk @ X_iblk))
            else:
                dLdX_iblk = R_iblk @ dLdZ_iblk
                dLdW_iblk = torch.sum(dLdZ_iblk * (dRdW_iblk.T @ X_iblk))
            expctddLdX[iblk,:,:] = dLdX_iblk
            expctddLdW[iblk,:] = dLdW_iblk.view(-1)

        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.angles = angles
        target.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.orthonormalTransforms.angles.grad[:,pdAng]

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(datatype,nblks,npoints,mode,nsamples,usegpu))
    )
    def testBackwordNxNRandnAngMusAllPdAngs(self,datatype,nblks,npoints,mode,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print("No GPU device was detected.")
                return
        else:
            device = torch.device("cpu")
        rtol,atol=5e-2,5e-2

        # Configuration
        nPoints = npoints
        nAngles = int(nPoints*(nPoints-1)/2)
        angs0 = (math.pi/6.)*torch.randn(nblks,nAngles)
        delta = 1e-4
        mus = (-1)**torch.randint(high=2,size=(nblks,nPoints),dtype=datatype)

        # Data
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
        dLdZ = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device)        

        for pdAng in range(nAngles):
            angs1 = angs0.clone()
            angs2 = angs0.clone()
            angs1[:,pdAng] = angs0[:,pdAng]-delta/2.
            angs2[:,pdAng] = angs0[:,pdAng]+delta/2.

            # Instantiation of target class
            target0 = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
            target0.angles = angs0
            target0.mus = mus
            #
            target1 = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
            target1.angles = angs1
            target1.mus = mus
            #
            target2 = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
            target2.angles = angs2
            target2.mus = mus

            # Expected values
            if mode=='Analysis':
                bwmode='Synthesis'
            else:
                bwmode='Analysis'
            backprop = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=bwmode,device=device,dtype=datatype)
            backprop.angles = angs0
            backprop.mus = mus
            torch.autograd.set_detect_anomaly(True)
            #
            expctddLdW = torch.zeros(nblks,nAngles,dtype=datatype,device=device)
            for iblk in range(nblks):
                dLdZ_iblk = dLdZ[iblk,:,:]
                dZdW_iblk = (target2(X)[iblk,:,:] - target1(X)[iblk,:,:])/delta
                dLdW_iblk = torch.sum(dLdZ_iblk * dZdW_iblk) # ~ dLdW                
                expctddLdW[iblk,:] = dLdW_iblk.view(-1)

            # Actual values
            X.detach()
            X.requires_grad = True
            Z = target0(X)
            target0.zero_grad()
            Z.backward(dLdZ)
            actualdLdW = target0.orthonormalTransforms.angles.grad[:,pdAng]

            # Evaluation
            for iblk in range(nblks):
                err = torch.max(torch.abs(actualdLdW[iblk]-expctddLdW[iblk]))
                message = 'iblk={0},pdAng={1},err={2}'.format(iblk,pdAng,err)
                self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol),message)

    @parameterized.expand(
        list(itertools.product(nblks,npoints,mode,nsamples,usegpu))
    )
    def testGradCheckNxNRandnAngMus(self,nblks,npoints,mode,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print("No GPU device was detected.")
                return
        else:
            device = torch.device("cpu")

        # Configuration
        datatype = torch.float64
        nPoints = npoints
        nAngles = int(nPoints*(nPoints-1)/2)
        angs = (math.pi/6.)*torch.randn(nblks,nAngles,dtype=datatype,device=device)
        mus = (-1)**torch.randint(high=2,size=(nblks,nPoints),dtype=datatype,device=device)

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
        
        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device,dtype=datatype)
        target.angles = angs
        target.mus = mus
        torch.autograd.set_detect_anomaly(True)
        #Z = target(X)
        #target.zero_grad()

        # Evaluation
        self.assertTrue(torch.autograd.gradcheck(target,(X,)))

    @parameterized.expand(
        list(itertools.product(nblks,npoints,mode,nsamples,usegpu))
    )
    def testGradCheckNxNRandnAngMusToDevice(self,nblks,npoints,mode,nsamples,usegpu):
        if usegpu:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print("No GPU device was detected.")
                return
        else:
            device = torch.device("cpu")

        # Configuration
        datatype = torch.float64
        nPoints = npoints
        nAngles = int(nPoints*(nPoints-1)/2)
        angs = (math.pi/6.)*torch.randn(nblks,nAngles)
        mus = (-1)**torch.randint(high=2,size=(nblks,nPoints),dtype = datatype)

        # Expected values
        X = torch.randn(nblks,nPoints,nsamples,dtype=datatype,device=device,requires_grad=True)
               
        # Instantiation of target class
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,dtype=datatype)
        target.angles = angs
        target.mus = mus
        target = target.to(device)
        torch.autograd.set_detect_anomaly(True)
        #Z = target(X)
        #target.zero_grad()

        # Evaluation
        self.assertTrue(torch.autograd.gradcheck(target,(X,)))

if __name__ == '__main__':
    unittest.main() # failfast=True)

    """
    # Create a test suite
    suite = unittest.TestSuite()

    # Add specific test methods to the suite

    #of = [ 18 ] 

    #for i in of:
    suite.addTest(SetOfOrthonormalTransformsTestCase('testGradCheckNxNRandnAngMus_002'))

    # Create a test runner
    runner = unittest.TextTestRunner()

    # Run the tests
    runner.run(suite)
    """
    