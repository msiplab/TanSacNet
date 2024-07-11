import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import math
from random import *
from orthonormalTransform import SetOfOrthonormalTransforms
from lsunUtility import OrthonormalMatrixGenerationSystem
from lsunLayerExceptions import InvalidMode, InvalidMus

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
    
    Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
    Copyright (c) 2021-2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
        Faculty of Engineering, Niigata University,
        8050 2-no-cho Ikarashi, Nishi-ku,
        Niigata, 950-2181, JAPAN
    
        http://www.eng.niigata-u.ac.jp/~msiplab/
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
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,device=device)

        # Actual values
        with torch.no_grad():
            actualZ = target(X)
        actualNTrxs = len(target.orthonormalTransforms)
        actualMode = target.mode
        
        # Evaluation
        self.assertTrue(isinstance(target,nn.Module))        
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))    
        self.assertEqual(actualNTrxs,expctdNTrxs)
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
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks)
        target = target.to(device)

        # Actual values
        with torch.no_grad():
            actualZ = target(X)
        actualNTrxs = len(target.orthonormalTransforms)
        actualMode = target.mode
        
        # Evaluation
        self.assertTrue(isinstance(target,nn.Module))        
        self.assertTrue(torch.allclose(actualZ,expctdZ,rtol=rtol,atol=atol))    
        self.assertEqual(actualNTrxs,expctdNTrxs)
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
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,mode=mode,device=device)
        for iblk in range(nblks):
            target.orthonormalTransforms[iblk].angles = nn.init.constant_(target.orthonormalTransforms[iblk].angles,val=math.pi/4)

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
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,mode=mode)
        target = target.to(device)
        for iblk in range(nblks):
            target.orthonormalTransforms[iblk].angles = nn.init.constant_(target.orthonormalTransforms[iblk].angles,val=math.pi/4)

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
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,mode=mode,device=device)
        for iblk in range(nblks):
            target.orthonormalTransforms[iblk].angles = nn.init.constant_(target.orthonormalTransforms[iblk].angles,val=math.pi/4)
            target.orthonormalTransforms[iblk].mus = torch.tensor([1, -1])

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
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,mode=mode)
        target = target.to(device)
        for iblk in range(nblks):
            target.orthonormalTransforms[iblk].angles = nn.init.constant_(target.orthonormalTransforms[iblk].angles,val=math.pi/4)
            target.orthonormalTransforms[iblk].mus = torch.tensor([1, -1])

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
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,mode=mode,device=device)

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
        for iblk in range(nblks):
            target.orthonormalTransforms[iblk].angles.data = torch.tensor([math.pi/4]).to(device)
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
        target = SetOfOrthonormalTransforms(n=2,nblks=nblks,mode=mode)
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
        for iblk in range(nblks):
            target.orthonormalTransforms[iblk].angles.data = torch.tensor([math.pi/4]).to(device)
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
        target = SetOfOrthonormalTransforms(n=4,nblks=nblks,mode=mode,device=device)
        for iblk in range(nblks):
            target.orthonormalTransforms[iblk].angles = nn.init.normal_(target.orthonormalTransforms[iblk].angles)

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
        target = SetOfOrthonormalTransforms(n=8,nblks=nblks,mode=mode,device=device)
        for iblk in range(nblks):
            target.orthonormalTransforms[iblk].angles = nn.init.normal_(target.orthonormalTransforms[iblk].angles)

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
        target = SetOfOrthonormalTransforms(n=npoints,nblks=nblks,mode=mode,device=device)
        for iblk in range(nblks):
            target.orthonormalTransforms[iblk].angles = nn.init.normal_(target.orthonormalTransforms[iblk].angles)

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
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device)
        for iblk in range(nblks):
            target.orthonormalTransforms[iblk].angles = nn.init.uniform_(target.orthonormalTransforms[iblk].angles,a=0.0,b=2*math.pi)
            target.orthonormalTransforms[iblk].angles.data[:nPoints-1] = torch.zeros(nPoints-1)

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
        target = SetOfOrthonormalTransforms(n=nPoints,nblks=nblks,mode=mode,device=device)
        for iblk in range(nblks):
            target.orthonormalTransforms[iblk].angles = nn.init.uniform_(target.orthonormalTransforms[iblk].angles,a=0.0,b=2*math.pi)
            target.orthonormalTransforms[iblk].angles.data[:nPoints-1] = torch.zeros(nPoints-1)

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
        target = SetOfOrthonormalTransforms(n=npoints,nblks=nblks,mode=mode,device=device)
        for iblk in range(nblks):
            target.orthonormalTransforms[iblk].angles = nn.init.uniform_(target.orthonormalTransforms[iblk].angles,a=0.0,b=2*math.pi)
            target.orthonormalTransforms[iblk].angles.data[:npoints-1] = torch.zeros(npoints-1)

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
        rtol, atol = 1e-5, 1e-8

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
                dLdX_iblk = R.T @ dLdZ_iblk
                dLdW_iblk = dLdZ_iblk.T @ dRdW @ X_iblk
            else:
                dLdX_iblk = R @ dLdZ_iblk
                dLdW_iblk = dLdZ_iblk.T @ dRdW.T @ X_iblk
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
        actualdLdW = [ target.orthonormalTransforms[iblk].angles.grad for iblk in range(nblks) ]

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
        actualdLdW = [ target.orthonormalTransforms[iblk].angles.grad for iblk in range(nblks) ]

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
        actualdLdW = [ target.orthonormalTransforms[iblk].angles.grad for iblk in range(nblks) ]
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
        actualdLdW = [ target.orthonormalTransforms[iblk].angles.grad for iblk in range(nblks) ]

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
        actualdLdW = [ target.orthonormalTransforms[iblk].angles.grad for iblk in range(nblks) ]

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
        mus = [ 2 for _ in range(nblks) ]        
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
        actualdLdW = [ target.orthonormalTransforms[iblk].angles.grad for iblk in range(nblks) ]

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
        actualdLdW = [ target.orthonormalTransforms[iblk].angles.grad for iblk in range(nblks) ]

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
        rtol,atol=1e-4,1e-7

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
        rtol,atol=1e-4,1e-7

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
        actualdLdW = [ target.orthonormalTransforms[iblk].angles.grad[pdAng] for iblk in range(nblks) ]

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
        rtol,atol=1e-4,1e-7

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
        actualdLdW = [ target.orthonormalTransforms[iblk].angles.grad[pdAng] for iblk in range(nblks) ]

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
        actualdLdW = [ target.orthonormalTransforms[iblk].angles.grad[pdAng] for iblk in range(nblks) ]

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        for iblk in range(nblks):
            self.assertTrue(torch.allclose(actualdLdW[iblk],expctddLdW[iblk],rtol=rtol,atol=atol))

"""
   
    @parameterized.expand(
        list(itertools.product(mode,ncols))
    )
    def testBackward8x8RandAngPdAng4(self,mode,ncols):
        datatype=torch.double
        rtol,atol=1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode = 'Synthesis'
        nPoints = 8
        #ncols = 2
        angs0 = 2*math.pi*torch.rand(28,dtype=datatype,device=device)
        angs1 = angs0.clone()
        angs2 = angs0.clone()
        pdAng = 4
        delta = 1e-4
        angs1[pdAng] = angs0[pdAng]-delta/2.
        angs2[pdAng] = angs0[pdAng]+delta/2.

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)   
        dLdZ = dLdZ.to(device)     
        omgs = SingleOrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=False
            )
        R = omgs(angles=angs0,mus=1)
        R = R.to(device)
        dRdW = ( omgs(angles=angs2,mus=1) - omgs(angles=angs1,mus=1) )/delta
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW @ X)) 
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW.T @ X))    

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,device=device,mode=mode)
        target.angles.data = angs0

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad[pdAng]
        
        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(mode,ncols))
    )
    def testBackward8x8RandAngMusPdAng13(self,mode,ncols):
        datatype = torch.double
        rtol,atol=1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode = 'Synthesis'
        nPoints = 8
        #ncols = 2
        mus = [ 1,1,1,1,-1,-1,-1,-1 ]
        angs0 = 2*math.pi*torch.rand(28,dtype=datatype,device=device)
        angs1 = angs0.clone()
        angs2 = angs0.clone()
        pdAng = 13
        delta = 1e-4
        angs1[pdAng] = angs0[pdAng]-delta/2.
        angs2[pdAng] = angs0[pdAng]+delta/2.

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)        
        dLdZ = dLdZ.to(device)
        omgs = SingleOrthonormalMatrixGenerationSystem(
                dtype=datatype,
                partial_difference=False
            )
        R = omgs(angles=angs0,mus=mus)
        R = R.to(device)
        dRdW = ( omgs(angles=angs2,mus=mus) - omgs(angles=angs1,mus=mus) )/delta
        dRdW = dRdW.to(device)
        if mode!='Synthesis':
            expctddLdX = R.T @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW @ X)) 
        else:
            expctddLdX = R @ dLdZ # = dZdX @ dLdZ
            expctddLdW = torch.sum(dLdZ * (dRdW.T @ X))    

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,device=device,mode=mode)
        target.angles.data = angs0
        target.mus = mus

        # Actual values
        torch.autograd.set_detect_anomaly(True)        
        Z = target(X)
        target.zero_grad()
        Z.backward(dLdZ)
        actualdLdX = X.grad
        actualdLdW = target.angles.grad[pdAng]
        
        # Evaluation
        self.assertTrue(torch.allclose(actualdLdX,expctddLdX,rtol=rtol,atol=atol))
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(mode,ncols))
    )
    def testBackword8x8RandAngMusPdAng7(self,mode,ncols):
        datatype = torch.double
        rtol,atol=1e-4,1e-7
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        # Configuration
        #mode = 'Synthesis'
        nPoints = 8
        #ncols = 2
        mus = [ 1,-1,1,-1,1,-1,1,-1 ]
        angs0 = 2*math.pi*torch.rand(28,dtype=datatype).to(device)
        angs1 = angs0.clone()
        angs2 = angs0.clone()
        pdAng = 7
        delta = 1e-4
        angs1[pdAng] = angs0[pdAng]-delta/2.
        angs2[pdAng] = angs0[pdAng]+delta/2.

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=False)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)  
        dLdZ = dLdZ.to(device) 

        # Instantiation of target class
        target0 = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target0.angles.data = angs0
        target0.mus = mus
        target1 = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target1.angles.data = angs1
        target1.mus = mus
        target2 = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target2.angles.data = angs2
        target2.mus = mus    

        # Expctd values
        if mode=='Analysis':
            bwmode='Synthesis'
        else:
            bwmode='Analysis'
        backprop = OrthonormalTransform(n=nPoints,dtype=datatype,mode=bwmode)
        backprop.angles.data = angs0
        backprop.mus = mus
        torch.autograd.set_detect_anomaly(True)                
        dZdW = (target2.forward(X) - target1.forward(X))/delta # ~ d(R*X)/dW
        expctddLdW = torch.sum(dLdZ * dZdW) # ~ dLdW

        # Actual values
        X.detach()
        X.requires_grad = True
        Z = target0.forward(X)
        target0.zero_grad()
        #print(torch.autograd.gradcheck(target0,(X,angs0)))
        Z.backward(dLdZ)
        actualdLdW = target0.angles.grad[pdAng]

        # Evaluation
        self.assertTrue(torch.allclose(actualdLdW,expctddLdW,rtol=rtol,atol=atol))

    @parameterized.expand(
        list(itertools.product(mode,ncols,npoints))
    )
    def testGradCheckNxNRandAngMus(self,mode,ncols,npoints):
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
            
        # Configuration
        datatype = torch.double
        nPoints = npoints
        nAngs = int(nPoints*(nPoints-1)/2.)
        angs = 2.*math.pi*torch.randn(nAngs,dtype=datatype).to(device)
        mus = (-1)**torch.randint(high=2,size=(nPoints,))

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)   
        dLdZ = dLdZ.to(device)

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target.angles.data = angs
        target.mus = mus
        torch.autograd.set_detect_anomaly(True)                
        Z = target(X)
        target.zero_grad()

        # Evaluation        
        self.assertTrue(torch.autograd.gradcheck(target,(X,)))

    @parameterized.expand(
        list(itertools.product(mode,ncols,npoints))
    )
    def testGradCheckNxNRandAngMusToDevice(self,mode,ncols,npoints):
        if isdevicetest:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
            
        # Configuration
        datatype = torch.double
        nPoints = npoints
        nAngs = int(nPoints*(nPoints-1)/2.)
        mus = (-1)**torch.randint(high=2,size=(nPoints,))
        angs = 2.*math.pi*torch.randn(nAngs,dtype=datatype)

        # Expcted values
        X = torch.randn(nPoints,ncols,dtype=datatype,device=device,requires_grad=True)
        #X = X.to(device)
        dLdZ = torch.randn(nPoints,ncols,dtype=datatype)   
        dLdZ = dLdZ.to(device)

        # Instantiation of target class
        target = OrthonormalTransform(n=nPoints,dtype=datatype,mode=mode)
        target.angles.data = angs
        target.mus = mus
        target = target.to(device)
        torch.autograd.set_detect_anomaly(True)                
        Z = target(X)
        target.zero_grad()

        # Evaluation        
        self.assertTrue(torch.autograd.gradcheck(target,(X,)))
"""
if __name__ == '__main__':
    unittest.main()