import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import math
from random import *
from OrthonormalMatrixGenerationSystem import OrthonormalMatrixGenerationSystem

nblks = [ 1, 2, 4 ]

class OrthonormalMatrixGenerationSystemTestCase(unittest.TestCase):
    """
    ORTHONORMALMATRIXGENERATIONSYSTEMTESTCASE Test case for OrthonormalMatrixGenerationSystem
    
    Requirements: Python 3.10/11.x, PyTorch 2.3.x
    
    Copyright (c) 2024, Shogo MURAMATSU
    
    All rights reserved.
    
    Contact address: Shogo MURAMATSU,
                    Faculty of Engineering, Niigata University,
                    8050 2-no-cho Ikarashi, Nishi-ku,
                    Niigata, 950-2181, JAPAN
    
    http://www.eng.niigata-u.ac.jp/~msiplab/
    """

    # Test for default construction
    def testConstructor(self):
        rtol,atol = 1e-5,1e-8 

        # Expected values
        coefExpctd = torch.zeros(1,2,2)
        coefExpctd[0] = torch.tensor([
            [ 1., 0. ],
            [ 0., 1. ] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            coefActual = omgs(angles=0,mus=1)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol)) 

    # Test for default construction
    def testConstructorWithAngles(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(1,2,2)
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4),  math.cos(math.pi/4) ] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            coefActual = omgs(angles=math.pi/4,mus=1)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for default construction
    def testConstructorWithAnglesMultiple(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(2,2,2)
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4),  math.cos(math.pi/4) ] ])
        coefExpctd[1] = torch.tensor([
            [ math.cos(math.pi/6), -math.sin(math.pi/6) ],
            [ math.sin(math.pi/6),  math.cos(math.pi/6) ] ])

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem()
            
        # Actual values
        with torch.no_grad():
            angles = [ [math.pi/4], [math.pi/6] ]
            coefActual = omgs(angles=angles,mus=1)

        # Evaluation            
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for default construction
    def testConstructorWithAnglesAndMus(self):    
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(1,2,2)        
        coefExpctd = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ -math.sin(math.pi/4), -math.cos(math.pi/4) ] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem()            

        # Actual values
        coefActual = omgs(angles=math.pi/4,mus=[ [1,-1] ])
            
        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for default construction
    def testConstructorWithMus(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(1,2,2)
        coefExpctd[0] = torch.tensor([
            [ 1., 0. ],
            [ 0., -1. ] ])

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem()            
            
        # Actual values
        coefActual = omgs(angles=[],mus=[ [1,-1] ])

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for default construction
    def testConstructorWithMusMultiple(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(2,2,2)
        coefExpctd[0] = torch.tensor([
            [ 1., 0. ],
            [ 0., -1. ] ])
        coefExpctd[1] = torch.tensor([
            [ -1., 0. ],
            [ 0., 1. ] ])

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values            
        mus = [ [1,-1], [-1,1] ]
        coefActual = omgs(angles=[],mus=mus)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for set angle and mus 
    def testConstructionWithAnglesAndMusMultiple(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(2,2,2)
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ -math.sin(math.pi/4), -math.cos(math.pi/4) ] ])
        coefExpctd[1] = torch.tensor([
            [ -math.cos(math.pi/6), math.sin(math.pi/6) ],
            [ math.sin(math.pi/6), math.cos(math.pi/6) ] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            angles = [ [math.pi/4], [math.pi/6] ]
            mus = [ [1,-1], [-1,1] ]
            coefActual = omgs(angles=angles,mus=mus)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))


    # Test for set angle
    def testSetAngles(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(1,2,2)
        coefExpctd[0] = torch.tensor([
            [ 1., 0. ],
            [ 0., 1. ] ])

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            coefActual = omgs(angles=0,mus=1)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

        # Expected values
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4),  math.cos(math.pi/4) ] ])
        
        # Actual values
        with torch.no_grad():
            coefActual = omgs(angles=math.pi/4,mus=1)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))


    # Test for set angle for multiple blocks
    def testSetAnglesMultiple(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(2,2,2)
        coefExpctd[0] = torch.tensor([ 
            [ 1.0, 0.0  ],
            [ 0.0, 1.0 ] ])
        coefExpctd[1] = torch.tensor([ 
            [ 1.0, 0.0  ],
            [ 0.0, 1.0 ] ])

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            angles = [ [0], [0] ]
            coefActual = omgs(angles=angles,mus=1)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

        # Expected values
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4),  math.cos(math.pi/4) ] ])
        coefExpctd[1] = torch.tensor([
            [ math.cos(math.pi/6), -math.sin(math.pi/6) ],
            [ math.sin(math.pi/6),  math.cos(math.pi/6) ] ])

        # Actual values
        with torch.no_grad():
            angles = [ [math.pi/4], [math.pi/6] ]
            coefActual = omgs(angles=angles,mus=1)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for generation of multiple matrices in 2x2 blocks
    @parameterized.expand(itertools.product(nblks))
    def test2x2Multiple(self,nblks):
        rtol,atol = 1e-5,1e-8

        # Expected values
        normExpctd = torch.ones(nblks,2)

        # Instantiation of target class
        angs = 2*math.pi*torch.rand(nblks,1)
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            matrices = omgs(angles=angs,mus=1)
            normActual = torch.linalg.vector_norm(matrices,ord=2,dim=2)

        # Evaluation
        message = "normActual=" + str(normActual) + " differs from 1"
        self.assertTrue(torch.allclose(normActual,normExpctd,rtol=rtol,atol=atol), message)

    # Test for generation of single matrix in 4x4 block
    def test4x4(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        normExpctd = torch.ones(1,4)

        # Instantiation of target class
        angs = 2*math.pi*torch.rand(1,6)
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            matrix = omgs(angles=angs,mus=1)
            normActual = torch.linalg.vector_norm(matrix,ord=2,dim=2)

        # Evaluation
        message = "normActual=" + str(normActual) + " differs from 1"            
        self.assertTrue(torch.allclose(normActual,normExpctd,rtol=rtol,atol=atol), message)

    # Test for generation of multiple matrices in 4x4 blocks
    @parameterized.expand(itertools.product(nblks))
    def test4x4Multiple(self,nblks):
        rtol,atol = 1e-5,1e-8

        # Expected values
        normExpctd = torch.ones(nblks,4)

        # Instantiation of target class
        angs = 2*math.pi*torch.rand(nblks,6)            
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            matrices = omgs(angles=angs,mus=1)
            normActual = torch.linalg.vector_norm(matrices,ord=2,dim=2)

        # Evaluation
        message = "normActual=" + str(normActual) + " differs from 1"
        self.assertTrue(torch.allclose(normActual,normExpctd,rtol=rtol,atol=atol), message)

    # Test for generation of single matrix in 8x8 block
    def test8x8(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        normExpctd = torch.ones(1,8)

        # Instantiation of target class
        angs = 2*math.pi*torch.rand(1,28)
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            normActual = torch.linalg.vector_norm(omgs(angles=angs,mus=1),ord=2,dim=2)  

        # Evaluation
        message = "normActual=" + str(normActual) + " differs from 1"
        self.assertTrue(torch.allclose(normActual,normExpctd,rtol=rtol,atol=atol), message)
    
    # Test for generation of multiple matrices in 8x8 blocks
    @parameterized.expand(itertools.product(nblks))
    def test8x8Multiple(self,nblks):
        rtol, atol = 1e-5, 1e-8

        # Expected values
        normExpctd = torch.ones(nblks,8)

        # Instantiation of target class
        angs = 2*math.pi*torch.rand(nblks,28)
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            matrices = omgs(angles=angs,mus=1)
            normActual = torch.linalg.vector_norm(matrices,ord=2,dim=1)

        # Evaluation
        message = "normActual=" + str(normActual) + " differs from 1"
        self.assertTrue(torch.allclose(normActual,normExpctd,rtol=rtol,atol=atol), message)

    # Test for reduction of rotational dimension in single block of size 4x4
    def test4x4red(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        ltExpctd = torch.tensor([ 1. ])
        
        # Instantiation of target class
        angs = 2*math.pi*torch.rand(1,6)
        nSize = 4
        angs[0,0:nSize-1] = 0
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            matrices = omgs(angles=angs,mus=1)
            matrix = matrices[0]
            ltActual = matrix[0,0]

        # Evaluation
        message = "ltActual=" + str(ltActual) + " differs from 1"
        self.assertTrue(torch.allclose(ltActual,ltExpctd,rtol=rtol,atol=atol), message)

    # Test for reduction of rotational dimension in multiple blocks of size 4x4
    @parameterized.expand(itertools.product(nblks))
    def test4x4redMultiple(self,nblks):
        rtol,atol = 1e-5,1e-8

        # Expected values
        ltExpctd = torch.ones(nblks,1)

        # Instantiation of target class
        angs = 2*math.pi*torch.rand(nblks,6)
        nSize = 4
        angs[:,0:nSize-1] = 0
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            matrices = omgs(angles=angs,mus=1)
            ltActual = matrices[:,0,0]

        # Evaluation
        message = "ltActual=" + str(ltActual) + " differs from 1"
        self.assertTrue(torch.allclose(ltActual,ltExpctd,rtol=rtol,atol=atol), message)

    # Test for reduction of rotational dimension in single block of size 8x8
    def test8x8red(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        ltExpcted = torch.tensor([ 1. ])

        # Instantiation of target class
        ang = 2*math.pi*torch.rand(1,28)
        nSize = 8
        ang[0,0:nSize-1] = 0
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            matrices = omgs(angles=ang,mus=1)
            matrix = matrices[0]
            ltActual = matrix[0,0]

        # Evaluation
        message = "ltActual=" + str(ltActual) + " differs from 1"
        self.assertTrue(torch.allclose(ltActual,ltExpcted,rtol=rtol,atol=atol), message)

    # Test for reduction of rotational dimension in multiple blocks  of size 8x8
    @parameterized.expand(itertools.product(nblks))
    def test8x8redMultiple(self,nblks):
        rtol,atol = 1e-5,1e-8

        # Expected values
        ltExpcted = torch.ones(nblks,1)

        # Instantiation of target class
        angs = 2*math.pi*torch.rand(nblks,28)
        nSize = 8
        angs[:,0:nSize-1] = 0
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            matrices = omgs(angles=angs,mus=1)
            ltActual = matrices[:,0,0]

        # Evaluation
        message = "ltActual=" + str(ltActual) + " differs from 1"
        self.assertTrue(torch.allclose(ltActual,ltExpcted,rtol=rtol,atol=atol), message)

    # Test for partial difference 
    def testPartialDifference(self):
        rtol,atol = 1e-5,1e-7

        # Expected values
        coefExpctd = torch.zeros(1,2,2)
        coefExpctd[0] = torch.tensor([
            [ 0., -1. ],
            [ 1.,  0. ] ])

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        coefActual = omgs(angles=0,mus=1,index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference
    def testPartialDifferenceMultiple(self):
        rtol,atol = 1e-5,1e-7

        # Expected values
        coefExpctd = torch.zeros(2,2,2)
        coefExpctd[0] = torch.tensor([
            [ 0., -1. ],
            [ 1.,  0. ] ])
        coefExpctd[1] = torch.tensor([
            [ 0., -1. ],
            [ 1.,  0. ] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        angles = [ [0], [0] ]
        coefActual = omgs(angles=angles,mus=1,index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    
    # Test for partial difference with angles
    def testPartialDifferenceWithAngles(self):
        rtol,atol = 1e-5,1e-8
        
        # Expected values
        coefExpctd = torch.zeros(1,2,2)
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2)],
            [ math.sin(math.pi/4+math.pi/2),  math.cos(math.pi/4+math.pi/2)]])

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        coefActual = omgs(angles=math.pi/4,mus=1,index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        
    # Test for partial difference with angles
    def testPartialDifferenceWithAnglesMultiple(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(2,2,2)
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2)],
            [ math.sin(math.pi/4+math.pi/2),  math.cos(math.pi/4+math.pi/2)]])
        coefExpctd[1] = torch.tensor([
            [ math.cos(math.pi/6+math.pi/2), -math.sin(math.pi/6+math.pi/2)],
            [ math.sin(math.pi/6+math.pi/2),  math.cos(math.pi/6+math.pi/2)]])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        angles = [ [math.pi/4], [math.pi/6] ]
        coefActual = omgs(angles=angles,mus=1,index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles and mus
    def testPartialDifferenceWithAnglesAndMus(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(1,2,2)
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2)],
            [ -math.sin(math.pi/4+math.pi/2), -math.cos(math.pi/4+math.pi/2)] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        coefActual = omgs(angles=math.pi/4,mus=[ [1,-1] ],index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles and mus
    def testPartialDifferenceWithAnglesAndMusMultiple(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(2,2,2)
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2)],
            [ -math.sin(math.pi/4+math.pi/2), -math.cos(math.pi/4+math.pi/2)] ])
        coefExpctd[1] = torch.tensor([
            [ -math.cos(math.pi/6+math.pi/2), math.sin(math.pi/6+math.pi/2)],
            [ math.sin(math.pi/6+math.pi/2), math.cos(math.pi/6+math.pi/2)] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        angles = [ [math.pi/4], [math.pi/6] ]
        coefActual = omgs(angles=angles,mus=[ [1,-1], [-1,1] ],index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles 
    def testPartialDifferenceSetAngles(self):
        rtol,atol = 1e-5,1e-7

        # Expected values
        coefExpctd = torch.zeros(1,2,2)
        coefExpctd[0] = torch.tensor([
            [ 0., -1. ],
            [ 1.,  0. ] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        coefActual = omgs(angles=0,mus=1,index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

        # Expected values
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2)],
            [ math.sin(math.pi/4+math.pi/2),  math.cos(math.pi/4+math.pi/2)]])
        
        # Actual values
        coefActual = omgs(angles=math.pi/4,mus=1,index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles 
    def testPartialDifferenceSetAnglesMultiple(self):
        rtol,atol = 1e-5,1e-7

        # Expected values
        coefExpctd = torch.zeros(2,2,2)
        coefExpctd[0] = torch.tensor([
            [ 0., -1. ],
            [ 1.,  0. ] ])
        coefExpctd[1] = torch.tensor([
            [ 0., -1. ],
            [ 1.,  0. ] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        angles = [ [0], [0] ]

        coefActual = omgs(angles=angles,mus=1,index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

        # Expected values
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2)],
            [ math.sin(math.pi/4+math.pi/2),  math.cos(math.pi/4+math.pi/2)]])
        coefExpctd[1] = torch.tensor([
            [ math.cos(math.pi/6+math.pi/2), -math.sin(math.pi/6+math.pi/2)],
            [ math.sin(math.pi/6+math.pi/2),  math.cos(math.pi/6+math.pi/2)]])
        
        # Actual values
        angles = [ [math.pi/4], [math.pi/6] ]
        coefActual = omgs(angles=angles,mus=1,index_pd_angle=0)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles
    def test4x4RandAngs(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        mus = [ [-1, 1, -1, 1] ]
        angs = 2.*math.pi*torch.rand(1,6)
        coefExpctd = torch.zeros(1,4,4)
        coefExpctd[0] = torch.diag(torch.tensor(mus[0]).to(dtype=angs.dtype)) @ \
            torch.tensor([
                [ 1., 0., 0., 0. ],
                [ 0., 1., 0., 0. ],
                [ 0., 0., math.cos(angs[0,5]), -math.sin(angs[0,5]) ],
                [ 0., 0., math.sin(angs[0,5]),  math.cos(angs[0,5]) ] ]) @ \
            torch.tensor([
                [ 1., 0., 0., 0. ],
                [ 0., math.cos(angs[0,4]), 0., -math.sin(angs[0,4]) ],
                [ 0., 0., 1., 0. ], 
                [ 0., math.sin(angs[0,4]), 0., math.cos(angs[0,4]) ] ]) @ \
            torch.tensor([
                [ 1., 0., 0., 0. ],
                [ 0., math.cos(angs[0,3]), -math.sin(angs[0,3]), 0. ],
                [ 0., math.sin(angs[0,3]),  math.cos(angs[0,3]), 0. ],
                [ 0., 0., 0., 1. ] ]) @ \
            torch.tensor([
                [ math.cos(angs[0,2]), 0., 0., -math.sin(angs[0,2]) ],
                [ 0., 1., 0., 0. ],
                [ 0., 0., 1., 0. ],
                [ math.sin(angs[0,2]), 0., 0.,  math.cos(angs[0,2]) ] ]) @ \
            torch.tensor([
                [ math.cos(angs[0,1]), 0., -math.sin(angs[0,1]), 0. ],
                [ 0., 1., 0., 0. ],
                [ math.sin(angs[0,1]), 0.,  math.cos(angs[0,1]), 0. ],
                [ 0., 0., 0., 1. ] ]) @ \
            torch.tensor([
                [ math.cos(angs[0,0]), -math.sin(angs[0,0]), 0., 0. ],
                [ math.sin(angs[0,0]),  math.cos(angs[0,0]), 0., 0. ],
                [ 0., 0., 1., 0. ],
                [ 0., 0., 0., 1. ] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem()

        # Actual values
        with torch.no_grad():
            coefActual = omgs(angles=angs,mus=mus)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles
    def test4x4PartialDifference4x4RandAngPdAng2(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        mus = [ [-1, 1, -1, 1] ]
        angs = 2.*math.pi*torch.rand(1,6)
        pdAng = 2
        coefExpctd = torch.zeros(1,4,4)
        coefExpctd[0] = torch.diag(torch.tensor(mus[0]).to(dtype=angs.dtype)) @ \
            torch.tensor([
                [ 1., 0., 0., 0. ],
                [ 0., 1., 0., 0. ],
                [ 0., 0., math.cos(angs[0,5]), -math.sin(angs[0,5]) ],
                [ 0., 0., math.sin(angs[0,5]),  math.cos(angs[0,5]) ] ]) @ \
            torch.tensor([
                [ 1., 0., 0., 0. ],
                [ 0., math.cos(angs[0,4]), 0., -math.sin(angs[0,4]) ],
                [ 0., 0., 1., 0. ], 
                [ 0., math.sin(angs[0,4]), 0., math.cos(angs[0,4]) ] ]) @ \
            torch.tensor([
                [ 1., 0., 0., 0. ],
                [ 0., math.cos(angs[0,3]), -math.sin(angs[0,3]), 0. ],
                [ 0., math.sin(angs[0,3]),  math.cos(angs[0,3]), 0. ],
                [ 0., 0., 0., 1. ] ]) @ \
            torch.tensor([ # Partial Diff.
                [ math.cos(angs[0,2]+math.pi/2.), 0., 0., -math.sin(angs[0,2]+math.pi/2.) ],
                [ 0., 0., 0., 0. ],
                [ 0., 0., 0., 0. ],
                [ math.sin(angs[0,2]+math.pi/2.), 0., 0.,  math.cos(angs[0,2]+math.pi/2.) ] ]) @ \
            torch.tensor([
                [ math.cos(angs[0,1]), 0., -math.sin(angs[0,1]), 0. ],
                [ 0., 1., 0., 0. ],
                [ math.sin(angs[0,1]), 0.,  math.cos(angs[0,1]), 0. ],
                [ 0., 0., 0., 1. ] ]) @ \
            torch.tensor([
                [ math.cos(angs[0,0]), -math.sin(angs[0,0]), 0., 0. ],
                [ math.sin(angs[0,0]),  math.cos(angs[0,0]), 0., 0. ],
                [ 0., 0., 1., 0. ],
                [ 0., 0., 0., 1. ] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles
    @parameterized.expand(itertools.product(nblks))
    def testPartialDifference4x4RandAngPdAng2Multiple(self,nblks):
        rtol,atol = 1e-5,1e-7

        # Expected values
        mus = [ [-1, 1, -1, 1] ]
        angs = 2.*math.pi*torch.rand(nblks,6)
        pdAng = 2
        coefExpctd = torch.zeros(nblks,4,4)
        for iblk in range(nblks):
            coefExpctd[iblk] = torch.diag(torch.tensor(mus[0]).to(dtype=angs.dtype)) @ \
                torch.tensor([
                    [ 1., 0., 0., 0. ],
                    [ 0., 1., 0., 0. ],
                    [ 0., 0., math.cos(angs[iblk,5]), -math.sin(angs[iblk,5]) ],
                    [ 0., 0., math.sin(angs[iblk,5]),  math.cos(angs[iblk,5]) ] ]) @ \
                torch.tensor([
                    [ 1., 0., 0., 0. ],
                    [ 0., math.cos(angs[iblk,4]), 0., -math.sin(angs[iblk,4]) ],
                    [ 0., 0., 1., 0. ], 
                    [ 0., math.sin(angs[iblk,4]), 0., math.cos(angs[iblk,4]) ] ]) @ \
                torch.tensor([
                    [ 1., 0., 0., 0. ],
                    [ 0., math.cos(angs[iblk,3]), -math.sin(angs[iblk,3]), 0. ],
                    [ 0., math.sin(angs[iblk,3]),  math.cos(angs[iblk,3]), 0. ],
                    [ 0., 0., 0., 1. ] ]) @ \
                torch.tensor([ # Partial Diff.
                    [ math.cos(angs[iblk,2]+math.pi/2.), 0., 0., -math.sin(angs[iblk,2]+math.pi/2.) ],
                    [ 0., 0., 0., 0. ],
                    [ 0., 0., 0., 0. ],
                    [ math.sin(angs[iblk,2]+math.pi/2.), 0., 0.,  math.cos(angs[iblk,2]+math.pi/2.) ] ]) @ \
                torch.tensor([
                    [ math.cos(angs[iblk,1]), 0., -math.sin(angs[iblk,1]), 0. ],
                    [ 0., 1., 0., 0. ],
                    [ math.sin(angs[iblk,1]), 0.,  math.cos(angs[iblk,1]), 0. ],
                    [ 0., 0., 0., 1. ] ]) @ \
                torch.tensor([
                    [ math.cos(angs[iblk,0]), -math.sin(angs[iblk,0]), 0., 0. ],
                    [ math.sin(angs[iblk,0]),  math.cos(angs[iblk,0]), 0., 0. ],
                    [ 0., 0., 1., 0. ],
                    [ 0., 0., 0., 1. ] ])
            
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles
    def testPartialDifference4x4RandAngPdAng5(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        mus = [ [-1, 1, -1, 1] ]
        angs = 2.*math.pi*torch.rand(1,6)
        pdAng = 5
        coefExpctd = torch.zeros(1,4,4)
        coefExpctd[0] = torch.diag(torch.tensor(mus[0]).to(dtype=angs.dtype)) @ \
            torch.tensor([ # Partial Diff.
                [ 0., 0., 0., 0. ],
                [ 0., 0., 0., 0. ],
                [ 0., 0., math.cos(angs[0,5]+math.pi/2.), -math.sin(angs[0,5]+math.pi/2.) ],
                [ 0., 0., math.sin(angs[0,5]+math.pi/2.),  math.cos(angs[0,5]+math.pi/2.) ] ]) @ \
            torch.tensor([
                [ 1., 0., 0., 0. ],
                [ 0., math.cos(angs[0,4]), 0., -math.sin(angs[0,4]) ],
                [ 0., 0., 1., 0. ], 
                [ 0., math.sin(angs[0,4]), 0., math.cos(angs[0,4]) ] ]) @ \
            torch.tensor([
                [ 1., 0., 0., 0. ],
                [ 0., math.cos(angs[0,3]), -math.sin(angs[0,3]), 0. ],
                [ 0., math.sin(angs[0,3]),  math.cos(angs[0,3]), 0. ],
                [ 0., 0., 0., 1. ] ]) @ \
            torch.tensor([
                [ math.cos(angs[0,2]), 0., 0., -math.sin(angs[0,2]) ],
                [ 0., 1., 0., 0. ],
                [ 0., 0., 1., 0. ],
                [ math.sin(angs[0,2]), 0., 0.,  math.cos(angs[0,2]) ] ]) @ \
            torch.tensor([
                [ math.cos(angs[0,1]), 0., -math.sin(angs[0,1]), 0. ],
                [ 0., 1., 0., 0. ],
                [ math.sin(angs[0,1]), 0.,  math.cos(angs[0,1]), 0. ],
                [ 0., 0., 0., 1. ] ]) @ \
            torch.tensor([
                [ math.cos(angs[0,0]), -math.sin(angs[0,0]), 0., 0. ],
                [ math.sin(angs[0,0]),  math.cos(angs[0,0]), 0., 0. ],
                [ 0., 0., 1., 0. ],
                [ 0., 0., 0., 1. ] ])

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles
    @parameterized.expand(itertools.product(nblks))
    def testPartialDifference4x4RandAngPdAng5Multiple(self,nblks):
        rtol,atol = 1e-5,1e-7

        # Expected values
        mus = [ [1, 1, -1, -1] ]
        angs = 2.*math.pi*torch.rand(nblks,6)
        pdAng = 5
        coefExpctd = torch.zeros(nblks,4,4)
        for iblk in range(nblks):
            coefExpctd[iblk] = torch.diag(torch.tensor(mus[0]).to(dtype=angs.dtype)) @ \
                torch.tensor([ # Partial Diff.
                    [ 0., 0., 0., 0. ],
                    [ 0., 0., 0., 0. ],
                    [ 0., 0., math.cos(angs[iblk,5]+math.pi/2.), -math.sin(angs[iblk,5]+math.pi/2.) ],
                    [ 0., 0., math.sin(angs[iblk,5]+math.pi/2.),  math.cos(angs[iblk,5]+math.pi/2.) ] ]) @ \
                torch.tensor([
                    [ 1., 0., 0., 0. ],
                    [ 0., math.cos(angs[iblk,4]), 0., -math.sin(angs[iblk,4]) ],
                    [ 0., 0., 1., 0. ], 
                    [ 0., math.sin(angs[iblk,4]), 0., math.cos(angs[iblk,4]) ] ]) @ \
                torch.tensor([
                    [ 1., 0., 0., 0. ],
                    [ 0., math.cos(angs[iblk,3]), -math.sin(angs[iblk,3]), 0. ],
                    [ 0., math.sin(angs[iblk,3]),  math.cos(angs[iblk,3]), 0. ],
                    [ 0., 0., 0., 1. ] ]) @ \
                torch.tensor([
                    [ math.cos(angs[iblk,2]), 0., 0., -math.sin(angs[iblk,2]) ],
                    [ 0., 1., 0., 0. ],
                    [ 0., 0., 1., 0. ],
                    [ math.sin(angs[iblk,2]), 0., 0.,  math.cos(angs[iblk,2]) ] ]) @ \
                torch.tensor([
                    [ math.cos(angs[iblk,1]), 0., -math.sin(angs[iblk,1]), 0. ],
                    [ 0., 1., 0., 0. ],
                    [ math.sin(angs[iblk,1]), 0.,  math.cos(angs[iblk,1]), 0. ],
                    [ 0., 0., 0., 1. ] ]) @ \
                torch.tensor([
                    [ math.cos(angs[iblk,0]), -math.sin(angs[iblk,0]), 0., 0. ],
                    [ math.sin(angs[iblk,0]),  math.cos(angs[iblk,0]), 0., 0. ],
                    [ 0., 0., 1., 0. ],
                    [ 0., 0., 0., 1. ] ])
            
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles
    def testPartialDifference4x4RandAngPdAng2(self):
        rtol,atol = 1e-3, 1e-3

        # Expected values
        mus = [ [-1,  -1, -1, -1 ] ]
        angs = 2.*math.pi*torch.rand(1,6)
        pdAng = 2
        delta = 1.e-3
        coefExpctd = 1./delta * \
            torch.diag(torch.tensor(mus[0]).to(dtype=angs.dtype))  @ \
            torch.tensor([
                [ 1., 0., 0., 0. ],
                [ 0., 1., 0., 0. ],
                [ 0., 0., math.cos(angs[0,5]), -math.sin(angs[0,5]) ],
                [ 0., 0., math.sin(angs[0,5]),  math.cos(angs[0,5]) ] ]) @ \
            torch.tensor([
                [ 1., 0., 0., 0. ],
                [ 0., math.cos(angs[0,4]), 0., -math.sin(angs[0,4]) ],
                [ 0., 0., 1., 0. ], 
                [ 0., math.sin(angs[0,4]), 0., math.cos(angs[0,4]) ] ]) @ \
            torch.tensor([
                [ 1., 0., 0., 0. ],
                [ 0., math.cos(angs[0,3]), -math.sin(angs[0,3]), 0. ],
                [ 0., math.sin(angs[0,3]),  math.cos(angs[0,3]), 0. ],
                [ 0., 0., 0., 1. ] ]) @ \
            ( torch.tensor([
                [ math.cos(angs[0,2]+delta), 0., 0., -math.sin(angs[0,2]+delta) ],
                [ 0., 1., 0., 0. ],
                [ 0., 0., 1., 0. ],
                [ math.sin(angs[0,2]+delta), 0., 0.,  math.cos(angs[0,2]+delta) ] ]) - \
              torch.tensor([
                [ math.cos(angs[0,2]), 0., 0., -math.sin(angs[0,2]) ],
                [ 0., 1., 0., 0. ],
                [ 0., 0., 1., 0. ],
                [ math.sin(angs[0,2]), 0., 0.,  math.cos(angs[0,2]) ] ]) ) @ \
            torch.tensor([
                [ math.cos(angs[0,1]), 0., -math.sin(angs[0,1]), 0. ],
                [ 0., 1., 0., 0. ],
                [ math.sin(angs[0,1]), 0.,  math.cos(angs[0,1]), 0. ],
                [ 0., 0., 0., 1. ] ]) @ \
            torch.tensor([
                [ math.cos(angs[0,0]), -math.sin(angs[0,0]), 0., 0. ],
                [ math.sin(angs[0,0]),  math.cos(angs[0,0]), 0., 0. ],
                [ 0., 0., 1., 0. ],
                [ 0., 0., 0., 1. ] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles
    @parameterized.expand(itertools.product(nblks))
    def testPartialDifference4x4RandAngPdAng2Multiple(self,nblks):
        rtol,atol = 1e-3, 1e-3

        # Expected values
        mus = [ [-1,  -1, -1, -1 ] ]        
        angs = 2.*math.pi*torch.rand(nblks,6)
        pdAng = 2
        delta = 1.e-3
        coefExpctd = torch.zeros(nblks,4,4)
        for iblk in range(nblks):
            coefExpctd[iblk] = 1./delta * \
                torch.diag(torch.tensor(mus[0]).to(dtype=angs.dtype))  @ \
                torch.tensor([
                    [ 1., 0., 0., 0. ],
                    [ 0., 1., 0., 0. ],
                    [ 0., 0., math.cos(angs[iblk,5]), -math.sin(angs[iblk,5]) ],
                    [ 0., 0., math.sin(angs[iblk,5]),  math.cos(angs[iblk,5]) ] ]) @ \
                torch.tensor([
                    [ 1., 0., 0., 0. ],
                    [ 0., math.cos(angs[iblk,4]), 0., -math.sin(angs[iblk,4]) ],
                    [ 0., 0., 1., 0. ], 
                    [ 0., math.sin(angs[iblk,4]), 0., math.cos(angs[iblk,4]) ] ]) @ \
                torch.tensor([
                    [ 1., 0., 0., 0. ],
                    [ 0., math.cos(angs[iblk,3]), -math.sin(angs[iblk,3]), 0. ],
                    [ 0., math.sin(angs[iblk,3]),  math.cos(angs[iblk,3]), 0. ],
                    [ 0., 0., 0., 1. ] ]) @ \
                ( torch.tensor([
                    [ math.cos(angs[iblk,2]+delta), 0., 0., -math.sin(angs[iblk,2]+delta) ],
                    [ 0., 1., 0., 0. ],
                    [ 0., 0., 1., 0. ],
                    [ math.sin(angs[iblk,2]+delta), 0., 0.,  math.cos(angs[iblk,2]+delta) ] ]) - \
                  torch.tensor([
                    [ math.cos(angs[iblk,2]), 0., 0., -math.sin(angs[iblk,2]) ],
                    [ 0., 1., 0., 0. ],
                    [ 0., 0., 1., 0. ],
                    [ math.sin(angs[iblk,2]), 0., 0.,  math.cos(angs[iblk,2]) ] ]) ) @ \
                torch.tensor([
                    [ math.cos(angs[iblk,1]), 0., -math.sin(angs[iblk,1]), 0. ],
                    [ 0., 1., 0., 0. ],
                    [ math.sin(angs[iblk,1]), 0.,  math.cos(angs[iblk,1]), 0. ],
                    [ 0., 0., 0., 1. ] ]) @ \
                torch.tensor([
                    [ math.cos(angs[iblk,0]), -math.sin(angs[iblk,0]), 0., 0. ],
                    [ math.sin(angs[iblk,0]),  math.cos(angs[iblk,0]), 0., 0. ],
                    [ 0., 0., 1., 0. ],
                    [ 0., 0., 0., 1. ] ])
            
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        
    # Test for partial difference with angles
    def testPartialDifference8x8RandAngPdAng2(self):
        rtol,atol = 1e-3,1e-3

        # Expected values
        pdAng = 13
        delta = 1.e-3
        angs0 = 2.*math.pi*torch.rand(1,28)
        angs1 = angs0.clone()
        angs1[0,pdAng] += delta

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=False)
        coefExpctd = 1./delta * \
            ( omgs(angles=angs1,mus=1) - omgs(angles=angs0,mus=1) )
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        coefActual = omgs(angles=angs0,mus=1,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

        
    # Test for partial difference with angles
    @parameterized.expand(itertools.product(nblks))
    def testPartialDifference8x8RandAngPdAng2Multiple(self,nblks):
        rtol,atol = 1e-3,1e-3

        # Expected values
        pdAng = 13
        delta = 1.e-3
        angs0 = 2.*math.pi*torch.rand(nblks,28)
        angs1 = angs0.clone()
        angs1[:,pdAng] += delta

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=False)
        coefExpctd = 1./delta * \
            ( omgs(angles=angs1,mus=1) - omgs(angles=angs0,mus=1) )
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True)

        # Actual values
        coefActual = omgs(angles=angs0,mus=1,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles
    def testPartialDifferenceInSequentialMode(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(1,2,2)
        coefExpctd[0] = torch.tensor([
            [ -1., 0. ],
            [ 0., -1. ] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        coefActual = omgs(angles=0,mus=-1,index_pd_angle=None)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles
    def testPartialDifferenceInSequentialModeMultiple(self):
        rtol,atol = 1e-5,1e-8

        # Expected values
        coefExpctd = torch.zeros(2,2,2)
        coefExpctd[0] = torch.tensor([
            [ -1., 0. ],
            [ 0., -1. ] ])
        coefExpctd[1] = torch.tensor([
            [ -1., 0. ],
            [ 0., -1. ] ])
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        coefActual = omgs(angles=0,mus=-1,index_pd_angle=None)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles
    def testPartialDifferenceWithAngsInSequentialModeInitalization(self):
        rtol,atol = 1e-5,1e-8

        # Configuration
        pdAng = None

        # Expected values
        coefExpctd = torch.zeros(1,2,2)
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4), -math.sin(math.pi/4) ],
            [ math.sin(math.pi/4),  math.cos(math.pi/4) ] ])
        nextangleExpctd = 0
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        coefActual = omgs(angles=math.pi/4,mus=1,index_pd_angle=pdAng)
        nextangleActual = omgs.nextangle

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        self.assertEqual(nextangleActual,nextangleExpctd)

    # Test for partial difference with angles
    def testPartialDifferenceWithAngsInSequentialMode(self):
        rtol,atol = 1e-5,1e-8

        # Configuration
        pdAng = 0

        # Expected values
        coefExpctd = torch.zeros(1,2,2)
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2) ],
            [ math.sin(math.pi/4+math.pi/2),  math.cos(math.pi/4+math.pi/2) ] ])
        nextangleExpctd = 1        
        
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        omgs(angles=math.pi/4,mus=1,index_pd_angle=None)
        coefActual = omgs(angles=math.pi/4,mus=1,index_pd_angle=pdAng)
        nextangleActual = omgs.nextangle

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        self.assertEqual(nextangleActual,nextangleExpctd)        

    # Test for partial difference with angles
    def testPartialDifferenceWithAngsInSequentialModeMultiple(self):
        rtol,atol = 1e-5,1e-8

        # Configuration
        pdAng = 0

        # Expected values
        coefExpctd = torch.zeros(2,2,2)
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2) ],
            [ math.sin(math.pi/4+math.pi/2),  math.cos(math.pi/4+math.pi/2) ] ])
        coefExpctd[1] = torch.tensor([
            [ math.cos(math.pi/6+math.pi/2), -math.sin(math.pi/6+math.pi/2) ],
            [ math.sin(math.pi/6+math.pi/2),  math.cos(math.pi/6+math.pi/2) ] ])
        nextangleExpctd = 1

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        angs = [ [math.pi/4], [math.pi/6] ]
        omgs(angles=angs,mus=1,index_pd_angle=None)
        coefActual = omgs(angles=angs,mus=1,index_pd_angle=pdAng)
        nextangleActual = omgs.nextangle

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        self.assertEqual(nextangleActual,nextangleExpctd)

        """
        %
        function testPartialDifferenceWithAnglesAndMusInSequentialMode(testCase)
             
            % Configuration
            pdAng = 1;
            
            % Expected values
            coefExpctd = [
                cos(pi/4+pi/2) -sin(pi/4+pi/2) ;
                -sin(pi/4+pi/2) -cos(pi/4+pi/2) ];
            
            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');
            
            % Actual values
            step(testCase.omgs,pi/4,[ 1 -1 ],0);                        
            coefActual = step(testCase.omgs,pi/4,[ 1 -1 ],pdAng);            
            
            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);
            
        end

        %
        function testPartialDifferenceWithAnglesAndMusInSequentialModeMultiple(testCase)
             
            % Configuration
            pdAng = 1;
            
            % Expected values
            coefExpctd(:,:,1) = [
                cos(pi/4+pi/2) -sin(pi/4+pi/2) ;
                -sin(pi/4+pi/2) -cos(pi/4+pi/2) ];
            coefExpctd(:,:,2) = [
                cos(pi/6+pi/2) -sin(pi/6+pi/2) ;
                -sin(pi/6+pi/2) -cos(pi/6+pi/2) ];            
            
            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');
            
            % Actual values
            angs = [pi/4 pi/6];
            mus = [1 -1];
            step(testCase.omgs,angs,mus,0);                        
            coefActual = step(testCase.omgs,angs,mus,pdAng);            
            
            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);
            
        end       
        
        function testPartialDifference4x4RandAngPdAng3InSequentialMode(testCase)
          
            % Expected values
            mus = [ -1 1 -1 1 ];
            angs = 2*pi*rand(6,1);
            pdAng = 3;
            coefExpctd = ...
                diag(mus) * ...
               [ 1  0   0             0            ;
                 0  1   0             0            ;
                 0  0   cos(angs(6)) -sin(angs(6)) ;
                 0  0   sin(angs(6))  cos(angs(6)) ] *...                                                                            
               [ 1  0            0  0            ;
                 0  cos(angs(5)) 0 -sin(angs(5)) ;
                 0  0            1  0            ;
                 0  sin(angs(5)) 0 cos(angs(5))  ] *...                                                            
               [ 1  0             0            0 ;
                 0  cos(angs(4)) -sin(angs(4)) 0 ;
                 0  sin(angs(4))  cos(angs(4)) 0 ;
                 0  0             0            1 ] *...                                            
               [ cos(angs(3)+pi/2) 0 0 -sin(angs(3)+pi/2)  ; % Partial Diff.
                 0            0 0  0             ;
                 0            0 0  0             ;        
                 sin(angs(3)+pi/2) 0 0  cos(angs(3)+pi/2) ] *...                            
               [ cos(angs(2)) 0 -sin(angs(2)) 0  ;
                 0            1  0            0  ; 
                 sin(angs(2)) 0  cos(angs(2)) 0  ;
                 0            0  0            1 ] *...            
               [ cos(angs(1)) -sin(angs(1)) 0 0  ;
                 sin(angs(1)) cos(angs(1))  0 0  ;
                 0            0             1 0  ;
                 0            0             0 1 ];
            
            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');
            
            % Actual values
            for iAng = 0:pdAng-1
                testCase.omgs.step(angs,mus,iAng);            
            end
            coefActual = testCase.omgs.step(angs,mus,pdAng);            
            
            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end
        
        
        function testPartialDifference4x4RandAngPdAng3InSequentialModeMultiple(testCase,nblks)
          
            % Expected values
            mus = [ -1 1 -1 1 ];
            angs = 2*pi*rand(6,nblks);
            pdAng = 3;
            coefExpctd = zeros(4,4,nblks);
            for iblk = 1:nblks
                coefExpctd(:,:,iblk) = ...
                    diag(mus) * ...
                    [ 1  0   0             0            ;
                    0  1   0             0            ;
                    0  0   cos(angs(6,iblk)) -sin(angs(6,iblk)) ;
                    0  0   sin(angs(6,iblk))  cos(angs(6,iblk)) ] *...
                    [ 1  0            0  0            ;
                    0  cos(angs(5,iblk)) 0 -sin(angs(5,iblk)) ;
                    0  0            1  0            ;
                    0  sin(angs(5,iblk)) 0 cos(angs(5,iblk))  ] *...
                    [ 1  0             0            0 ;
                    0  cos(angs(4,iblk)) -sin(angs(4,iblk)) 0 ;
                    0  sin(angs(4,iblk))  cos(angs(4,iblk)) 0 ;
                    0  0             0            1 ] *...
                    [ cos(angs(3,iblk)+pi/2) 0 0 -sin(angs(3,iblk)+pi/2)  ; % Partial Diff.
                    0            0 0  0             ;
                    0            0 0  0             ;
                    sin(angs(3,iblk)+pi/2) 0 0  cos(angs(3,iblk)+pi/2) ] *...
                    [ cos(angs(2,iblk)) 0 -sin(angs(2,iblk)) 0  ;
                    0            1  0            0  ;
                    sin(angs(2,iblk)) 0  cos(angs(2,iblk)) 0  ;
                    0            0  0            1 ] *...
                    [ cos(angs(1,iblk)) -sin(angs(1,iblk)) 0 0  ;
                    sin(angs(1,iblk)) cos(angs(1,iblk))  0 0  ;
                    0            0             1 0  ;
                    0            0             0 1 ];
            end

            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');
            
            % Actual values
            for iAng = 0:pdAng-1
                testCase.omgs.step(angs,mus,iAng);            
            end
            coefActual = testCase.omgs.step(angs,mus,pdAng);            
            
            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);

        end

        %
        function testPartialDifference4x4RandAngPdAng6InSequentialMode(testCase)
 
            % Expected values
            mus = [ 1 1 -1 -1 ];
            angs = 2*pi*rand(6,1);
            pdAng = 6;
            coefExpctd = ...
                diag(mus) * ...
               [ 0  0   0             0            ;
                 0  0   0             0            ;
                 0  0   cos(angs(6)+pi/2) -sin(angs(6)+pi/2) ; % Partial Diff.
                 0  0   sin(angs(6)+pi/2)  cos(angs(6)+pi/2) ] *...                                                                            
               [ 1  0            0  0            ;
                 0  cos(angs(5)) 0 -sin(angs(5)) ;
                 0  0            1  0            ;
                 0  sin(angs(5)) 0 cos(angs(5))  ] *...                                                            
               [ 1  0             0            0 ;
                 0  cos(angs(4)) -sin(angs(4)) 0 ;
                 0  sin(angs(4))  cos(angs(4)) 0 ;
                 0  0             0            1 ] *...                                            
               [ cos(angs(3)) 0 0 -sin(angs(3))  ; 
                 0            1 0  0             ;
                 0            0 1  0             ;        
                 sin(angs(3)) 0 0  cos(angs(3)) ] *...                            
               [ cos(angs(2)) 0 -sin(angs(2)) 0  ;
                 0            1  0            0  ; 
                 sin(angs(2)) 0  cos(angs(2)) 0  ;
                 0            0  0            1 ] *...            
               [ cos(angs(1)) -sin(angs(1)) 0 0  ;
                 sin(angs(1)) cos(angs(1))  0 0  ;
                 0            0             1 0  ;
                 0            0             0 1 ];
            
            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');
            
            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs,mus,iAng);            
            end
            coefActual = testCase.omgs.step(angs,mus,pdAng);            
            
            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);
            
        end

        function testPartialDifference4x4RandAngPdAng6InSequentialModeMultiple(testCase,nblks)
 
            % Expected values
            mus = [ 1 1 -1 -1 ];
            angs = 2*pi*rand(6,nblks);
            pdAng = 6;
            coefExpctd = zeros(4,4,nblks);
            for iblk = 1:nblks
                coefExpctd(:,:,iblk) = ...
                    diag(mus) * ...
                    [ 0  0   0             0            ;
                    0  0   0             0            ;
                    0  0   cos(angs(6,iblk)+pi/2) -sin(angs(6,iblk)+pi/2) ; % Partial Diff.
                    0  0   sin(angs(6,iblk)+pi/2)  cos(angs(6,iblk)+pi/2) ] *...
                    [ 1  0            0  0            ;
                    0  cos(angs(5,iblk)) 0 -sin(angs(5,iblk)) ;
                    0  0            1  0            ;
                    0  sin(angs(5,iblk)) 0 cos(angs(5,iblk))  ] *...
                    [ 1  0             0            0 ;
                    0  cos(angs(4,iblk)) -sin(angs(4,iblk)) 0 ;
                    0  sin(angs(4,iblk))  cos(angs(4,iblk)) 0 ;
                    0  0             0            1 ] *...
                    [ cos(angs(3,iblk)) 0 0 -sin(angs(3,iblk))  ;
                    0            1 0  0             ;
                    0            0 1  0             ;
                    sin(angs(3,iblk)) 0 0  cos(angs(3,iblk)) ] *...
                    [ cos(angs(2,iblk)) 0 -sin(angs(2,iblk)) 0  ;
                    0            1  0            0  ;
                    sin(angs(2,iblk)) 0  cos(angs(2,iblk)) 0  ;
                    0            0  0            1 ] *...
                    [ cos(angs(1,iblk)) -sin(angs(1,iblk)) 0 0  ;
                    sin(angs(1,iblk)) cos(angs(1,iblk))  0 0  ;
                    0            0             1 0  ;
                    0            0             0 1 ];
            end

            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');
            
            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs,mus,iAng);            
            end
            coefActual = testCase.omgs.step(angs,mus,pdAng);            
            
            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-10);
            
        end        

        %
        function testPartialDifference4x4RandAngPdAng2InSequentialMode(testCase)
   
            % Expected values
            mus = [ -1 -1 -1 -1 ];
            angs = 2*pi*rand(6,1);
            pdAng = 2;
            delta = 1e-10;
            coefExpctd = 1/delta * ...
                diag(mus) * ...
               [ 1  0   0             0            ;
                 0  1   0             0            ;
                 0  0   cos(angs(6)) -sin(angs(6)) ; 
                 0  0   sin(angs(6))  cos(angs(6)) ] *...                                                                            
               [ 1  0            0  0            ;
                 0  cos(angs(5)) 0 -sin(angs(5)) ;
                 0  0            1  0            ;
                 0  sin(angs(5)) 0 cos(angs(5))  ] *...                                                            
               [ 1  0             0            0 ;
                 0  cos(angs(4)) -sin(angs(4)) 0 ;
                 0  sin(angs(4))  cos(angs(4)) 0 ;
                 0  0             0            1 ] *...                                            
               [ cos(angs(3)) 0 0 -sin(angs(3))  ; 
                 0            1 0  0             ;
                 0            0 1  0             ;        
                 sin(angs(3)) 0 0  cos(angs(3)) ] * ...                           
            ( ...
               [ cos(angs(2)+delta) 0 -sin(angs(2)+delta) 0  ; 
                 0            1  0            0  ; 
                 sin(angs(2)+delta) 0  cos(angs(2)+delta) 0  ;
                 0            0  0            1 ] - ...
               [ cos(angs(2)) 0 -sin(angs(2)) 0  ; 
                 0            1  0            0  ; 
                 sin(angs(2)) 0  cos(angs(2)) 0  ;
                 0            0  0            1 ] ...                 
             ) *...            
               [ cos(angs(1)) -sin(angs(1)) 0 0  ;
                 sin(angs(1)) cos(angs(1))  0 0  ;
                 0            0             1 0  ;
                 0            0             0 1 ];
            
            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');
            
            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs,mus,iAng);            
            end
            coefActual = step(testCase.omgs,angs,mus,pdAng);            
            
            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);
            
        end
        
        function testPartialDifference4x4RandAngPdAng2InSequentialModeMultiple(testCase,nblks)
   
            % Expected values
            mus = [ -1 -1 -1 -1 ];
            angs = 2*pi*rand(6,nblks);
            pdAng = 2;
            delta = 1e-10;
            coefExpctd = zeros(4,4,nblks);
            for iblk = 1:nblks
            coefExpctd(:,:,iblk) = 1/delta * ...
                diag(mus) * ...
               [ 1  0   0             0            ;
                 0  1   0             0            ;
                 0  0   cos(angs(6,iblk)) -sin(angs(6,iblk)) ; 
                 0  0   sin(angs(6,iblk))  cos(angs(6,iblk)) ] *...                                                                            
               [ 1  0            0  0            ;
                 0  cos(angs(5,iblk)) 0 -sin(angs(5,iblk)) ;
                 0  0            1  0            ;
                 0  sin(angs(5,iblk)) 0 cos(angs(5,iblk))  ] *...                                                            
               [ 1  0             0            0 ;
                 0  cos(angs(4,iblk)) -sin(angs(4,iblk)) 0 ;
                 0  sin(angs(4,iblk))  cos(angs(4,iblk)) 0 ;
                 0  0             0            1 ] *...                                            
               [ cos(angs(3,iblk)) 0 0 -sin(angs(3,iblk))  ; 
                 0            1 0  0             ;
                 0            0 1  0             ;        
                 sin(angs(3,iblk)) 0 0  cos(angs(3,iblk)) ] * ...                           
            ( ...
               [ cos(angs(2,iblk)+delta) 0 -sin(angs(2,iblk)+delta) 0  ; 
                 0            1  0            0  ; 
                 sin(angs(2,iblk)+delta) 0  cos(angs(2,iblk)+delta) 0  ;
                 0            0  0            1 ] - ...
               [ cos(angs(2,iblk)) 0 -sin(angs(2,iblk)) 0  ; 
                 0            1  0            0  ; 
                 sin(angs(2,iblk)) 0  cos(angs(2,iblk)) 0  ;
                 0            0  0            1 ] ...                 
             ) *...            
               [ cos(angs(1,iblk)) -sin(angs(1,iblk)) 0 0  ;
                 sin(angs(1,iblk)) cos(angs(1,iblk))  0 0  ;
                 0            0             1 0  ;
                 0            0             0 1 ];
            end

            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');
            
            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs,mus,iAng);            
            end
            coefActual = step(testCase.omgs,angs,mus,pdAng);            
            
            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);
            
        end
        

% TODO test for multiple blocks     


        %
        function testPartialDifference8x8RandAngPdAng2InSequentialMode(testCase)
  
            % Expected values
            pdAng = 14;            
            delta = 1e-10;            
            angs0 = 2*pi*rand(28,1);
            angs1 = angs0;
            angs1(pdAng) = angs1(pdAng)+delta;
            
            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','off');            
            coefExpctd = 1/delta * ...
                ( step(testCase.omgs,angs1,1) ...
                - step(testCase.omgs,angs0,1));
            
            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');
            
            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs0,1,iAng);            
            end
            coefActual = step(testCase.omgs,angs0,1,pdAng);            
            
            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);
            
        end

        %
        function testPartialDifference8x8RandAngPdAng2InSequentialModeMultiple(testCase,nblks)
  
            % Expected values
            pdAng = 14;            
            delta = 1e-10;            
            angs0 = 2*pi*rand(28,nblks);
            angs1 = angs0;
            angs1(pdAng,:) = angs1(pdAng,:)+delta;
            
            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','off');            
            coefExpctd = 1/delta * ...
                ( step(testCase.omgs,angs1,1) ...
                - step(testCase.omgs,angs0,1));
            
            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');
            
            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs0,1,iAng);            
            end
            coefActual = step(testCase.omgs,angs0,1,pdAng);            
            
            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);
            
        end
    
        %
        function testPartialDifference8x8RandMusAngPdAng2InSequentialMode(testCase,nblks)
  
            % Expected values
            mus = 2*round(rand(8,nblks))-1;
            pdAng = 14;            
            delta = 1e-10;            
            angs0 = 2*pi*rand(28,nblks);
            angs1 = angs0;
            angs1(pdAng,:) = angs1(pdAng,:)+delta;
            
            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','off');            
            coefExpctd = 1/delta * ...
                ( step(testCase.omgs,angs1,mus) ...
                - step(testCase.omgs,angs0,mus));
            
            % Instantiation of target class
            import tansacnet.utility.*            
            testCase.omgs = OrthonormalMatrixGenerationSystem(...
                'PartialDifference','sequential');
            
            % Actual values
            for iAng = 0:pdAng-1
                step(testCase.omgs,angs0,mus,iAng);            
            end
            coefActual = step(testCase.omgs,angs0,mus,pdAng);            
            
            % Evaluation
            testCase.verifyEqual(coefActual,coefExpctd,'AbsTol',1e-5);
            
        end
    end
end
"""

if __name__ == '__main__':
    unittest.main()