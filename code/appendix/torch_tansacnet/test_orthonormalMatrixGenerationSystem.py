import itertools
import unittest
from parameterized import parameterized
import torch
import torch.nn as nn
import math
from random import *
from orthonormalMatrixGenerationSystem import OrthonormalMatrixGenerationSystem

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
        angs = [ [math.pi/4], [math.pi/6] ]
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
        omgs(angles=angs,mus=1,index_pd_angle=None)
        coefActual = omgs(angles=angs,mus=1,index_pd_angle=pdAng)
        nextangleActual = omgs.nextangle

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        self.assertEqual(nextangleActual,nextangleExpctd)

    # Test for partial difference with angles
    def testPartialDifferenceWithAnglesAndMuesInSequentialMode(self):
        rtol,atol = 1e-5,1e-8

        # Configuration
        mus = [ [ 1, -1 ] ]        
        pdAng = 0

        # Expected values
        coefExpctd = torch.zeros(1,2,2)
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2) ],
            [ -math.sin(math.pi/4+math.pi/2),  -math.cos(math.pi/4+math.pi/2) ] ])
        nextangleExpctd = 1

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        omgs(angles=math.pi/4,mus=mus,index_pd_angle=None)
        coefActual = omgs(angles=math.pi/4,mus=mus,index_pd_angle=pdAng)
        nextangleActual = omgs.nextangle

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        self.assertEqual(nextangleActual,nextangleExpctd)

    # Test for partial difference with angles
    def testPartialDifferenceWithAnglesAndMuesInSequentialModeMultiple(self):
        rtol,atol = 1e-5,1e-8

        # Configuration
        mus = [ [ 1, -1 ] ]
        angs = [ [math.pi/4], [math.pi/6] ]
        pdAng = 0

        # Expected values
        coefExpctd = torch.zeros(2,2,2)
        coefExpctd[0] = torch.tensor([
            [ math.cos(math.pi/4+math.pi/2), -math.sin(math.pi/4+math.pi/2) ],
            [ -math.sin(math.pi/4+math.pi/2),  -math.cos(math.pi/4+math.pi/2) ] ])
        coefExpctd[1] = torch.tensor([
            [ math.cos(math.pi/6+math.pi/2), -math.sin(math.pi/6+math.pi/2) ],
            [ -math.sin(math.pi/6+math.pi/2),  -math.cos(math.pi/6+math.pi/2) ] ])
        nextangleExpctd = 1

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        omgs(angles=angs,mus=mus,index_pd_angle=None)
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)        
        nextangleActual = omgs.nextangle

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        self.assertEqual(nextangleActual,nextangleExpctd)

    # Test for partial difference with angles
    def testPartialDifference4x4RandAngPdAng2InSequentialModeInitialization(self):
        rtol,atol = 1e-5,1e-6

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
            torch.tensor([
                [ math.cos(angs[0,2]+math.pi/2), 0., 0., -math.sin(angs[0,2]+math.pi/2) ],
                [ 0., 0., 0., 0. ],
                [ 0., 0., 0., 0. ],
                [ math.sin(angs[0,2]+math.pi/2), 0., 0.,  math.cos(angs[0,2]+math.pi/2) ] ]) @ \
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
        nextangleExpctd = pdAng + 1

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        omgs(angles=angs,mus=mus,index_pd_angle=None)
        for iAng in range(pdAng):
            omgs(angles=angs,mus=mus,index_pd_angle=iAng) 
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)            
        nextangleActual = omgs.nextangle

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        self.assertEqual(nextangleActual,nextangleExpctd)

    # Test for partial difference with angles
    @parameterized.expand(itertools.product(nblks))
    def testPartialDifference4x4RandAngPdAng2InSequentialMode(self,nblks):
        rtol,atol = 1e-5,1e-6

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
                torch.tensor([
                    [ math.cos(angs[iblk,2]+math.pi/2), 0., 0., -math.sin(angs[iblk,2]+math.pi/2) ],
                    [ 0., 0., 0., 0. ],
                    [ 0., 0., 0., 0. ],
                    [ math.sin(angs[iblk,2]+math.pi/2), 0., 0.,  math.cos(angs[iblk,2]+math.pi/2) ] ]) @ \
                torch.tensor([
                    [ math.cos(angs[iblk,1]), 0., -math.sin(angs[iblk,1]), 0. ],
                    [ 0., 1., 0., 0. ],
                    [ math.sin(angs[iblk,1]), 0., math.cos(angs[iblk,1]), 0. ],
                    [ 0., 0., 0., 1. ] ]) @ \
                torch.tensor([
                    [ math.cos(angs[iblk,0]), -math.sin(angs[iblk,0]), 0., 0. ],
                    [ math.sin(angs[iblk,0]),  math.cos(angs[iblk,0]), 0., 0. ],
                    [ 0., 0., 1., 0. ],
                    [ 0., 0., 0., 1. ] ])
        nextangleExpctd = pdAng + 1

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        omgs(angles=angs,mus=mus,index_pd_angle=None)
        for iAng in range(pdAng):
            omgs(angles=angs,mus=mus,index_pd_angle=iAng)
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)
        nextangleActual = omgs.nextangle

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        self.assertEqual(nextangleActual,nextangleExpctd)

    # Test for partial difference with angles
    def testPartialDifference4x4RandAngPdAng5InSequentialModeInitialization(self):
        rtol,atol = 1e-5,1e-6

        # Expected values
        mus = [ [1, 1, -1, -1] ]
        angs = 2.*math.pi*torch.rand(1,6)
        pdAng = 5
        coefExpctd = torch.zeros(1,4,4)
        coefExpctd[0] = torch.diag(torch.tensor(mus[0]).to(dtype=angs.dtype)) @ \
            torch.tensor([
                [ 0., 0., 0., 0. ],
                [ 0., 0., 0., 0. ],
                [ 0., 0., math.cos(angs[0,5]+math.pi/2.0), -math.sin(angs[0,5]+math.pi/2.0) ],
                [ 0., 0., math.sin(angs[0,5]+math.pi/2.0),  math.cos(angs[0,5]+math.pi/2.0) ] ]) @ \
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
                [ math.sin(angs[0,2]), 0., 0.,  math.cos(angs[0,2]) ]]) @ \
            torch.tensor([
                [ math.cos(angs[0,1]), 0., -math.sin(angs[0,1]), 0. ],
                [ 0., 1., 0., 0. ],
                [ math.sin(angs[0,1]), 0., math.cos(angs[0,1]), 0. ],
                [ 0., 0., 0., 1. ] ]) @ \
            torch.tensor([
                [ math.cos(angs[0,0]), -math.sin(angs[0,0]), 0., 0. ],
                [ math.sin(angs[0,0]),  math.cos(angs[0,0]), 0., 0. ],
                [ 0., 0., 1., 0. ],
                [ 0., 0., 0., 1. ] ])
        nextangleExpctd = pdAng + 1

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        omgs(angles=angs,mus=mus,index_pd_angle=None)
        for iAng in range(pdAng):
            omgs(angles=angs,mus=mus,index_pd_angle=iAng)
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)
        nextangleActual = omgs.nextangle

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        self.assertEqual(nextangleActual,nextangleExpctd)

    # Test for partial difference with angles
    @parameterized.expand(itertools.product(nblks))
    def testPartialDifference4x4RandAngPdAng5InSequentialMode(self,nblks):
        rtol,atol = 1e-5,1e-6

        # Expected values
        mus = [ [1, 1, -1, -1] ]
        angs = 2.*math.pi*torch.rand(nblks,6)
        pdAng = 5
        coefExpctd = torch.zeros(nblks,4,4)

        for iblk in range(nblks):
            coefExpctd[iblk] = torch.diag(torch.tensor(mus[0]).to(dtype=angs.dtype)) @ \
                torch.tensor([
                    [ 0., 0., 0., 0. ],
                    [ 0., 0., 0., 0. ],
                    [ 0., 0., math.cos(angs[iblk,5]+math.pi/2.0), -math.sin(angs[iblk,5]+math.pi/2.0) ],
                    [ 0., 0., math.sin(angs[iblk,5]+math.pi/2.0),  math.cos(angs[iblk,5]+math.pi/2.0) ] ]) @ \
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
                    [ math.sin(angs[iblk,2]), 0., 0.,  math.cos(angs[iblk,2]) ]]) @ \
                torch.tensor([
                    [ math.cos(angs[iblk,1]), 0., -math.sin(angs[iblk,1]), 0. ],
                    [ 0., 1., 0., 0. ],
                    [ math.sin(angs[iblk,1]), 0., math.cos(angs[iblk,1]), 0. ],
                    [ 0., 0., 0., 1. ] ]) @ \
                torch.tensor([
                    [ math.cos(angs[iblk,0]), -math.sin(angs[iblk,0]), 0., 0. ],
                    [ math.sin(angs[iblk,0]),  math.cos(angs[iblk,0]), 0., 0. ],
                    [ 0., 0., 1., 0. ],
                    [ 0., 0., 0., 1. ] ])
        nextangleExpctd = pdAng + 1

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        omgs(angles=angs,mus=mus,index_pd_angle=None)
        for iAng in range(pdAng):
            omgs(angles=angs,mus=mus,index_pd_angle=iAng)
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)
        nextangleActual = omgs.nextangle

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        self.assertEqual(nextangleActual,nextangleExpctd)

    # Test for partial difference with angles
    def testPartialDifference4x4RandAngPdAng1InSequentialMode(self):
        rtol,atol = 1e-3,1e-3

        # Expected values
        mus = [ [-1, -1, -1, -1] ]
        angs = 2.*math.pi*torch.rand(1,6)
        pdAng = 1
        coefExpctd = torch.zeros(1,4,4)
        delta = 1.e-3
        coefExpctd[0] = 1./delta * \
            torch.diag(torch.tensor(mus[0]).to(dtype=angs.dtype)) @ \
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
                [ math.sin(angs[0,2]), 0., 0.,  math.cos(angs[0,2]) ]]) @ \
              ( torch.tensor([
                [ math.cos(angs[0,1]+delta), 0., -math.sin(angs[0,1]+delta), 0. ],
                [ 0., 1., 0., 0. ],
                [ math.sin(angs[0,1]+delta), 0., math.cos(angs[0,1]+delta), 0. ],
                [ 0., 0., 0., 1. ] ]) - \
                torch.tensor([
                [ math.cos(angs[0,1]), 0., -math.sin(angs[0,1]), 0. ],
                [ 0., 1., 0., 0. ],
                [ math.sin(angs[0,1]), 0., math.cos(angs[0,1]), 0. ],
                [ 0., 0., 0., 1. ] ]) ) @ \
            torch.tensor([
                [ math.cos(angs[0,0]), -math.sin(angs[0,0]), 0., 0. ],
                [ math.sin(angs[0,0]),  math.cos(angs[0,0]), 0., 0. ],
                [ 0., 0., 1., 0. ],
                [ 0., 0., 0., 1. ] ])
        nextangleExpctd = pdAng + 1

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        omgs(angles=angs,mus=mus,index_pd_angle=None)
        for iAng in range(pdAng):
            omgs(angles=angs,mus=mus,index_pd_angle=iAng)
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)
        nextangleActual = omgs.nextangle

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        self.assertEqual(nextangleActual,nextangleExpctd)

    # Test for partial difference with angles
    @parameterized.expand(itertools.product(nblks))
    def testPartialDifference4x4RandAngPdAng1InSequentialModeMultiple(self,nblks):
        rtol,atol = 1e-3,1e-3

        # Expected values
        mus = [ [-1, -1, -1, -1] ]
        angs = 2.*math.pi*torch.rand(nblks,6)
        pdAng = 1
        delta = 1.e-3
        coefExpctd = torch.zeros(nblks,4,4)

        for iblk in range(nblks):
            coefExpctd[iblk] = 1./delta * \
                torch.diag(torch.tensor(mus[0]).to(dtype=angs.dtype)) @ \
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
                torch.tensor([
                    [ math.cos(angs[iblk,2]), 0., 0., -math.sin(angs[iblk,2]) ],
                    [ 0., 1., 0., 0. ],
                    [ 0., 0., 1., 0. ],
                    [ math.sin(angs[iblk,2]), 0., 0.,  math.cos(angs[iblk,2]) ]]) @ \
                  ( torch.tensor([
                    [ math.cos(angs[iblk,1]+delta), 0., -math.sin(angs[iblk,1]+delta), 0. ],
                    [ 0., 1., 0., 0. ],
                    [ math.sin(angs[iblk,1]+delta), 0., math.cos(angs[iblk,1]+delta), 0. ],
                    [ 0., 0., 0., 1. ]])   - \
                    torch.tensor([
                    [ math.cos(angs[iblk,1]), 0., -math.sin(angs[iblk,1]), 0. ],
                    [ 0., 1., 0., 0. ],
                    [ math.sin(angs[iblk,1]), 0., math.cos(angs[iblk,1]), 0. ],
                    [ 0., 0., 0., 1. ] ]) ) @ \
                torch.tensor([
                    [ math.cos(angs[iblk,0]), -math.sin(angs[iblk,0]), 0., 0. ],
                    [ math.sin(angs[iblk,0]),  math.cos(angs[iblk,0]), 0., 0. ],
                    [ 0., 0., 1., 0. ],
                    [ 0., 0., 0., 1. ] ])
        nextangleExpctd = pdAng + 1

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        omgs(angles=angs,mus=mus,index_pd_angle=None)
        for iAng in range(pdAng):
            omgs(angles=angs,mus=mus,index_pd_angle=iAng)
        coefActual = omgs(angles=angs,mus=mus,index_pd_angle=pdAng)
        nextangleActual = omgs.nextangle

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))
        self.assertEqual(nextangleActual,nextangleExpctd)

    # Test for partial difference with angles
    def testPartialDifference8x8RandAngPdAng0InSequentialMode(self):
        rtol,atol = 1e-3,1e-3

        # Expected values
        pdAng = 0
        delta = 1.e-3
        angs0 = 2.*math.pi*torch.rand(1,28)
        angs1 = angs0.clone()
        angs1[0,pdAng] = angs1[0,pdAng] + delta
    
        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=False)
        coefExpctd = 1./delta * ( omgs(angles=angs1,mus=1) - omgs(angles=angs0,mus=1) )

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        omgs(angles=angs0,mus=1,index_pd_angle=None)        
        for iAng in range(pdAng):
            omgs(angles=angs0,mus=1,index_pd_angle=iAng)
        coefActual = omgs(angles=angs0,mus=1,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol)) 

    # Test for partial difference with angles
    @parameterized.expand(itertools.product(nblks))
    def testPartialDifference8x8RandAngPdAng0InSequentialModeMultiple(self,nblks):
        rtol,atol = 1e-3,1e-3

        # Expected values
        pdAng = 0
        delta = 1.e-3
        angs0 = 2.*math.pi*torch.rand(nblks,28)
        angs1 = angs0.clone()
        angs1[:,pdAng] = angs1[:,pdAng] + delta

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=False)
        coefExpctd = 1./delta * ( omgs(angles=angs1,mus=1) - omgs(angles=angs0,mus=1) )

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        omgs(angles=angs0,mus=1,index_pd_angle=None)
        for iAng in range(pdAng):
            omgs(angles=angs0,mus=1,index_pd_angle=iAng)
        coefActual = omgs(angles=angs0,mus=1,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

    # Test for partial difference with angles
    @parameterized.expand(itertools.product(nblks))
    def testPartialDifference8x8RandMusAngPdAng0InSequentialMode(self,nblks):
        rtol,atol = 1e-3,1e-3

        # Expected values
        mus = 2*torch.round(torch.rand(nblks,8))-1
        pdAng = 0
        delta = 1.e-3
        angs0 = 2.*math.pi*torch.rand(nblks,28)
        angs1 = angs0.clone()
        angs1[:,pdAng] = angs1[:,pdAng] + delta

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=False)
        coefExpctd = 1./delta * ( omgs(angles=angs1,mus=mus) - omgs(angles=angs0,mus=mus) )

        # Instantiation of target class
        omgs = OrthonormalMatrixGenerationSystem(partial_difference=True,mode='sequential')

        # Actual values
        omgs(angles=angs0,mus=mus,index_pd_angle=None)
        for iAng in range(pdAng):
            omgs(angles=angs0,mus=mus,index_pd_angle=iAng)
        coefActual = omgs(angles=angs0,mus=mus,index_pd_angle=pdAng)

        # Evaluation
        self.assertTrue(torch.allclose(coefActual,coefExpctd,rtol=rtol,atol=atol))

if __name__ == '__main__':
    unittest.main()