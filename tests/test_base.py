import numpy as np
import tensorly as tl
import src._base as base


def test_random_init_fac() : 
    """
    Test of random_init_fac
    """
    # create a random 3x2x4 tensor on [0,1)
    t_rand = np.random.random((3, 2, 4))
    factors=base.random_init_fac(t_rand, 5)
    for i in factors :
        print(i)
        
def test_svd_init_fac() : 
    """
    Test of svd_init_fac
    """
    # create a random 3x2x4 tensor [0,1)
    t_rand = np.random.random((3, 2, 4))
    factors=base.svd_init_fac(t_rand, 5)
    for i in factors :
        print(i)

def test_err_fac():
    """
    Test of err_fac
    """
    # true factor matrices
    fac=[]
    fac_est=[]
    for i in range(3):
        rnd=np.random.random((3, 3))
        fac+=[rnd]
        fac_est+=[rnd[:,range(2,-1,-1)]]
    print(base.err_fac(fac,fac_est))
    
def test_err():
    """
    Test of err
    """
    # create a kruskal tensor
    # factor matrices
    A=np.arange(9).reshape(3,3)
    B=np.arange(6).reshape(2,3)+9
    C=np.arange(6).reshape(2,3)+15
    factors=[]
    factors+=[A]
    factors+=[B]
    factors+=[C]
    t_krus = tl.cp_to_tensor((None,factors))
    weights_cp,factors_cp=tl.cp_normalize((None,factors))
    print(base.err(t_krus,weights_cp,factors_cp))
    
def test_score():
    fac=[]
    fac_est=[]
    for i in range(4):
        rnd=np.random.random((2, 3))
        fac+=[rnd]
        fac_est+=[rnd[:,range(2,-1,-1)]]
    print(base.score(fac,fac_est))
    







