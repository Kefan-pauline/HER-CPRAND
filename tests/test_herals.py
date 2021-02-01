
import numpy as np
import tensorly as tl
from src._herals import her_Als


def test_herals():
    """
    Test of herals for a kruskal tensor
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


    weights,fac,it,error,cpt=her_Als(t_krus,3)
    for i in fac :
        print(i)
    print(it)
    print(error[len(error)-1])

    
