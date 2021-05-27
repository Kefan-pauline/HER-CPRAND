
import numpy as np
import tensorly as tl
from src._herals import her_Als
from src._base import random_init_fac,init_factors,score


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
    
    
def test_herals_score():
    """
    Test of score for a 3 order simple tensor

    """
    # create a kruskal tensor
    # factor matrices
    I=3
    J=3
    K=3
    r=3
    factors,noise=init_factors(I, J, K, r)

    t_krus = tl.cp_to_tensor((None,factors))
    factors_init=random_init_fac(t_krus, r)
    
    weights,factors1,it,error1,l,pct=her_Als(t_krus,r,factors=factors_init,it_max=500,list_factors=True)
    print(score(factors,factors1))
    weights1,factors1n=tl.cp_normalize((weights,factors1))
    weight,factorsn=tl.cp_normalize((None,factors))
    for i in factors1n :
        print(i)
    for i in factorsn :
        print(i)
    print(it)
    print(error1[len(error1)-1])


