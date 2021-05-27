import numpy as np
import tensorly as tl
from src._cprand import CPRAND,err_rand
from src._base import init_factors,random_init_fac
import matplotlib.pyplot as plt
import copy


def test_err_rand() :
    """
    Test of err_rand for a kruskal tensor
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
    print(err_rand(t_krus,None,factors,2))
    
    
def test_cprand():
    """
    Test of cprand for a kruskal tensor, start with the true factors
    """
    A=np.arange(9).reshape(3,3)
    B=np.arange(6).reshape(2,3)+9
    C=np.arange(6).reshape(2,3)+15
    factors=[]
    factors+=[A]
    factors+=[B]
    factors+=[C]
    t_krus = tl.cp_to_tensor((None,factors))
    rank=3
    n_samples=int(10*rank*np.log(rank)+1)
    weights,factors,it,err_ex,error=CPRAND(t_krus,rank,n_samples,factors,exact_err=False)
    print(it)
    for i in factors :
        print (i)
        
def test_cprand_random():
    """
    For a noised I*J*K rank r random tensor, with random initialized factor matrices
    plot err_fast and exact err for simple / complicated case
    """
    I=50
    J=50
    K=50
    r=10 # rank
    n_samples=int(10*r*np.log(r)+1) # nb of randomized samples
    fac_true,noise=init_factors(I,J,K,r,True)
    t=tl.cp_to_tensor((None,fac_true))+noise
    print(tl.norm(t))
    factors=random_init_fac(t,r)
    weights2,factors2,it2,error2,error_es2=CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=500,err_it_max=400)
    plt.figure(0)
    plt.plot(range(len(error2)),error2,'b-',label="exact")
    plt.plot(range(len(error_es2)),error_es2,'r--',label="err fast")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('cprand for complicated case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    plt.figure(1)
    fac_true,noise=init_factors(I,J,K,r,False)
    t=tl.cp_to_tensor((None,fac_true))+noise
    print(tl.norm(t))
    factors=random_init_fac(t,r)
    weights2,factors2,it2,error2,error_es2=CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=500,err_it_max=400)
    plt.plot(range(len(error2)),error2,'b-',label="exact")
    plt.plot(range(len(error_es2)),error_es2,'r--',label="err fast")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('cprand for simple case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    
