import numpy as np
import tensorly as tl
from src._als import als
from src._base import err,init_factors
import matplotlib.pyplot as plt

def test_err_fast():
    """
    Test err_fast for 3 tensors.
    plot the terminaison criterion (obtained by err_fast) and exact error.
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
    weights,factors,it,error,l=als(t_krus,3,list_factors=True)
    err_ex=[]
    for i in l :
        err_ex+=[err(t_krus,weights,i)]
    plt.figure(0)
    plt.plot(range(len(err_ex)),err_ex/tl.norm(t_krus),'b-',label="exact")
    plt.plot(range(len(error)),error,'r--',label="err fast")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('als for t_krus')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    
    # create an complicated random tensor
    I=50
    J=50
    K=50
    r=10 # rank
    A,B,C,noise=init_factors(I,J,K,r,True)
    fac_true=[A,B,C]
    t=tl.cp_to_tensor((None,fac_true))+noise
    weights,factors,it,error,l=als(t,r,list_factors=True)
    err_ex=[]
    for i in l :
        err_ex+=[err(t,weights,i)]
    plt.figure(1)
    plt.plot(range(len(err_ex)),err_ex/tl.norm(t),'b-',label="exact")
    plt.plot(range(len(error)),error,'r--',label="err fast")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('als for complicated case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')

    # create a simple random tensor
    A,B,C,noise=init_factors(I,J,K,r,False)
    fac_true=[A,B,C]
    t=tl.cp_to_tensor((None,fac_true))+noise
    weights,factors,it,error,l=als(t,r,list_factors=True)
    err_ex=[]
    for i in l :
        err_ex+=[err(t,weights,i)]
    plt.figure(2)
    plt.plot(range(len(err_ex)),err_ex/tl.norm(t),'b-',label="exact")
    plt.plot(range(len(error)),error,'r--',label="err fast")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('als for simple case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')

    
        
def test_als():
    """
    Test of als for a kruskal tensor, start with true factors

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
    #weights_cp,factors_cp=tl.cp_normalize((None,factors))
    
    weights,factors,it,error1,l=als(t_krus,3,factors=factors,list_factors=True)
    
    for i in factors :
        print(i)
    print(it)