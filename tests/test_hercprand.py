
from src._hercprand import her_CPRAND


import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

def test_hercprand():
    """
    Test of hercprand for a 3*2*2 kruskal tensor.
    Plot the terminaison criterion.
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
    weights,factors,it,err_ex,error,pct=her_CPRAND(t_krus,rank,n_samples,exact_err=False)
    print("pct restart",pct)
    plt.plot(range(len(err_ex)),err_ex,'b-',label="exact")
    plt.plot(range(len(error)),error,'r--',label="err fast")
    plt.yscale("log") 
    plt.xlabel('it')
    plt.ylabel('terminaison criterion')
    plt.title('her cprand')
    plt.legend(loc='best')
    
    