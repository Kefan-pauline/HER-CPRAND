
from src._hercprand import her_CPRAND_old,her_CPRAND1,her_CPRAND2,her_CPRAND,her_CPRAND4,her_CPRAND5


import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt
import copy
from src._base import init_factors,random_init_fac
from src._herals import her_Als


def test_hercprand_old():
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
    weights,factors,it,err_ex,error,pct=her_CPRAND_old(t_krus,rank,n_samples,exact_err=False)
    print("pct restart",pct)
    plt.plot(range(len(err_ex)),err_ex,'b-',label="exact")
    plt.plot(range(len(error)),error,'r--',label="err fast")
    plt.yscale("log") 
    plt.xlabel('it')
    plt.ylabel('terminaison criterion')
    plt.title('her cprand')
    plt.legend(loc='best')
    
def test_hercprand():
    """
    Run herCPRAND1 2 3 4 5 for the simple and complicated case, plot exact/estimated error.
    Print restart percentage.
    Compare running time with herCPRAND for complicated case.
    """
    I=50
    J=50
    K=50
    r=10 # rank
    n_samples=int(10*r*np.log(r)+1) # nb of randomized samples
    fac_true,noise=init_factors(I,J,K,r,True)
    t=tl.cp_to_tensor((None,fac_true))+noise
    factors=random_init_fac(t,r)
    weights1,factors1,it1,error1,error_es1,cpt1,time1=her_CPRAND1(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=200,err_it_max=100,time_rec=True)
    print("1 Complicated case pct restart",cpt1)
    print("1 Complicated case time err_rand",np.cumsum(time1)[len(time1)-1])
    weights2,factors2,it2,error2,error_es2,cpt2,time2=her_CPRAND2(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=200,err_it_max=100,time_rec=True)
    print("2 Complicated case pct restart",cpt2)
    print("2 Complicated case time err_rand",np.cumsum(time2)[len(time2)-1])
    weights3,factors3,it3,error3,error_es3,cpt3,time3=her_CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=200,err_it_max=100,time_rec=True)
    print("3 Complicated case pct restart",cpt3)
    print("3 Complicated case time err_rand",np.cumsum(time3)[len(time3)-1])
    weights4,factors4,it4,error4,error_es4,cpt4,time4=her_CPRAND4(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=200,err_it_max=100,time_rec=True)
    print("4 Complicated case pct restart",cpt4)
    print("4 Complicated case time err_rand",np.cumsum(time4)[len(time4)-1])
    weights5,factors5,it5,error5,error_es5,cpt5,time5=her_CPRAND5(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=200,err_it_max=100,time_rec=True)
    print("5 Complicated case pct restart",cpt5)
    print("5 Complicated case time err_rand",np.cumsum(time5)[len(time5)-1])
    
    
    
    
    plt.figure(0)
    plt.plot(range(len(error1)),error1,'b-',label="exact")
    plt.plot(range(len(error_es1)),error_es1,'r--',label="err rand")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand1 for complicated case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    plt.figure(1)
    plt.plot(range(len(error2)),error2,'b-',label="exact")
    plt.plot(range(len(error_es2)),error_es2,'r--',label="err rand")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand2 for complicated case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    plt.figure(2)
    plt.plot(range(len(error3)),error3,'b-',label="exact")
    plt.plot(range(len(error_es3)),error_es3,'r--',label="err rand")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand3 for complicated case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    plt.figure(3)
    plt.plot(range(len(error4)),error4,'b-',label="exact")
    plt.plot(range(len(error_es4)),error_es4,'r--',label="err rand")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand4 for complicated case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    plt.figure(4)
    plt.plot(range(len(error5)),error5,'b-',label="exact")
    plt.plot(range(len(error_es5)),error_es5,'r--',label="err rand")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand5 for complicated case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    

    fac_true,noise=init_factors(I,J,K,r,False)
    t=tl.cp_to_tensor((None,fac_true))+noise
    factors=random_init_fac(t,r)
    weights1,factors1,it1,error1,error_es1,cpt1,time1=her_CPRAND1(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=200,err_it_max=100,time_rec=True)
    print("Simple case pct restart",cpt1)
    weights2,factors2,it2,error2,error_es2,cpt2,time2=her_CPRAND2(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=200,err_it_max=100,time_rec=True)
    print("Simple case pct restart",cpt2)
    weights3,factors3,it3,error3,error_es3,cpt3,time3=her_CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=200,err_it_max=100,time_rec=True)
    print("Simple case pct restart",cpt3)
    weights4,factors4,it4,error4,error_es4,cpt4,time4=her_CPRAND4(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=200,err_it_max=100,time_rec=True)
    print("Simple case pct restart",cpt4)
    weights5,factors5,it5,error5,error_es5,cpt5,time5=her_CPRAND5(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=200,err_it_max=100,time_rec=True)
    print("Simple case pct restart",cpt5)
    plt.figure(5)
    plt.plot(range(len(error1)),error1,'b-',label="exact")
    plt.plot(range(len(error_es1)),error_es1,'r--',label="err rand")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand1 for simple case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    plt.figure(6)
    plt.plot(range(len(error2)),error2,'b-',label="exact")
    plt.plot(range(len(error_es2)),error_es2,'r--',label="err rand")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand2 for simple case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    plt.figure(7)
    plt.plot(range(len(error3)),error3,'b-',label="exact")
    plt.plot(range(len(error_es3)),error_es3,'r--',label="err rand")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand3 for simple case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    plt.figure(8)
    plt.plot(range(len(error4)),error4,'b-',label="exact")
    plt.plot(range(len(error_es4)),error_es4,'r--',label="err rand")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand4 for simple case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    plt.figure(9)
    plt.plot(range(len(error5)),error5,'b-',label="exact")
    plt.plot(range(len(error_es5)),error_es5,'r--',label="err rand")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand5 for simple case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    
def test_hercprand_35():
    """
    test hercprand 3 5 for complicated case
    """
    I=50
    J=50
    K=50
    r=10 # rank
    n_samples=int(10*r*np.log(r)+1) # nb of randomized samples
    fac_true,noise=init_factors(I,J,K,r,True)
    t=tl.cp_to_tensor((None,fac_true))+noise
    factors=random_init_fac(t,r)
    
    weights1,factors1,it1,error1,cpt1,time1=her_Als(t,r,factors=copy.deepcopy(factors),it_max=200,time_rec=True)
    print("her als Complicated case pct restart",cpt1)
    
    weights3,factors3,it3,error3,error_es3,cpt3,time3=her_CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=200,err_it_max=100,time_rec=True)
    print("3 Complicated case pct restart",cpt3)
    print("3 Complicated case time err_rand",np.cumsum(time3)[len(time3)-1])
    print("3 min error", np.min(error3))
    print("3 min error es", np.min(error_es3))
  
    weights5,factors5,it5,error5,error_es5,cpt5,time5=her_CPRAND5(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=200,err_it_max=100,time_rec=True)
    print("5 Complicated case pct restart",cpt5)
    print("5 Complicated case time err_rand",np.cumsum(time5)[len(time5)-1])
    print("5 min error", np.min(error5))
    print("5 min error es", np.min(error_es5))
    
    
    plt.figure(0)
    plt.plot(range(len(error3)),error3,'b-',label="exact 3")
    plt.plot(range(len(error_es3)),error_es3,'r--',label="err rand 3")
    plt.plot(range(len(error5)),error5,'g-',label="exact 5")
    plt.plot(range(len(error_es5)),error_es5,'k--',label="err rand 5")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand3 5 for complicated case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')

