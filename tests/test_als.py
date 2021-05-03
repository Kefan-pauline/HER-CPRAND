import numpy as np
import tensorly as tl
from src._als import als,nn_als
from src._base import err,init_factors,random_init_fac,score,init_factors_hernn
import matplotlib.pyplot as plt
import copy


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
    fac_true,noise=init_factors(I,J,K,r,True)
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
    fac_true,noise=init_factors(I,J,K,r,False)
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

    
    weights,factors1,it,error1,l=als(t_krus,3,factors=factors,list_factors=True)
    
    for i in factors1 :
        print(i)
    print(it)
    
def test_nnals():
    """
    Test of nn_als for a kruskal tensor, start with random factors

    """
    # create a kruskal tensor
    # factor matrices
    fac_true,noise=init_factors(200,200,200,10,scale=True,nn=True)

    t_krus = tl.cp_to_tensor((None,fac_true))+noise
    #weights_cp,factors_cp=tl.cp_normalize((None,factors))
    factors=random_init_fac(t_krus, 10)
    
    weights,factors1,it,error1,l=nn_als(t_krus,10,factors=factors,tol=0.1,list_factors=True)
    print(it)
    for i in error1:
        print(i)
    
def compar_nn_als(I=200,J=200,K=200,r=10,nb_rand=10,exact_err=False,scale=True,noise_level=0.1,tol=0.0001):
  fit={1 :  [],2 :  []}
  score_tot={1 :  [],2 :  []}
  it_tot={1 :  [],2 :  []}
  time_tot={1 :  [],2 :  []}
  # local variables
  error={1 :  [],2 :  []}
  l_fac={1 :  [],2 :  []}
  it={1 :  0, 2 :  0}
  time={1 :  [],2 :  []}
  for i in range(nb_rand) : 
    # Random initialization of a noised cp_tensor
    fac_true,noise=init_factors_hernn(I,J,K,r,noise_level,scale)
    t=tl.cp_to_tensor((None,fac_true))+noise
    #norm_tensor=tl.norm(t,2)
    for k in range(5):
      # random initialization of factors
      factors=random_init_fac(t,r)
      # run 4 methods
      weights1,l_fac[1],it[1],error[1],l_fac1,time[1]=als(t,r,factors=copy.deepcopy(factors),it_max=200,tol=tol,list_factors=True,time_rec=True)  
      weights2,l_fac[2],it[2],error[2],l_fac2,time[2]=nn_als(t,r,factors=copy.deepcopy(factors),it_max=200,tol=tol,list_factors=True,time_rec=True) 
      
      for j in range(1,3):
        fit[j].append(1-(error[j][len(error[j])-1]))
        score_tot[j].append(score(fac_true,l_fac[j]))
        it_tot[j].append(it[j])
        time_tot[j].append(np.cumsum(time[j])[len(time[j])-1])
  # figure
  labels=["als","nn als"]
  _, dataf = [*zip(*fit.items())]
  _, datas = [*zip(*score_tot.items())]
  _, datai = [*zip(*it_tot.items())]
  _, datat = [*zip(*time_tot.items())]
  plt.figure(0)
  plt.boxplot(dataf,vert=False)
  plt.yticks(range(1, len(labels) + 1), labels)
  plt.title('fits')
  plt.figure(1)
  plt.boxplot(datas,vert=False)
  plt.yticks(range(1, len(labels) + 1), labels)
  plt.title('scores')  
  plt.figure(2)
  plt.boxplot(datai,vert=False)
  plt.yticks(range(1, len(labels) + 1), labels)
  plt.title('it')
  plt.figure(3)
  plt.boxplot(datat,vert=False)
  plt.yticks(range(1, len(labels) + 1), labels)
  plt.title('time')


