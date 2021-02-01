"""
function to find the optimal parameters for hercprand
"""
import numpy as np
import tensorly as tl
from src._hercprand import her_CPRAND
from src._base import random_init_fac
import matplotlib.pyplot as plt
import copy

def param_research(I,J,K,r,nb_rand,n_samples,exact_err=True,beta=True,eta=False,gamma=False):
  """
    plot data fitting error for different values of beta, eta and gamma.
    For each parameter (beta for example), we initialize nb_rand noised I*J*K rank r 
    random tensors.
    For each tensor, we have 5 random factors initializations.
    Need to change plot label to run the test.

    Parameters
    ----------
    I : int
        dimension of mode 1.
    J : int
        dimension of mode 2.
    K : int
        dimension of mode 3.
    r : int
        rank.
    nb_rand : int
        nb of tensors.
    n_samples : int
        sample size used for herCPRAND.
    exact_err : boolean, optional
        whether use exact error computation or not for herCPRAND. The default is True.
    beta : boolean, optional
        plot figure for different values of beta. The default is True.
    eta : boolean, optional
        plot figure for different values of eta. The default is False.
    gamma : boolean, optional
        plot figure for different values of gamma. The default is False.

    Returns
    -------
    None.

  """
  list_err1=[]
  list_err2=[]
  list_err3=[]
  min_e=None
  for i in range(nb_rand) : 
    # Random initialization of a noised cp_tensor
    A=np.random.normal(0, 1, size=(I, r))
    B=np.random.normal(0, 1, size=(J, r))
    C=np.random.normal(0, 1, size=(K, r))
    noise=np.random.normal(0, 0.001, size=(I,J,K))
    tensor=tl.cp_to_tensor((None,[A,B,C]))+noise
    norm_tensor=tl.norm(tensor,2)
    if(min_e==None) : min_e=norm_tensor
    for j in range(5) :
        factors=random_init_fac(tensor,r)
        # parameter choice
        if(beta==True):
          weights1,factors1,it1,_,error1,cpt1=her_CPRAND(tensor,r,n_samples,factors=copy.deepcopy(factors),it_max=500,exact_err=exact_err,err_it_max=200,beta=0.1) # beta0=0.1
          weights2,factors2,it2,_,error2,cpt2=her_CPRAND(tensor,r,n_samples,factors=copy.deepcopy(factors),it_max=500,exact_err=exact_err,err_it_max=200,beta=0.3) # beta0=0.3
          weights3,factors3,it3,_,error3,cpt3=her_CPRAND(tensor,r,n_samples,factors=copy.deepcopy(factors),it_max=500,exact_err=exact_err,err_it_max=200,beta=0.5) # beta0=0.5
        if(eta==True):
          weights1,factors1,it1,_,error1,cpt1=her_CPRAND(tensor,r,n_samples,factors=copy.deepcopy(factors),it_max=500,exact_err=exact_err,err_it_max=200,eta=1.1) # eta=1.1
          weights2,factors2,it2,_,error2,cpt2=her_CPRAND(tensor,r,n_samples,factors=copy.deepcopy(factors),it_max=500,exact_err=exact_err,err_it_max=200,eta=2) # eta=2
          weights3,factors3,it3,_,error3,cpt3=her_CPRAND(tensor,r,n_samples,factors=copy.deepcopy(factors),it_max=500,exact_err=exact_err,err_it_max=200,eta=3) # eta=3
        if(gamma==True):
          weights1,factors1,it1,_,error1,cpt1=her_CPRAND(tensor,r,n_samples,factors=copy.deepcopy(factors),it_max=200,exact_err=exact_err,err_it_max=100,gamma=1.01,gamma_bar=1.005) 
          weights2,factors2,it2,_,error2,cpt2=her_CPRAND(tensor,r,n_samples,factors=copy.deepcopy(factors),it_max=200,exact_err=exact_err,err_it_max=100,gamma=1.05,gamma_bar=1.01) 
          weights3,factors3,it3,_,error3,cpt3=her_CPRAND(tensor,r,n_samples,factors=copy.deepcopy(factors),it_max=200,exact_err=exact_err,err_it_max=100,gamma=1.9,gamma_bar=1.5) 
        error1=[i * norm_tensor for i in error1]
        list_err1.append(error1)
        error2=[i * norm_tensor for i in error2]
        list_err2.append(error2)
        error3=[i * norm_tensor for i in error3]
        list_err3.append(error3)
        if (min_e>min(min(error1),min(error2),min(error3))) : min_e=min(min(error1),min(error2),min(error3))
  list_err1=[x-min_e for x in list_err1]
  list_err2=[x-min_e for x in list_err2]
  list_err3=[x-min_e for x in list_err3]
  # plot
  for i in range(len(list_err1)):
    if i==0 : plt.plot(range(len(list_err1[i])),list_err1[i],'b-',label='eta=1.1') 
    else: plt.plot(range(len(list_err1[i])),list_err1[i],'b-') 
    
  for i in range(len(list_err2)):
    if i==0: plt.plot(range(len(list_err2[i])),list_err2[i],'r-',label='eta=2') 
    else : plt.plot(range(len(list_err2[i])),list_err2[i],'r-') 
    
  for i in range(len(list_err3)):
    if i==0 : plt.plot(range(len(list_err3[i])),list_err3[i],'g-',label='eta=3')
    else : plt.plot(range(len(list_err3[i])),list_err3[i],'g-')
  plt.yscale("log") 
  plt.legend(loc='best')
  plt.xlabel('it')
  plt.ylabel('f')
  plt.title('f(iteration)')