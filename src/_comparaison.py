"""
compare error of the 4 algorithms along with it / time

"""
import numpy as np
import tensorly as tl
from src._base import random_init_fac,err_fac, init_factors, score,init_factors_hernn
from src._als import als,nn_als
from src._herals import her_Als,nn_her_Als
from src._cprand import CPRAND,nn_CPRAND
from src._hercprand import her_CPRAND,nn_her_CPRAND
import copy
import matplotlib.pyplot as plt

  
def comparison(I,J,K,r,nb_rand,n_samples,n_samples_err,exact_err=False,scale=False,noise_level=0.3,tol=0.31):
  """
    boxplot for fits, scores, it, time, restarts.
    We generate nb_rand noised I*J*K rank r random tensors, for each tensor, we have 5 factors initializations.
    Then we run the 4 algorithms.

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
        sample size used for (her)CPRAND.
    n_samples_err : int
        sample size used for error in (her)cprand
    exact_err : boolean, optional
        whether use exact error computation or not for (her)CPRAND. The default is False.
    scale : boolean, optional
        whether to scale the singular values of matrices or not. The default is False.
    
    Returns
    -------
    None.

  """

  fit={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  score_tot={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  it_tot={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  time_tot={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  restart={"herals" :  [],"hercprand" :  [],"hercprand small" :  []}
  # local variables
  error={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  l_fac={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  it={1 :  0, 2 :  0, 3 :  0, 4 :  0, 5 :  0, 6 :  0}
  time={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  for i in range(nb_rand) : 
    # Random initialization of a noised cp_tensor
    fac_true,noise=init_factors(I,J,K,r,noise_level,scale)
    t=tl.cp_to_tensor((None,fac_true))+noise
    for k in range(5):
      # random initialization of factors
      factors=random_init_fac(t,r)
      # run 4 methods
      weights1,l_fac[1],it[1],error[1],cpt1,l_fac1,time[1]=her_Als(t,r,factors=copy.deepcopy(factors),it_max=200,tol=tol,list_factors=True,time_rec=True)  
      weights2,l_fac[2],it[2],error[2],l_fac2,time[2]=als(t,r,factors=copy.deepcopy(factors),it_max=200,tol=tol,list_factors=True,time_rec=True) 
      weights3,l_fac[3],it[3],error[3],l_fac3,time[3]=CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True)
      weights4,l_fac[4],it[4],error[4],cpt4,l_fac4,time[4]=her_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True) 
      weights5,l_fac[5],it[5],error[5],l_fac5,time[5]=CPRAND(t,r,100,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True)
      weights6,l_fac[6],it[6],error[6],cpt6,l_fac6,time[6]=her_CPRAND(t,r,100,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True) 
      # information storage
      restart["herals"].append(cpt1)
      restart["hercprand"].append(cpt4)
      restart["hercprand small"].append(cpt6)
      for j in range(1,7):
        fit[j].append(1-(error[j][len(error[j])-1]))
        score_tot[j].append(score(fac_true,l_fac[j]))
        it_tot[j].append(it[j])
        time_tot[j].append(np.cumsum(time[j])[len(time[j])-1])
  # figure
  labels=["herals","als","cprand","hercprand","cprand small","hercprand small"]
  _, dataf = [*zip(*fit.items())]
  _, datas = [*zip(*score_tot.items())]
  _, datai = [*zip(*it_tot.items())]
  _, datat = [*zip(*time_tot.items())]
  _, datar = [*zip(*restart.items())]
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
  plt.figure(4)
  plt.boxplot(datar,vert=False)
  plt.yticks(range(1, 4), ["herals","hercprand","hercprand small"])
  plt.title('restarts')
  
  
def compar_filter(I,J,K,r,nb_rand,n_samples,n_samples_err,exact_err=False,scale=False,noise_level=0.1,tol=0.11):
  """
    boxplot for fits, scores, it, time, restarts in order to compare the filter used in HER-CPRAND.
    We generate nb_rand noised I*J*K rank r random tensors, for each tensor, we have 5 factors initializations.
    Then we run the 4 algorithms.

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
        sample size used for (her)CPRAND.
    n_samples_err : int
        sample size used for error in (her)cprand
    exact_err : boolean, optional
        whether use exact error computation or not for (her)CPRAND. The default is False.
    scale : boolean, optional
        whether to scale the singular values of matrices or not. The default is False.
    
    Returns
    -------
    None.

  """

  fit={1 :  [],2 :  [],3 :  [],4 :  []}
  score_tot={1 :  [],2 :  [],3 :  [],4 :  []}
  it_tot={1 :  [],2 :  [],3 :  [],4 :  []}
  time_tot={1 :  [],2 :  [],3 :  [],4 :  []}
  restart={1 :  [],2 :  [],3 :  [],4 :  []}
  # local variables
  error={1 :  [],2 :  [],3 :  [],4 :  []}
  l_fac={1 :  [],2 :  [],3 :  [],4 :  []}
  it={1 :  0, 2 :  0, 3 :  0, 4 :  0}
  time={1 :  [],2 :  [],3 :  [],4 :  []}
  for i in range(nb_rand) : 
    # Random initialization of a noised cp_tensor
    fac_true,noise=init_factors(I,J,K,r,noise_level,scale)
    t=tl.cp_to_tensor((None,fac_true))+noise
    for k in range(5):
      # random initialization of factors
      factors=random_init_fac(t,r)
      # run 4 methods
      weights1,l_fac[1],it[1],error[1],cpt1,l_fac1,time[1]=her_Als(t,r,factors=copy.deepcopy(factors),it_max=200,tol=tol,list_factors=True,time_rec=True)  
      weights4,l_fac[2],it[2],error[2],err_es4,cpt4,l_fac4,time[2]=her_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True) 
      weights6,l_fac[3],it[3],error[3],err_es6,cpt6,l_fac6,time[3]=her_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True,filter=5) 
      # information storage
      restart[1].append(cpt1)
      restart[2].append(cpt4)
      restart[3].append(cpt6)
      for j in range(1,4):
        fit[j].append(1-(error[j][len(error[j])-1]))
        score_tot[j].append(score(fac_true,l_fac[j]))
        it_tot[j].append(it[j])
        time_tot[j].append(np.cumsum(time[j])[len(time[j])-1])
  # figure
  labels=["herals","her cprand 10","hercprand 5"]
  _, dataf = [*zip(*fit.items())]
  _, datas = [*zip(*score_tot.items())]
  _, datai = [*zip(*it_tot.items())]
  _, datat = [*zip(*time_tot.items())]
  _, datar = [*zip(*restart.items())]
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
  plt.figure(4)
  plt.boxplot(datar,vert=False)
  plt.yticks(range(1, 4), labels)
  plt.title('restarts')
      
def nn_comparison(I,J,K,r,nb_rand,n_samples,n_samples_err,exact_err=False,scale=False,noise_level=0.1,tol=0.10):
  """
    boxplot for fits, scores, it, time, restarts.
    We generate nb_rand noised I*J*K rank r random tensors, for each tensor, we have 5 factors initializations.
    Then we run the 4 nn algorithms.

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
        sample size used for (her)CPRAND.
    n_samples_err : int
        sample size used for error in (her)cprand
    exact_err : boolean, optional
        whether use exact error computation or not for (her)CPRAND. The default is False.
    scale : boolean, optional
        whether to scale the singular values of matrices or not. The default is False.
    
    Returns
    -------
    None.

  """

  fit={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  score_tot={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  it_tot={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  time_tot={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  restart={"herals" :  [],"hercprand" :  [],"hercprand small" :  []}
  # local variables
  error={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  l_fac={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  it={1 :  0, 2 :  0, 3 :  0, 4 :  0, 5 :  0, 6 :  0}
  time={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  for i in range(nb_rand) : 
    # Random initialization of a noised cp_tensor
    fac_true,noise=init_factors(I,J,K,r,noise_level,scale,nn=True)
    t=tl.cp_to_tensor((None,fac_true))+noise
    for k in range(5):
      # random initialization of factors
      factors=random_init_fac(t,r)
      # run 4 methods
      weights1,l_fac[1],it[1],error[1],cpt1,l_fac1,time[1]=nn_her_Als(t,r,factors=copy.deepcopy(factors),it_max=200,tol=tol,list_factors=True,time_rec=True)  
      weights2,l_fac[2],it[2],error[2],l_fac2,time[2]=nn_als(t,r,factors=copy.deepcopy(factors),it_max=200,tol=tol,list_factors=True,time_rec=True) 
      weights3,l_fac[3],it[3],error[3],l_fac3,time[3]=nn_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True)
      weights4,l_fac[4],it[4],error[4],cpt4,l_fac4,time[4]=nn_her_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True) 
      weights5,l_fac[5],it[5],error[5],l_fac5,time[5]=nn_CPRAND(t,r,100,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True)
      weights6,l_fac[6],it[6],error[6],cpt6,l_fac6,time[6]=nn_her_CPRAND(t,r,100,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True) 
      # information storage
      restart["herals"].append(cpt1)
      restart["hercprand"].append(cpt4)
      restart["hercprand small"].append(cpt6)
      for j in range(1,7):
        fit[j].append(1-(error[j][len(error[j])-1]))
        score_tot[j].append(score(fac_true,l_fac[j]))
        it_tot[j].append(it[j])
        time_tot[j].append(np.cumsum(time[j])[len(time[j])-1])
  # figure
  labels=["herals","als","cprand","hercprand","cprand small","hercprand small"]
  _, dataf = [*zip(*fit.items())]
  _, datas = [*zip(*score_tot.items())]
  _, datai = [*zip(*it_tot.items())]
  _, datat = [*zip(*time_tot.items())]
  _, datar = [*zip(*restart.items())]
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
  plt.figure(4)
  plt.boxplot(datar,vert=False)
  plt.yticks(range(1, 4), ["herals","hercprand","hercprand small"])
  plt.title('restarts')
  
