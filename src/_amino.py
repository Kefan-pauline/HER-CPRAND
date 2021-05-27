

import scipy.io as sio
import numpy as np
import tensorly as tl
from src._base import random_init_fac
from src._als import nn_als
from src._herals import nn_her_Als
from src._cprand import nn_CPRAND
from src._hercprand import nn_her_CPRAND
import copy
import matplotlib.pyplot as plt



def test_amino(r,n_samples,n_samples_err,exact_err=False,tol=0.01):
  """
    boxplot for fits, scores, it, time, restarts.
    We generate nb_rand noised I*J*K rank r random tensors, for each tensor, we have 5 factors initializations.
    Then we run the (HER-)CPRAND nn algorithms.

    Parameters
    ----------

    r : int
        rank.
    n_samples : int
        sample size used for (her)CPRAND.
    n_samples_err : int
        sample size used for error in (her)cprand
    exact_err : boolean, optional
        whether use exact error computation or not for (her)CPRAND. The default is False.

    Returns
    -------
    None.

  """
  path='/Users/sunke/OneDrive/PIR/S8/PIRS8/data/'
  file='amino.mat'
  mat = sio.loadmat(path+file)
  X=mat['X']
  X=np.reshape(X,(5,61,201))
  t=tl.tensor(X)

  fit={3 :  [],4 :  [],5 :  [],6 :  []}
  score_tot={3 :  [],4 :  [],5 :  [],6 :  []}
  it_tot={3 :  [],4 :  [],5 :  [],6 :  []}
  time_tot={3 :  [],4 :  [],5 :  [],6 :  []}
  restart={"hercprand" :  [],"hercprand small" :  []}
  # local variables
  error={3 :  [],4 :  [],5 :  [],6 :  []}
  l_fac={3 :  [],4 :  [],5 :  [],6 :  []}
  it={ 3 :  0, 4 :  0, 5 :  0, 6 :  0}
  time={3 :  [],4 :  [],5 :  [],6 :  []}


  for k in range(5):
  # random initialization of factors
      factors=random_init_fac(t,r)
      # run 4 methods
      weights3,l_fac[3],it[3],error[3],l_fac3,time[3]=nn_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True)
      weights4,l_fac[4],it[4],error[4],cpt4,l_fac4,time[4]=nn_her_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True) 
      weights5,l_fac[5],it[5],error[5],l_fac5,time[5]=nn_CPRAND(t,r,10,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True)
      weights6,l_fac[6],it[6],error[6],cpt6,l_fac6,time[6]=nn_her_CPRAND(t,r,10,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200,tol=tol,list_factors=True,time_rec=True) 
      
      # information storage
      restart["hercprand"].append(cpt4)
      restart["hercprand small"].append(cpt6)
      for j in range(3,7):
        fit[j].append(1-(error[j][len(error[j])-1]))
        it_tot[j].append(it[j])
        time_tot[j].append(np.cumsum(time[j])[len(time[j])-1])
  # figure
  labels=["cprand","hercprand","cprand small","hercprand small"]
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
  plt.yticks(range(1,3), ["hercprand","hercprand small"])
  plt.title('restarts')
  
  
  
def amino_plot(r,n_samples,n_samples_err,exact_err=False,tol=0.0001):
  """
    plot factors of amino data, obtained with HER-CPRAND and ALS.

    Parameters
    ----------
    r : int
        the rank
    n_samples : int
        sample size for HER-CPRAND
    n_samples_err : int
        sample size for fast error estimation in HER-CPRAND
    exact_err : Bool, optional
        evaluate the error exactly or not. The default is False.
    tol : float, optional
        stopping criterion for HER-CPRAND and for ALS. The default is 0.0001.

    Returns
    -------
    None.

  """
  path='/Users/sunke/OneDrive/PIR/S8/PIRS8/data/'
  file='amino.mat'
  mat = sio.loadmat(path+file)
  X=mat['X']
  X=np.reshape(X,(5,61,201))
  t=tl.tensor(X)
  factors=random_init_fac(t,r)
  weights4,factors4,it4,error4,cpt=nn_her_CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=100,err_it_max=100,tol=tol) 
  weights1,factors1,it1,error1=nn_als(t,r,factors=copy.deepcopy(factors),it_max=100,tol=tol)
  print("HERCPRAND",it4)
  print("ALS",it1)
  # noramalize 
  weights4,factors4=tl.cp_normalize((weights4,factors4))
  weights1,factors1=tl.cp_normalize((weights1,factors1))
  plt.figure(0)
  i=factors4[0]
  plt.plot(abs(i[:,0]),label="Substance 1")
  plt.plot(abs(i[:,1]),label="Substance 2")
  plt.plot(abs(i[:,2]),label="Substance 3")
  plt.legend(loc='best')
  plt.title("Composition of each sample her CPRAND")
  plt.xlabel('Sample')
  plt.ylabel('Quantity')
  
  plt.figure(1)
  i=factors4[1]
  plt.plot(abs(i[:,0]),label="Substance 1")
  plt.plot(abs(i[:,1]),label="Substance 2")
  plt.plot(abs(i[:,2]),label="Substance 3")
  plt.legend(loc='best')
  plt.title("Excitation loadings her CPRAND")
  plt.xlabel('Excitation wavelength')
  plt.ylabel('Loading')
  
  plt.figure(2)
  i=factors4[2]
  plt.plot(abs(i[:,0]),label="Substance 1")
  plt.plot(abs(i[:,1]),label="Substance 2")
  plt.plot(abs(i[:,2]),label="Substance 3")
  plt.legend(loc='best')
  plt.title("Emission loadings her CPRAND")
  plt.xlabel('Emission wavelength')
  plt.ylabel('Loading')
  
  plt.figure(3)
  i=factors1[0]
  plt.plot(abs(i[:,0]),label="Substance 1")
  plt.plot(abs(i[:,1]),label="Substance 2")
  plt.plot(abs(i[:,2]),label="Substance 3")
  plt.legend(loc='best')
  plt.title("Composition of each sample als")
  plt.xlabel('Sample')
  plt.ylabel('Quantity')
  
  plt.figure(4)
  i=factors1[1]
  plt.plot(abs(i[:,0]),label="Substance 1")
  plt.plot(abs(i[:,1]),label="Substance 2")
  plt.plot(abs(i[:,2]),label="Substance 3")
  plt.legend(loc='best')
  plt.title("Excitation loadings als")
  plt.xlabel('Excitation wavelength')
  plt.ylabel('Loading')
  
  plt.figure(5)
  i=factors1[2]
  plt.plot(abs(i[:,0]),label="Substance 1")
  plt.plot(abs(i[:,1]),label="Substance 2")
  plt.plot(abs(i[:,2]),label="Substance 3")
  plt.legend(loc='best')
  plt.title("Emission loadings als")
  plt.xlabel('Emission wavelength')
  plt.ylabel('Loading')
  
  
def err_time(r,n_samples,n_samples_err,exact_err=False,tol=0.01):
  """
    visualize the stopping criterion along with time. Useful to find an appropriate tolerance for the aamino data.

    Parameters
    ----------
    r : int
        the rank
    n_samples : int
        sample size for HER-CPRAND
    n_samples_err : int
        sample size for fast error estimation in HER-CPRAND
    exact_err : Bool, optional
        evaluate the error exactly or not. The default is False.
    tol : float, optional
        stopping criterion for HER-CPRAND and for ALS. The default is 0.01.


    Returns
    -------
    None.

  """
  path='/Users/sunke/OneDrive/PIR/S8/PIRS8/data/'
  file='amino.mat'
  mat = sio.loadmat(path+file)
  X=mat['X']
  X=np.reshape(X,(5,61,201))
  t=tl.tensor(X)
  list_err1=[]
  list_time1=[]
  list_err2=[]
  list_time2=[]
  list_err3=[]
  list_time3=[]
  list_err4=[]
  list_time4=[]
  list_pct=[]


  for k in range(5): # 5 initializations
    factors=random_init_fac(t,r)   

    weights4,factors4,it4,error4,cpt1,time4=nn_her_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=100,err_it_max=100,tol=tol,time_rec=True) 
    weights1,factors1,it1,error1,cpt,time1=nn_her_Als(t,r,factors=copy.deepcopy(factors),it_max=100,tol=tol,time_rec=True) 
    weights3,factors3,it3,error3,time3=nn_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=100,err_it_max=100,tol=tol,time_rec=True) 
    weights2,factors2,it2,error2,time2=nn_als(t,r,factors=copy.deepcopy(factors),it_max=100,tol=tol,time_rec=True) 

    del error1[0]

    del error2[0]

    del error3[0]

    del error4[0]

    list_pct.append(cpt1)
    list_err1.append(error1)
    list_err2.append(error2)
    list_err3.append(error3)
    list_err4.append(error4)
    list_time1.append(time1)
    list_time2.append(time2)
    list_time3.append(time3)
    list_time4.append(time4)
  list_err1=np.array([np.array(i)  for i in list_err1])
  list_err2=np.array([np.array(i)  for i in list_err2])
  list_err3=np.array([np.array(i)  for i in list_err3])
  list_err4=np.array([np.array(i)  for i in list_err4])

    
  for i in range(len(list_err1)):
    plt.plot(np.cumsum(list_time1[i]),list_err1[i],'b-',linewidth=.3) 
  for i in range(len(list_err2)):
    plt.plot(np.cumsum(list_time2[i]),list_err2[i],'r-',linewidth=.3) 
  for i in range(len(list_err3)):
    plt.plot(np.cumsum(list_time3[i]),list_err3[i],'y-',linewidth=.3) 
  for i in range(len(list_err4)):
    plt.plot(np.cumsum(list_time4[i]),list_err4[i],'g-',linewidth=.3) 
 
  plt.yscale("log") 
  plt.xlabel('time')
  plt.ylabel('data fitting error')
  plt.title('amino')