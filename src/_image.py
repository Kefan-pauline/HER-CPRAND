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







def image_boxplot(r,n_samples,n_samples_err,exact_err=False,tol=0.0001):
  path='/Users/sunke/OneDrive/PIR/S8/PIRS8/data/'
  file='Indian_pines_corrected.mat'
  mat = sio.loadmat(path+file)
  X=mat['indian_pines_corrected']
  X=np.reshape(X,(145,145,200))
  t=tl.tensor(X)
  t=t.astype(float)
  
  fit={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  it_tot={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  time_tot={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  restart={"herals" :  [],"hercprand" :  [],"hercprand small" :  []}
  # local variables
  error={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}
  it={1 :  0, 2 :  0, 3 :  0, 4 :  0, 5 :  0, 6 :  0}
  time={1 :  [],2 :  [],3 :  [],4 :  [],5 :  [],6 :  []}

    #norm_tensor=tl.norm(t,2)
  for k in range(5):
      # random initialization of factors
    factors=random_init_fac(t,r)
      # run 4 methods
    weights1,_,it[1],error[1],cpt1,l_fac1,time[1]=nn_her_Als(t,r,factors=copy.deepcopy(factors),it_max=100,tol=tol,list_factors=True,time_rec=True)  
    weights2,_,it[2],error[2],l_fac2,time[2]=nn_als(t,r,factors=copy.deepcopy(factors),it_max=100,tol=tol,list_factors=True,time_rec=True) 
    weights3,_,it[3],error[3],l_fac3,time[3]=nn_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=100,err_it_max=200,tol=tol,list_factors=True,time_rec=True)
    weights4,_,it[4],error[4],cpt4,l_fac4,time[4]=nn_her_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=100,err_it_max=200,tol=tol,list_factors=True,time_rec=True) 
    weights5,_,it[5],error[5],l_fac5,time[5]=nn_CPRAND(t,r,200,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=100,err_it_max=200,tol=tol,list_factors=True,time_rec=True)
    weights6,_,it[6],error[6],cpt6,l_fac6,time[6]=nn_her_CPRAND(t,r,200,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=100,err_it_max=200,tol=tol,list_factors=True,time_rec=True) 
      # information storage
    restart["herals"].append(cpt1)
    restart["hercprand"].append(cpt4)
    restart["hercprand small"].append(cpt6)
    for j in range(1,7):
      fit[j].append(1-(error[j][len(error[j])-1]))
      it_tot[j].append(it[j])
      time_tot[j].append(np.cumsum(time[j])[len(time[j])-1])
  # figure
  labels=["herals","als","cprand","hercprand","cprand small","hercprand small"]
  _, dataf = [*zip(*fit.items())]
  _, datai = [*zip(*it_tot.items())]
  _, datat = [*zip(*time_tot.items())]
  _, datar = [*zip(*restart.items())]
  plt.figure(0)
  plt.boxplot(dataf,vert=False)
  plt.yticks(range(1, len(labels) + 1), labels)
  plt.title('fits')
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
 