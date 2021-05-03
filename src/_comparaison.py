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




def comparaison(I,J,K,r,nb_rand,n_samples,exact_err=False,list_factors=False,scale=False):
  """
    old one, need to change hercprand to hercprand_old
    plot data fitting/factors error over nb_rand noised I*J*K rank r random tensors,
    with the median value in bold.
    x aixs is it.
    For each tensor, we have 5 random factors initializations.
    Need to change plot x,y label and title to run the test.

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
    exact_err : boolean, optional
        whether use exact error computation or not for (her)CPRAND. The default is False.
    list_factors : boolean, optional
        whether compute factor error or data fitting error. The default is False.
    scale : boolean, optional
        whether to scale the singular values of matrices or not. The default is False.

    Returns
    -------
    None.

  """
  list_err1=[]
  list_err2=[]
  list_err3=[]
  list_err4=[]
  list_pct=[]
  min_e=None
  for i in range(nb_rand) : 
    # Random initialization of a noised cp_tensor
    # np.random.seed(i)
    fac_true,noise=init_factors(I,J,K,r,scale)
    t=tl.cp_to_tensor((None,fac_true))+noise
    norm_tensor=tl.norm(t,2)
    if(min_e==None) : min_e=norm_tensor
    for k in range(5): # 5 initializations
      factors=random_init_fac(t,r)   
      if list_factors ==False : 
        weights4,factors4,it4,error4,cpt1=nn_her_CPRAND(t,r,n_samples,n_samples_err=400,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200) 
        weights1,factors1,it1,error1,cpt=nn_her_Als(t,r,factors=copy.deepcopy(factors),it_max=200) 
        weights3,factors3,it3,error3=nn_CPRAND(t,r,n_samples,n_samples_err=400,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=200,err_it_max=200) 
        weights2,factors2,it2,error2=nn_als(t,r,factors=copy.deepcopy(factors),it_max=200) 
        # error1=[i * norm_tensor for i in error1]
        # error2=[i * norm_tensor for i in error2]
        # error3=[i * norm_tensor for i in error3]
        # error4=[i * norm_tensor for i in error4]
      else : 
        weights4,factors4,it4,error,_,cpt1,l4=her_CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=500,err_it_max=100,list_factors=list_factors) 
        weights1,factors1,it1,error,cpt,l1=her_Als(t,r,factors=copy.deepcopy(factors),it_max=500,list_factors=list_factors) 
        weights3,factors3,it3,error,_,l3=CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=500,err_it_max=100,list_factors=list_factors) 
        weights2,factors2,it2,error,l2=als(t,r,factors=copy.deepcopy(factors),it_max=500,list_factors=list_factors) 
        error1=[err_fac(fac_true,i) for i in l1]
        error2=[err_fac(fac_true,i) for i in l2]
        error3=[err_fac(fac_true,i) for i in l3]
        error4=[err_fac(fac_true,i) for i in l4]
      list_pct.append(cpt1)
      if (min_e>min(min(error1),min(error2),min(error3),min(error4))) : min_e=min(min(error1),min(error2),min(error3),min(error4))
      list_err1.append(error1)
      list_err2.append(error2)
      list_err3.append(error3)
      list_err4.append(error4)
  list_err1=np.array([np.array(i)  for i in list_err1])
  list_err2=np.array([np.array(i)  for i in list_err2])
  list_err3=np.array([np.array(i)  for i in list_err3])
  list_err4=np.array([np.array(i)  for i in list_err4])
  list_err1=list_err1-min_e
  list_err2=list_err2-min_e
  list_err3=list_err3-min_e
  list_err4=list_err4-min_e
  for i in list_err1:
    plt.plot(range(len(i)),i,'b-',linewidth=.3) 
  for i in list_err2:
    plt.plot(range(len(i)),i,'r-',linewidth=.3) 
  for i in list_err3:
    plt.plot(range(len(i)),i,'y-',linewidth=.3) 
  for i in list_err4:
    plt.plot(range(len(i)),i,'g-',linewidth=.3) 
  n_max1=len(max(list_err1, key=len)) # length of the longest error
  n_max2=len(max(list_err2, key=len))
  n_max3=len(max(list_err3, key=len))
  n_max4=len(max(list_err4, key=len))
  mat1=np.array([i.tolist() + [i[len(i)-1]]*(n_max1-len(i)) for i in list_err1])
  mat2=np.array([i.tolist() + [i[len(i)-1]]*(n_max2-len(i)) for i in list_err2])
  mat3=np.array([i.tolist() + [i[len(i)-1]]*(n_max3-len(i)) for i in list_err3])
  mat4=np.array([i.tolist() + [i[len(i)-1]]*(n_max4-len(i)) for i in list_err4])
  
  # plot
  plt.plot(range(n_max1),np.median(mat1, axis=0),'b-',linewidth=3,label="her_als") 
  plt.plot(range(n_max2),np.median(mat2, axis=0),'r-',linewidth=3,label="als") 
  plt.plot(range(n_max3),np.median(mat3, axis=0),'y-',linewidth=3,label="CPRAND") 
  plt.plot(range(n_max4),np.median(mat4, axis=0),'g-',linewidth=3,label="herCPRAND") 
  plt.yscale("log") 
  plt.xlabel('it')
  plt.ylabel('data fitting error')
  plt.legend(loc='best')
  plt.title('Complicated case')


def compar_time(I,J,K,r,nb_rand,n_samples,exact_err=False,list_factors=False,scale=False):
  """
    old one, need to change hercprand to hercprand_old
    plot data fitting/factors error over nb_rand noised I*J*K rank r random tensors,
    with the median value in bold.
    x aixs is time.
    For each tensor, we have 5 factors initializations.
    Need to change plot x,y label and title to run the test.

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
        whether use exact error computation or not for herCPRAND. The default is False.
    list_factors : boolean, optional
        whether compute factor error or data fitting error. The default is False.
    scale : boolean, optional
        whether to scale the singular values of matrices or not. The default is False.

    Returns
    -------
    None.

  """
  list_err1=[]
  list_time1=[]
  list_err2=[]
  list_time2=[]
  list_err3=[]
  list_time3=[]
  list_err4=[]
  list_time4=[]
  min_e=None
  for i in range(nb_rand) : 
    # Random initialization of a noised cp_tensor
    np.random.seed(i)
    fac_true,noise=init_factors(I,J,K,r,scale)
    t=tl.cp_to_tensor((None,fac_true))+noise
    norm_tensor=tl.norm(t,2)
    if(min_e==None) : min_e=norm_tensor
    for k in range(5):
      factors=random_init_fac(t,r)
      if list_factors ==False : 
        weights4,factors4,it4,error4,_,cpt,time4=her_CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=500,err_it_max=250,time_rec=True) 
        weights1,factors1,it1,error1,cpt,time1=her_Als(t,r,factors=copy.deepcopy(factors),it_max=500,time_rec=True) 
        weights3,factors3,it3,error3,_,time3=CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=500,err_it_max=250,time_rec=True) 
        weights2,factors2,it2,error2,time2=als(t,r,factors=copy.deepcopy(factors),it_max=500,time_rec=True) 
        error1=[i * norm_tensor for i in error1]
        del error1[0]
        error2=[i * norm_tensor for i in error2]
        del error2[0]
        error3=[i * norm_tensor for i in error3]
        del error3[0]
        error4=[i * norm_tensor for i in error4]
        del error4[0]
      else : 
        weights4,factors4,it4,error,_,cpt,l4,time4=her_CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=500,err_it_max=250,list_factors=list_factors,time_rec=True) 
        weights1,factors1,it1,error,cpt,l1,time1=her_Als(t,r,factors=copy.deepcopy(factors),it_max=500,list_factors=list_factors,time_rec=True) 
        weights3,factors3,it3,error,_,l3,time3=CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=500,err_it_max=250,list_factors=list_factors,time_rec=True) 
        weights2,factors2,it2,error,l2,time2=als(t,r,factors=copy.deepcopy(factors),it_max=500,list_factors=list_factors,time_rec=True) 
        error1=[err_fac(fac_true,i) for i in l1]
        del error1[0]
        error2=[err_fac(fac_true,i) for i in l2]
        del error2[0]
        error3=[err_fac(fac_true,i) for i in l3]
        del error3[0]
        error4=[err_fac(fac_true,i) for i in l4]
        del error4[0]
      if (min_e>min(min(error1),min(error2),min(error3),min(error4))) : min_e=min(min(error1),min(error2),min(error3),min(error4))
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
  list_err1=list_err1-min_e
  list_err2=list_err2-min_e
  list_err3=list_err3-min_e
  list_err4=list_err4-min_e
  for i in range(len(list_err1)):
    plt.plot(np.cumsum(list_time1[i]),list_err1[i],'b-',linewidth=.3) 
  for i in range(len(list_err2)):
    plt.plot(np.cumsum(list_time2[i]),list_err2[i],'r-',linewidth=.3) 
  for i in range(len(list_err3)):
    plt.plot(np.cumsum(list_time3[i]),list_err3[i],'y-',linewidth=.3) 
  for i in range(len(list_err4)):
    plt.plot(np.cumsum(list_time4[i]),list_err4[i],'g-',linewidth=.3) 
  n_max1=len(max(list_err1, key=len)) # length of the longest error
  n_max2=len(max(list_err2, key=len))
  n_max3=len(max(list_err3, key=len))
  n_max4=len(max(list_err4, key=len))
  mat1=np.array([i.tolist() + [i[len(i)-1]]*(n_max1-len(i)) for i in list_err1])
  mat2=np.array([i.tolist() + [i[len(i)-1]]*(n_max2-len(i)) for i in list_err2])
  mat3=np.array([i.tolist() + [i[len(i)-1]]*(n_max3-len(i)) for i in list_err3])
  mat4=np.array([i.tolist() + [i[len(i)-1]]*(n_max4-len(i)) for i in list_err4])
  

  t_max1=len(max(list_time1, key=len))
  t_max2=len(max(list_time2, key=len))
  t_max3=len(max(list_time3, key=len))
  t_max4=len(max(list_time4, key=len))
  mat_time1=np.array([i + [0]*(t_max1-len(i)) for i in list_time1])
  mat_time2=np.array([i + [0]*(t_max2-len(i)) for i in list_time2])
  mat_time3=np.array([i + [0]*(t_max3-len(i)) for i in list_time3])
  mat_time4=np.array([i + [0]*(t_max4-len(i)) for i in list_time4])
  # plot

  plt.plot(np.cumsum(np.median(mat_time1, axis=0)),np.median(mat1, axis=0),'b-',linewidth=3,label="her als") 
  plt.plot(np.cumsum(np.median(mat_time2, axis=0)),np.median(mat2, axis=0),'r-',linewidth=3,label="als") 
  plt.plot(np.cumsum(np.median(mat_time3, axis=0)),np.median(mat3, axis=0),'y-',linewidth=3,label="CPRAND") 
  plt.plot(np.cumsum(np.median(mat_time4, axis=0)),np.median(mat4, axis=0),'g-',linewidth=3,label="herCPRAND") 
  plt.yscale("log") 
  plt.xlabel('time')
  plt.ylabel('data fitting error')
  plt.legend(loc='best')
  plt.title('Simple case exact')
  
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
    #norm_tensor=tl.norm(t,2)
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
    #norm_tensor=tl.norm(t,2)
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
    #fac_true,noise=init_factors_hernn(I,J,K,r,noise_level,scale)
    fac_true,noise=init_factors(I,J,K,r,noise_level,scale,nn=True)
    #t=tl.clip(tl.cp_to_tensor((None,fac_true))+noise,a_min=0.0)
    t=tl.cp_to_tensor((None,fac_true))+noise
    #norm_tensor=tl.norm(t,2)
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
  
def nn_err_it(I,J,K,r,nb_rand,n_samples,n_samples_err,exact_err=False,list_factors=False,scale=False,tol=0.0001):
  """
    old one, need to change hercprand to hercprand_old
    plot data fitting/factors error over nb_rand noised I*J*K rank r random tensors,
    with the median value in bold.
    x aixs is it.
    For each tensor, we have 5 random factors initializations.
    Need to change plot x,y label and title to run the test.

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
    exact_err : boolean, optional
        whether use exact error computation or not for (her)CPRAND. The default is False.
    list_factors : boolean, optional
        whether compute factor error or data fitting error. The default is False.
    scale : boolean, optional
        whether to scale the singular values of matrices or not. The default is False.

    Returns
    -------
    None.

  """
  list_err1=[]
  list_err2=[]
  list_err3=[]
  list_err4=[]
  list_pct=[]
  min_e=None
  for i in range(nb_rand) : 
    # Random initialization of a noised cp_tensor
    # np.random.seed(i)
    fac_true,noise=init_factors(I,J,K,r,scale)
    t=tl.cp_to_tensor((None,fac_true))+noise
    norm_tensor=tl.norm(t,2)
    if(min_e==None) : min_e=norm_tensor
    for k in range(5): # 5 initializations
      factors=random_init_fac(t,r)   
      if list_factors ==False :
 
          
          
        weights4,factors4,it4,_,error4,cpt1=nn_her_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=100,err_it_max=100,tol=tol) 
        weights1,factors1,it1,error1,cpt=nn_her_Als(t,r,factors=copy.deepcopy(factors),it_max=100,tol=tol) 
        weights3,factors3,it3,_,error3=nn_CPRAND(t,r,n_samples,n_samples_err,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=100,err_it_max=100,tol=tol) 
        weights2,factors2,it2,error2=nn_als(t,r,factors=copy.deepcopy(factors),it_max=100,tol=tol) 
        # error1=[i * norm_tensor for i in error1]
        # error2=[i * norm_tensor for i in error2]
        # error3=[i * norm_tensor for i in error3]
        # error4=[i * norm_tensor for i in error4]
      else : 
        weights4,factors4,it4,error,_,cpt1,l4=her_CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=500,err_it_max=100,list_factors=list_factors) 
        weights1,factors1,it1,error,cpt,l1=her_Als(t,r,factors=copy.deepcopy(factors),it_max=500,list_factors=list_factors) 
        weights3,factors3,it3,error,_,l3=CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=exact_err,it_max=500,err_it_max=100,list_factors=list_factors) 
        weights2,factors2,it2,error,l2=als(t,r,factors=copy.deepcopy(factors),it_max=500,list_factors=list_factors) 
        error1=[err_fac(fac_true,i) for i in l1]
        error2=[err_fac(fac_true,i) for i in l2]
        error3=[err_fac(fac_true,i) for i in l3]
        error4=[err_fac(fac_true,i) for i in l4]
      list_pct.append(cpt1)
      if (min_e>min(min(error1),min(error2),min(error3),min(error4))) : min_e=min(min(error1),min(error2),min(error3),min(error4))
      list_err1.append(error1)
      list_err2.append(error2)
      list_err3.append(error3)
      list_err4.append(error4)
  list_err1=np.array([np.array(i)  for i in list_err1])
  list_err2=np.array([np.array(i)  for i in list_err2])
  list_err3=np.array([np.array(i)  for i in list_err3])
  list_err4=np.array([np.array(i)  for i in list_err4])
  list_err1=list_err1-min_e
  list_err2=list_err2-min_e
  list_err3=list_err3-min_e
  list_err4=list_err4-min_e
  for i in list_err1:
    plt.plot(range(len(i)),i,'b-',linewidth=.3) 
  for i in list_err2:
    plt.plot(range(len(i)),i,'r-',linewidth=.3) 
  for i in list_err3:
    plt.plot(range(len(i)),i,'y-',linewidth=.3) 
  for i in list_err4:
    plt.plot(range(len(i)),i,'g-',linewidth=.3) 
  n_max1=len(max(list_err1, key=len)) # length of the longest error
  n_max2=len(max(list_err2, key=len))
  n_max3=len(max(list_err3, key=len))
  n_max4=len(max(list_err4, key=len))
  mat1=np.array([i.tolist() + [i[len(i)-1]]*(n_max1-len(i)) for i in list_err1])
  mat2=np.array([i.tolist() + [i[len(i)-1]]*(n_max2-len(i)) for i in list_err2])
  mat3=np.array([i.tolist() + [i[len(i)-1]]*(n_max3-len(i)) for i in list_err3])
  mat4=np.array([i.tolist() + [i[len(i)-1]]*(n_max4-len(i)) for i in list_err4])
  
  # plot
  plt.plot(range(n_max1),np.median(mat1, axis=0),'b-',linewidth=.3,label="her_als") 
  plt.plot(range(n_max2),np.median(mat2, axis=0),'r-',linewidth=.3,label="als") 
  plt.plot(range(n_max3),np.median(mat3, axis=0),'y-',linewidth=.3,label="CPRAND") 
  plt.plot(range(n_max4),np.median(mat4, axis=0),'g-',linewidth=.3,label="herCPRAND") 
  plt.yscale("log") 
  plt.xlabel('it')
  plt.ylabel('data fitting error')
  plt.legend(loc='best')
  plt.title('Complicated case')