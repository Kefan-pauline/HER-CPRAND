"""
CPRAND meethode for CP decomposition and corresponding error estimation
return also exact error
"""

import numpy as np
import tensorly as tl
import copy 
import time
from src._base import svd_init_fac,err
from tensorly.decomposition import sample_khatri_rao


def err_rand(tensor,weights,factors,nb_samples,indices_list=None): 
  """
    Error estimation proposed in CPRAND

    Parameters
    ----------
    tensor : tensor
    weights : vector
        the weights of CP decomposition
    factors : list of matrices
        factor matrices of CP decomposition
    nb_samples : int
        nb of sample
    indices_list : tuple, optional
        indices list of sample. The default is None.

    Returns
    -------
    float
        the error estimation and the used indices_list

  """
  # if indices_list is not given
  if indices_list==None : 
    #indices_list = [np.random.randint(0, tl.shape(m)[0], size=nb_samples, dtype=int) for m in factors] 
    indices_list = [np.random.choice(tl.shape(m)[0], nb_samples) for m in factors]   # to avoid duplicate choices
    indices_list = [i.tolist() for i in indices_list]
    indices_list = tuple(indices_list)
  est_values=[]
  # nb of terms in tensor
  P=1
  for i in tl.shape(tensor) : P=P*i
  for i in range(nb_samples):
     if weights is None : value=1
     else : value=weights
     for mode in range(tl.ndim(tensor)) :
       value=value*factors[mode][indices_list[mode][i],:]
     est_values+=[sum(value)]
  list_e=(tensor[indices_list]-est_values)**2  
  # assume max(list_e) = 1 if terms are in [0,1]
  return(np.sqrt(sum(list_e)*P/nb_samples),indices_list) 


def err_rand_fast(tensor,A,V,W,indices_list,nb_samples=None): 
  """
    randomised err_fast as for als

    Parameters
    ----------
    tensor : tensor
    A : matrix
        factor matrix
    V : matrix
        random matrix V defined as in CPRAND
    W : matrix
        random matrix W defined as in CPRAND
    indices_list : tuple
        indices list used for V and W
    nb_samples : int, optional
        sample size. The default is None.

    Returns
    -------
    error estimation, used indices list

  """
  # randomised tensor norm  
  norm_tensor=tl.norm(tensor[indices_list])
  res=sum(sum(V*(np.transpose(A).dot(A))))
  res=norm_tensor**2+res-2*sum(sum(W*A)) 
  if nb_samples == None : nb_samples = np.shape(indices_list[0])[0]
  res=res/nb_samples
  P=1
  for i in tl.shape(tensor) : P=P*i
  return(np.sqrt(res*P),indices_list) 


def CPRAND(tensor,rank,n_samples,factors=None,exact_err=False,it_max=100,err_it_max=20,tol=1e-7,list_factors=False,time_rec=False):
  """
    CPRAND for CP-decomposition
    return also exact error

    Parameters
    ----------
    tensor : tensor
    rank : int
    n_samples : int
        sample size
    factors : list of matrices, optional
        initial factor matrices. The default is None.
    exact_err : boolean, optional
        whether use err or err_rand_fast for terminaison criterion. The default is False.
        (not useful for this version)
    it_max : int, optional
        maximal number of iteration. The default is 100.
    err_it_max : int, optional
        maximal of iteration if terminaison critirion is not improved. The default is 20.
    tol : float, optional
        error tolerance. The default is 1e-7.
    list_factors : boolean, optional
        If true, then return factor matrices of each iteration. The default is False.
    time_rec : boolean, optional
        If true, return computation time of each iteration. The default is False.

    Returns
    -------
    the CP decomposition, number of iteration and exact / estimated termination criterion. 
    list_fac and list_time are optional.

    """
  N=tl.ndim(tensor) # order of tensor
  norm_tensor=tl.norm(tensor) # norm of tensor

  if list_factors==True : list_fac=[]
  if time_rec == True : list_time=[]
  if (factors==None): factors=svd_init_fac(tensor,rank)
  # weights,factors=tl.cp_tensor.cp_normalize((None,factors))
  if list_factors==True : list_fac.append(copy.deepcopy(factors))
  weights=None
  it=0
  err_it=0
  ########################################
  ######### error initialization #########
  ########################################
  temp,ind_err=err_rand(tensor,weights,factors,400) 
  error=[temp/norm_tensor] 
  error_ex=[err(tensor,weights,factors)/norm_tensor]

  min_err=error[len(error)-1]
  rng = tl.random.check_random_state(None)
  while (min_err>tol and it<it_max and err_it<err_it_max): 
    if time_rec == True : tic=time.time()
    for n in range(N):
      Zs,indices=sample_khatri_rao(factors,n_samples,skip_matrix=n,random_state=rng)
      indices_list = [i.tolist() for i in indices]
      indices_list.insert(n, slice(None, None, None))
      indices_list = tuple(indices_list)
      if (n==0) :sampled_unfolding = tensor[indices_list]
      else : sampled_unfolding =tl.transpose(tensor[indices_list])
      V=tl.dot(tl.transpose(Zs),Zs)
      W=tl.dot(sampled_unfolding,Zs)
      # update
      factors[n] = tl.transpose(tl.solve(V,tl.transpose(W))) # solve needs a squared matrix
    

    if list_factors==True : list_fac.append(copy.deepcopy(factors))
    it=it+1

    ################################
    ######### error update #########
    ################################
    error.append(err_rand_fast(tensor,factors[N-1],V,W,indices_list,n_samples)[0]/norm_tensor) # same indices used as for Random Lesat Square Calculation

    
    if (error[len(error)-1]<min_err) : min_err=error[len(error)-1] # err update
    else : err_it=err_it+1
    if time_rec == True : 
      toc=time.time()
      list_time.append(toc-tic)
    error_ex.append(err(tensor,weights,factors)/norm_tensor)
  # weights,factors=tl.cp_tensor.cp_normalize((None,factors))
  if time_rec == True and list_factors==True: return(weights,factors,it,error_ex,error,list_fac,list_time)
  if list_factors==True : return(weights,factors,it,error_ex,error,list_fac)
  if time_rec==True : return(weights,factors,it,error_ex,error,list_time)
  return(weights,factors,it,error_ex,error)
