"""
herCPRAND meethode for CP decomposition 
return also exact error
"""

import tensorly as tl
import copy 
import time
from src._base import svd_init_fac,err
from tensorly.decomposition import sample_khatri_rao
from src._cprand import err_rand,err_rand_fast


def her_CPRAND(tensor,rank,n_samples,factors=None,exact_err=True,it_max=100,err_it_max=20,tol=1e-7,beta=0.1,eta=3,gamma=1.01,gamma_bar=1.005,list_factors=False,time_rec=False):
  """
    herCPRAND for CP-decomposition
    return also exact error

    Parameters
    ----------
    tensor : tensor
    rank : int
    n_samples : int
        sample size
     factors : list of matrices, optional
        an initial factor matrices. The default is None.
    exact_err : boolean, optional
        whether use err or err_rand_fast for terminaison criterion. The default is False.
        (not useful for this version)
    it_max : int, optional
        maximal number of iteration. The default is 100.
    err_it_max : int, optional
        maximal of iteration if terminaison critirion is not improved. The default is 20.
    tol : float, optional
        error tolerance. The default is 1e-7.
    beta : float, optional
        extrapolation parameter. The default is 0.1.
    eta : float, optional
        decrease coefficient of beta. The default is 3.
    gamma : float, optional
        increase coefficient of beta. The default is 1.01.
    gamma_bar : float, optional
        increase coeefficient of beta_bar. The default is 1.005.
    list_factors : boolean, optional
        If true, then return factor matrices of each iteration. The default is False.
    time_rec : boolean, optional
        If true, return computation time of each iteration. The default is False.

    Returns
    -------
    the CP decomposition, number of iteration and exact / estimated termination criterion. 
    list_fac and list_time are optional.

    """
  beta_bar=1
  N=tl.ndim(tensor) # order of tensor
  norm_tensor=tl.norm(tensor) # norm of tensor
  if list_factors==True : list_fac=[]
  if (time_rec==True) : list_time=[]

  if (factors==None) : factors=svd_init_fac(tensor,rank)
  # Initialization of factor hat matrice by factor matrices
  factors_hat=factors
  if list_factors==True : list_fac.append(copy.deepcopy(factors))

  weights=None
  it=0
  err_it=0
  cpt=0
  ########################################
  ######### error initialization #########
  ########################################
  F_hat_bf,ind_bf=err_rand(tensor,None,factors,n_samples)
  F_hat_bf_ex=err(tensor,None,factors) # exact cost
  rng = tl.random.check_random_state(None)
  error=[F_hat_bf/norm_tensor]
  error_ex=[F_hat_bf_ex/norm_tensor]
  min_err=error[len(error)-1]

  while (min_err>tol and it<it_max and err_it<err_it_max): 
    if time_rec==True :tic=time.time()
    factors_hat_bf=factors_hat
    for n in range(N):
      Zs,indices=sample_khatri_rao(factors_hat,n_samples,skip_matrix=n,random_state=rng)
      indices_list = [i.tolist() for i in indices]
      indices_list.insert(n, slice(None, None, None))
      indices_list = tuple(indices_list)
      V=tl.dot(tl.transpose(Zs),Zs)
      # J'ai du mal avec la syntaxe tensor[indices_list],
      # Ca renvoie une matrices et non un tenseur?
      if (n==0) :sampled_unfolding = tensor[indices_list] 
      else : sampled_unfolding = tl.transpose(tensor[indices_list])
      W=tl.dot(sampled_unfolding,Zs)
      factor_bf=factors[n]
      # update
      factors[n] = tl.transpose(tl.solve(V,tl.transpose(W))) # solve needs a squared full rank matrix, if rank>nb_sampls ok
      # if (n==N-1) : F_hat_new=tl.norm(tl.dot(Zs,tl.transpose(factors[n]))-sampled_unfolding,2) # cost update 
      # extrapolate
      factors_hat[n]=factors[n]+beta*(factors[n]-factor_bf)
    ########################################
    #########      error update    #########
    ########################################

    matrices = factors_hat_bf[:-1]
    Zs_bf = tl.ones((n_samples, rank), **tl.context(matrices[0]))
    for indices, matrix in zip(indices_list, matrices): 
      Zs_bf = Zs_bf * matrix[indices, :]
    V_bf=tl.dot(tl.transpose(Zs_bf),Zs_bf)
    W_bf=tl.dot(tl.transpose(tensor[indices_list]),Zs_bf)
    F_hat_bf,a= err_rand_fast(tensor,factor_bf,V_bf,W_bf,indices_list,n_samples)
    F_hat_new,_= err_rand_fast(tensor,factors[N-1],V,W,indices_list,n_samples) 
 
    if (F_hat_new>F_hat_bf):
      factors_hat=factors
      beta_bar=beta
      beta=beta/eta
      cpt=cpt+1
    else :
      factors=factors_hat
      beta_bar=min(1,beta_bar*gamma_bar)
      beta=min(beta_bar,gamma*beta)
    ########################################
    ######### update for next it   #########
    ########################################
    it=it+1
    if list_factors==True : list_fac.append(copy.deepcopy(factors))
    error.append(F_hat_new/norm_tensor)
    if (error[len(error)-1]<min_err) : min_err=error[len(error)-1] # err update
    else : err_it=err_it+1
    if time_rec==True : 
      toc=time.time()
      list_time.append(toc-tic)
    error_ex.append(err(tensor,None,factors)/norm_tensor)  # exact cost update
  # weights,factors=tl.cp_normalize((None,factors))
  if list_factors==True and time_rec==True: return(weights,factors,it,error_ex,error,cpt/it,list_fac,list_time)
  if list_factors==True : return(weights,factors,it,error_ex,error,cpt/it,list_fac)
  if time_rec==True : return(weights,factors,it,error_ex,error,cpt/it,list_time)
  return(weights,factors,it,error_ex,error,cpt/it)