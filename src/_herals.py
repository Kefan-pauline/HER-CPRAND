"""
her als meethode for CP decomposition
"""
import numpy as np
import tensorly as tl
import time
from src._base import svd_init_fac,err
from src._als import err_fast
import copy
from tensorly.tenalg.proximal import hals_nnls

def her_Als(tensor,rank,factors=None,it_max=100,tol=1e-7,beta=0.5,eta=1.5,gamma=1.05,gamma_bar=1.01,list_factors=False,error_fast=True,time_rec=False):
  """
    her ALS methode of CP decomposition

    Parameters
    ----------
    tensor : tensor
    rank : int
    factors : list of matrices, optional
        an initial factor matrices. The default is None.
    it_max : int, optional
        maximal number of iteration. The default is 100.
    tol : float, optional
        error tolerance. The default is 1e-7.
    beta : float, optional
        extrapolation parameter. The default is 0.5.
    eta : float, optional
        decrease coefficient of beta. The default is 1.5.
    gamma : float, optional
        increase coefficient of beta. The default is 1.05.
    gamma_bar : float, optional
        increase coeefficient of beta_bar. The default is 1.01.
    list_factors : boolean, optional
        If true, then return factor matrices of each iteration. The default is False.
    error_fast : boolean, optional
        If true, use err_fast to compute data fitting error, otherwise, use err. The default is True.
    time_rec : boolean, optional
        If true, return computation time of each iteration. The default is False.

    Returns
    -------
    the CP decomposition, number of iteration, error and restart pourcentage. 
    list_fac and list_time are optional.
  """
  beta_bar=1
  N=tl.ndim(tensor) # order of tensor
  norm_tensor=tl.norm(tensor) # norm of tensor
  weights=None
  if time_rec == True : list_time=[]
  if list_factors==True : list_fac=[]

  if (factors==None): factors=svd_init_fac(tensor,rank)

  # Initialization of factor hat matrices by factor matrices
  factors_hat=factors
  if list_factors==True : list_fac.append(copy.deepcopy(factors))

  it=0
  cpt=0
  F_hat_bf=err(tensor,None,factors) # cost
  error=[F_hat_bf/norm_tensor]

  while (error[len(error)-1] > tol and it<it_max ):
    if time_rec == True : tic=time.time()
    for n in range(N):
      V=np.ones((rank,rank))
      for i in range(len(factors)):
        if i != n : V=V*tl.dot(tl.transpose(factors_hat[i]),factors_hat[i])
      W=tl.cp_tensor.unfolding_dot_khatri_rao(tensor, (None,factors_hat), n) 
      factor_bf=factors[n]
      # update
      factors[n]= tl.transpose(tl.solve(tl.transpose(V),tl.transpose(W)))
      # extrapolate
      factors_hat[n]=factors[n]+beta*(factors[n]-factor_bf)

    if (error_fast==False) : F_hat_new = err(tensor,None,factors_hat) # cost update 
    else : F_hat_new = err_fast(norm_tensor,factors[N-1],V,W)
    if (F_hat_new>F_hat_bf):
      factors_hat=factors
      beta_bar=beta
      beta=beta/eta
      cpt=cpt+1
    else :
      factors=factors_hat
      beta_bar=min(1,beta_bar*gamma_bar)
      beta=min(beta_bar,gamma*beta)
    F_hat_bf=F_hat_new
    it=it+1
    if list_factors==True : list_fac.append(copy.deepcopy(factors))
    error.append(F_hat_new/norm_tensor)
    if time_rec == True : 
      toc=time.time()
      list_time.append(toc-tic)
  if time_rec == True and list_factors==True: return(weights,factors,it,error,cpt/it,list_fac,list_time)
  if list_factors==True : return(weights,factors,it,error,cpt/it,list_fac)
  if time_rec == True : return(weights,factors,it,error,cpt/it,list_time)
  return(weights,factors,it,error,cpt/it)


def nn_her_Als(tensor,rank,factors=None,it_max=100,tol=1e-7,beta=0.5,eta=1.5,gamma=1.05,gamma_bar=1.01,list_factors=False,error_fast=True,time_rec=False):
  """
    her ALS methode of CP decomposition for non negative case

    Parameters
    ----------
    tensor : tensor
    rank : int
    factors : list of matrices, optional
        an initial non negative factor matrices. The default is None.
    it_max : int, optional
        maximal number of iteration. The default is 100.
    tol : float, optional
        error tolerance. The default is 1e-7.
    beta : float, optional
        extrapolation parameter. The default is 0.5.
    eta : float, optional
        decrease coefficient of beta. The default is 1.5.
    gamma : float, optional
        increase coefficient of beta. The default is 1.05.
    gamma_bar : float, optional
        increase coeefficient of beta_bar. The default is 1.01.
    list_factors : boolean, optional
        If true, then return factor matrices of each iteration. The default is False.
    error_fast : boolean, optional
        If true, use err_fast to compute data fitting error, otherwise, use err. The default is True.
    time_rec : boolean, optional
        If true, return computation time of each iteration. The default is False.

    Returns
    -------
    the CP decomposition, number of iteration, error and restart pourcentage. 
    list_fac and list_time are optional.
  """
  beta_bar=1
  N=tl.ndim(tensor) # order of tensor
  norm_tensor=tl.norm(tensor) # norm of tensor
  weights=None
  if time_rec == True : list_time=[]
  if list_factors==True : list_fac=[]

  if (factors==None): factors=svd_init_fac(tensor,rank)

  # Initialization of factor hat matrices by factor matrices
  factors_hat=factors
  if list_factors==True : list_fac.append(copy.deepcopy(factors))

  it=0
  cpt=0
  F_hat_bf=err(tensor,None,factors) # cost
  error=[F_hat_bf/norm_tensor]

  while (error[len(error)-1] > tol and it<it_max ):
    if time_rec == True : tic=time.time()
    for n in range(N):
      V=np.ones((rank,rank))
      for i in range(len(factors)):
        if i != n : V=V*tl.dot(tl.transpose(factors_hat[i]),factors_hat[i])
      W=tl.cp_tensor.unfolding_dot_khatri_rao(tensor, (None,factors_hat), n) 
      factor_bf=factors[n]
      # update
      fac, _, _, _ = hals_nnls(tl.transpose(W), V,tl.transpose(factors[n]))
      factors[n]=tl.transpose(fac)
      # extrapolate
      factors_hat[n]=tl.clip(factors[n]+beta*(factors[n]-factor_bf),a_min=0.0)

    if (error_fast==False) : F_hat_new = err(tensor,None,factors_hat) # cost update 
    else : F_hat_new = err_fast(norm_tensor,factors[N-1],V,W)
    if (F_hat_new>F_hat_bf):
      factors_hat=factors
      beta_bar=beta
      beta=beta/eta
      cpt=cpt+1
    else :
      factors=factors_hat
      beta_bar=min(1,beta_bar*gamma_bar)
      beta=min(beta_bar,gamma*beta)
    F_hat_bf=F_hat_new
    it=it+1
    if list_factors==True : list_fac.append(copy.deepcopy(factors))
    error.append(F_hat_new/norm_tensor)
    if time_rec == True : 
      toc=time.time()
      list_time.append(toc-tic)
  if time_rec == True and list_factors==True: return(weights,factors,it,error,cpt/it,list_fac,list_time)
  if list_factors==True : return(weights,factors,it,error,cpt/it,list_fac)
  if time_rec == True : return(weights,factors,it,error,cpt/it,list_time)
  return(weights,factors,it,error,cpt/it)