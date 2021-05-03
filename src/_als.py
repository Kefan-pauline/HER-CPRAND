"""
ALS methode for CP decomposition
"""
import numpy as np
import tensorly as tl
import time
import copy
from src._base import svd_init_fac,err
from tensorly.tenalg.proximal import hals_nnls


def err_fast(norm_tensor,A,V,W):
  """
    fast data fitting error calculation of als
    Parameters
    ----------
    norm_tensor : float
        norm of the tensor
    A : matrix
        factor matrix
    V : matrix
        matrix V defined as in als
    W : matrix
        matrix W defined as in als

    Returns
    -------
    float
    data fitting error

  """
  res=sum(sum(V*(np.transpose(A).dot(A))))
  res=res-2*sum(sum(W*A))
  return(np.sqrt(norm_tensor**2+res))


def als(tensor,rank,factors=None,it_max=100,tol=1e-7,list_factors=False,error_fast=True,time_rec=False):
  """
    ALS methode of CP decomposition

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
    list_factors : boolean, optional
        If true, then return factor matrices of each iteration. The default is False.
    error_fast : boolean, optional
        If true, use err_fast to compute data fitting error, otherwise, use err. The default is True.
    time_rec : boolean, optional
        If true, return computation time of each iteration. The default is False.

    Returns
    -------
    the CP decomposition, number of iteration and termination criterion. 
    list_fac and list_time are optional.
  """
  N=tl.ndim(tensor) # order of tensor
  norm_tensor=tl.norm(tensor) # norm of tensor
  if time_rec == True : list_time=[]
  if list_factors==True : list_fac=[] # list of factor matrices

  if (factors==None): factors=svd_init_fac(tensor,rank)

  weights=None
  it=0
  if list_factors==True : list_fac.append(copy.deepcopy(factors))
  error=[err(tensor,weights,factors)/norm_tensor]
  while (error[len(error)-1]>tol and it<it_max):
    if time_rec == True : tic=time.time() 
    for n in range(N):
      V=np.ones((rank,rank))
      for i in range(len(factors)):
        if i != n : V=V*tl.dot(tl.transpose(factors[i]),factors[i])
      W=tl.cp_tensor.unfolding_dot_khatri_rao(tensor, (None,factors), n) 
      factors[n]= tl.transpose(tl.solve(tl.transpose(V),tl.transpose(W)))
    if list_factors==True : list_fac.append(copy.deepcopy(factors))
    it=it+1
    if (error_fast==False) : error.append(err(tensor,weights,factors)/norm_tensor)
    else : error.append(err_fast(norm_tensor,factors[N-1],V,W)/norm_tensor)
    if time_rec == True : 
      toc=time.time() 
      list_time.append(toc-tic)
  # weights,factors=tl.cp_tensor.cp_normalize((None,factors))
  if list_factors==True and time_rec==True: return(weights,factors,it,error,list_fac,list_time)
  if time_rec==True : return(weights,factors,it,error,list_time)
  if list_factors==True : return(weights,factors,it,error,list_fac)
  return(weights,factors,it,error)

def nn_als(tensor,rank,factors,it_max=100,tol=1e-7,list_factors=False,error_fast=True,time_rec=False):
  """
    ALS methode of CP decomposition for non negative case

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
    list_factors : boolean, optional
        If true, then return factor matrices of each iteration. The default is False.
    error_fast : boolean, optional
        If true, use err_fast to compute data fitting error, otherwise, use err. The default is True.
    time_rec : boolean, optional
        If true, return computation time of each iteration. The default is False.

    Returns
    -------
    the CP decomposition, number of iteration and termination criterion. 
    list_fac and list_time are optional.
  """
  N=tl.ndim(tensor) # order of tensor
  norm_tensor=tl.norm(tensor) # norm of tensor
  if time_rec == True : list_time=[]
  if list_factors==True : list_fac=[] # list of factor matrices


  weights=None
  it=0
  if list_factors==True : list_fac.append(copy.deepcopy(factors))
  error=[err(tensor,weights,factors)/norm_tensor]
  while (error[len(error)-1]>tol and it<it_max):
    if time_rec == True : tic=time.time() 
    for n in range(N):
      V=np.ones((rank,rank))
      for i in range(len(factors)):
        if i != n : V=V*tl.dot(tl.transpose(factors[i]),factors[i])
      W=tl.cp_tensor.unfolding_dot_khatri_rao(tensor, (None,factors), n)
      fac, _, _, _ = hals_nnls(tl.transpose(W), V,tl.transpose(factors[n]))
      factors[n]=tl.transpose(fac)
    if list_factors==True : list_fac.append(copy.deepcopy(factors))
    it=it+1
    if (error_fast==False) : error.append(err(tensor,weights,factors)/norm_tensor)
    else : error.append(err_fast(norm_tensor,factors[N-1],V,W)/norm_tensor)
    if time_rec == True : 
      toc=time.time() 
      list_time.append(toc-tic)
  # weights,factors=tl.cp_tensor.cp_normalize((None,factors))
  if list_factors==True and time_rec==True: return(weights,factors,it,error,list_fac,list_time)
  if time_rec==True : return(weights,factors,it,error,list_time)
  if list_factors==True : return(weights,factors,it,error,list_fac)
  return(weights,factors,it,error)