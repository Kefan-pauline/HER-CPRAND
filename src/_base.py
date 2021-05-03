"""
Basic functions
"""

import numpy as np
import tensorly as tl
from scipy.optimize import linear_sum_assignment


def random_init_fac(tensor,rank,nn=False) : 
  """
    Random initialization of factor matrices for a given tensor and rank
    random numbers on [0,1)
                       
    Parameters
    ----------
    tensor : tensor
    rank : int

    Returns
    -------
    factors : list of matrices
  """
  factors=[]
  for mode in range(tl.ndim(tensor)):
    factors+=[np.random.random((tl.shape(tensor)[mode], rank))] # random on [0,1)
  return factors


def svd_init_fac(tensor,rank) :
  """
    svd initialization of factor matrices for a given tensor and rank
    
    Parameters
    ----------
    tensor : tensor
    rank : int

    Returns
    -------
    factors : list of matrices

  """
  factors=[]
  for mode in range(tl.ndim(tensor)):
    # unfolding of a given mode
    unfolded=tl.unfold(tensor, mode)
    if rank<=tl.shape(tensor)[mode] : 
      u,s,v=tl.partial_svd(unfolded,n_eigenvecs=rank) # first rank eigenvectors/values (ascendent)
    else : 
      u,s,v=tl.partial_svd(unfolded,n_eigenvecs=tl.shape(tensor)[mode]) 
      # completed by random columns
      u=np.append(u,np.random.random((np.shape(u)[0],rank-tl.shape(tensor)[mode])),axis=1)  
      # sometimes we have singular matrix error for als
    factors+=[u]
  return (factors)


def err_fac(fac,fac_est) :
  """
    factor error computation
    Parameters
    ----------
    fac : list of matrices
        true factor matrices
    fac_est : list of matrices
        factor matrices estimation

    Returns
    -------
    float
        factor error
  """
  # normalize factor matrices
  weights,fac=tl.cp_normalize((None,fac))
  weights_est,fac_est=tl.cp_normalize((None,fac_est))
  err=0
  for i in range(len(fac)):
    # find the corresponding columns of fac and fac_est
    if i==0 : row_ind, col_ind = linear_sum_assignment(-np.abs(np.dot(np.transpose(fac[i]),fac_est[i])))
    err=err+(tl.norm(fac[i]-fac_est[i][:,col_ind])/tl.norm(fac[i]))
  return (err/len(fac))


def err(tensor,weight,factors): 
  """
    calculate exact data fitting error of a tensor and its CP decomposition

    Parameters
    ----------
    tensor : tensor
    weight : vector
        the weight vector of CP decompositiion
    factors : list of matrices
        the factor matrices of CP decomposition

    Returns
    -------
    None.

    """
  t_tilde=tl.cp_to_tensor((weight,factors)) # transform tensor decomposition (kruskal tensor) to tensor
  return(tl.norm(tensor-t_tilde))


def sv_scale_to_100(A):
  """
    scale the singular value of matrix A to 1,..,100

    Parameters
    ----------
    A : matrix

    Returns
    -------
    matrix
        scaled matrix

  """
  u,d,v=np.linalg.svd(A,full_matrices=False)
  scale=9/(d[0]-d[len(d)-1])
  for i in range(len(d)) : d[i]=(d[i]-d[len(d)-1])*scale+1
  d[len(d)-1]=1
  return (u@np.diag(d)@v)

def init_factors(I,J,K,r,noise_level=0.1,scale=False,nn=False,snr=False) :
  """
    Initialize a three way tensor's factor matrices
    
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
    noise_level : float, optional
        noise level. The default is 0.001.
    scale : boolean, optional
        whether to scale the singular value or not. The default is False.
    snr : boolean, optional
        whether return SNR or not

    Returns
    -------
    factors : list of matrices
        factors
    noise : tensor
        noise tensor.

  """
  A=np.random.normal(0, 1, size=(I, r))
  B=np.random.normal(0, 1, size=(J, r))
  C=np.random.normal(0, 1, size=(K, r))
  if nn==True : 
      A=np.abs(A)
      B=np.abs(B)
      C=np.abs(C)
  if (scale==True) :  
    A=sv_scale_to_100(A)
    B=sv_scale_to_100(B)
    C=sv_scale_to_100(C)
  factors=[A,B,C]
  x=tl.cp_to_tensor((None,factors))
  N=np.random.normal(0, 1, size=(I,J,K))
  noise=noise_level*tl.norm(x)/tl.norm(N)*N
  SNR = 10*np.log10(tl.norm(x)**2/tl.norm(noise)**2)
  if snr==True:
      return (factors,noise,SNR)
  else:
      return(factors,noise)


def init_factors_hernn(I,J,K,r,noise_level=0.1,scale=False) :
  """
    Initialize a three way tensor's factor matrices
    
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
    noise_level : float, optional
        noise level. The default is 0.001.
    scale : boolean, optional
        whether to scale the singular value or not. The default is False.

    Returns
    -------
    factors : list of matrices
        factors
    noise : tensor
        noise tensor.

  """
  A=np.random.random((I, r))
  B=np.random.random((J, r))
  C=np.random.random((K, r))

  if (scale==True) :  
    A=sv_scale_to_100(A)
    B=sv_scale_to_100(B)
    C=sv_scale_to_100(C)
  factors=[A,B,C]
  N=noise_level*np.random.normal(0, 1, size=(I,J,K))
  return (factors,N)

def score(fac,fac_est):
  """
    standard score metric

    Parameters
    ----------
    fac : list of matrices
        true factors
    fac_est : list of matrices
        factors returned by CP decomposition

    Returns
    -------
    double
        Score.

  """
  weights,fac=tl.cp_normalize((None,fac))
  weights_est,fac_est=tl.cp_normalize((None,fac_est))
  score=0
  # find the corresponding columns of fac and fac_est
  row_ind, col_ind = linear_sum_assignment(-np.abs(np.dot(np.transpose(fac[0]),fac_est[0])))
  for k in range(len(fac)):
    fac_est[k]=fac_est[k][:,col_ind]
  for i in range(fac[0].shape[1]): # R
    temp=1
    for j in range(len(fac)): # N
      temp=temp*np.dot(fac[j][:,i],fac_est[j][:,i])/(tl.norm(fac[j][:,i])*tl.norm(fac_est[j][:,i]))
      # np.transpose(fac[j][:,i] ? 
    score=score+temp
  return (score/fac[0].shape[1])