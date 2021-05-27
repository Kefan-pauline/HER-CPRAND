"""
herCPRAND methode for CP decomposition 
return also exact error
"""

import tensorly as tl
import numpy as np
import copy 
import time
from src._base import svd_init_fac,err
from tensorly.decomposition import sample_khatri_rao
from src._cprand import err_rand
from tensorly.tenalg.proximal import hals_nnls


def her_CPRAND(tensor,rank,n_samples,n_samples_err=400,factors=None,exact_err=False,it_max=100,err_it_max=20,tol=1e-7,beta=0.1,eta=3,gamma=1.01,gamma_bar=1.005,list_factors=False,time_rec=False,filter=10):
    """
     herCPRAND for CP-decomposition
     same err sample taking mean value of last filter values
     
     Parameters
    ----------
    tensor : tensor
    rank : int
    n_samples : int
        sample size
    n_samples_err : int, optional
        sample size used for error estimation. The default is 400.
    factors : list of matrices, optional
        an initial factor matrices. The default is None.
    exact_err : boolean, optional
        whether use err or err_rand_fast for terminaison criterion. The default is False.
    it_max : int, optional
        maximal number of iteration. The default is 100.
    err_it_max : int, optional
        maximal of iteration if terminaison critirion is not improved. The default is 20.
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
    time_rec : boolean, optional
        If true, return computation time of each iteration. The default is False.
    filter : int, optional
        The filter size used for the mean value

    Returns
    -------
    the CP decomposition, number of iteration, error and restart pourcentage. 
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
    list_F_hat_bf=[]
  
    weights=None
    it=0
    err_it=0
    cpt=0
    ########################################
    ######### error initialization #########
    ########################################
    if(exact_err==True):
        F_hat_bf = err(tensor,weights,factors)
    else :
        F_hat_bf,ind_bf=err_rand(tensor,None,factors,n_samples_err)
    list_F_hat_bf.append(F_hat_bf)
    
    rng = tl.check_random_state(None)
    error=[F_hat_bf/norm_tensor]
    min_err=error[len(error)-1]
  
    while (min_err>tol and it<it_max and err_it<err_it_max): 
      if time_rec==True :tic=time.time()
      for n in range(N):
        Zs,indices=sample_khatri_rao(factors_hat,n_samples,skip_matrix=n,random_state=rng)
        indices_list = [i.tolist() for i in indices]
        indices_list.insert(n, slice(None, None, None))
        indices_list = tuple(indices_list)
        V=tl.dot(tl.transpose(Zs),Zs)
        if (n==0) :sampled_unfolding = tensor[indices_list] 
        else : sampled_unfolding = tl.transpose(tensor[indices_list])
        W=tl.dot(sampled_unfolding,Zs)
        factor_bf=factors[n]
        # update
        factors[n] = tl.transpose(tl.solve(V,tl.transpose(W))) # solve needs a squared full rank matrix, if rank>nb_sampls ok
        # extrapolate
        factors_hat[n]=factors[n]+beta*(factors[n]-factor_bf)
      ########################################
      #########      error update    #########
      ########################################
      if(exact_err==False):
          
          F_hat_new,_= err_rand(tensor,weights,factors,n_samples_err,indices_list=ind_bf)
      else : 
          F_hat_new = err(tensor,weights,factors)
      list_F_hat_bf.append(F_hat_new)

   
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
      if (exact_err==False):
          if it<filter:
              F_hat_bf=np.mean(list_F_hat_bf)
          else:
              F_hat_bf=np.mean(list_F_hat_bf[(len(list_F_hat_bf)-filter):(len(list_F_hat_bf)-1)])
      else : 
          F_hat_bf=F_hat_new



      if list_factors==True : list_fac.append(copy.deepcopy(factors))
      error.append(F_hat_new/norm_tensor)
      if (error[len(error)-1]<min_err) : min_err=error[len(error)-1] # err update
      else : err_it=err_it+1
      if time_rec==True : 
        toc=time.time()
        list_time.append(toc-tic)
    if list_factors==True and time_rec==True: return(weights,factors,it,error,cpt/it,list_fac,list_time)
    if list_factors==True : return(weights,factors,it,error,cpt/it,list_fac)
    if time_rec==True : return(weights,factors,it,error,cpt/it,list_time)
    return(weights,factors,it,error,cpt/it)



def nn_her_CPRAND(tensor,rank,n_samples,n_samples_err=400,factors=None,exact_err=False,it_max=100,err_it_max=20,tol=1e-7,beta=0.1,eta=3,gamma=1.01,gamma_bar=1.005,list_factors=False,time_rec=False,filter=10):
    """
     herCPRAND for nonnegative CP-decomposition
     same err sample taking mean value of last filter values
     
     Parameters
    ----------
    tensor : tensor
    rank : int
    n_samples : int
        sample size
    n_samples_err : int, optional
        sample size used for error estimation. The default is 400.
    factors : list of matrices, optional
        an initial factor matrices. The default is None.
    exact_err : boolean, optional
        whether use err or err_rand_fast for terminaison criterion. The default is False.
    it_max : int, optional
        maximal number of iteration. The default is 100.
    err_it_max : int, optional
        maximal of iteration if terminaison critirion is not improved. The default is 20.
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
    time_rec : boolean, optional
        If true, return computation time of each iteration. The default is False.
    filter : int, optional
        The filter size used for the mean value

    Returns
    -------
    the CP decomposition, number of iteration, error and restart pourcentage. 
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
    list_F_hat_bf=[]
  
    weights=None
    it=0
    err_it=0
    cpt=0
    ########################################
    ######### error initialization #########
    ########################################
    if(exact_err==True):
        F_hat_bf = err(tensor,weights,factors)
    else :
        F_hat_bf,ind_bf=err_rand(tensor,None,factors,n_samples_err)
    list_F_hat_bf.append(F_hat_bf)
    
    rng = tl.check_random_state(None)
    error=[F_hat_bf/norm_tensor]
    min_err=error[len(error)-1]
  
    while (min_err>tol and it<it_max and err_it<err_it_max): 
      if time_rec==True :tic=time.time()
      for n in range(N):
        Zs,indices=sample_khatri_rao(factors_hat,n_samples,skip_matrix=n,random_state=rng)
        indices_list = [i.tolist() for i in indices]
        indices_list.insert(n, slice(None, None, None))
        indices_list = tuple(indices_list)
        V=tl.dot(tl.transpose(Zs),Zs)
        if (n==0) :sampled_unfolding = tensor[indices_list] 
        else : sampled_unfolding = tl.transpose(tensor[indices_list])
        W=tl.dot(sampled_unfolding,Zs)
        factor_bf=factors[n]
        # update
        fac, _, _, _ = hals_nnls(tl.transpose(W), V,tl.transpose(factors[n]))
        factors[n]=tl.transpose(fac) # solve needs a squared full rank matrix, if rank>nb_sampls ok 
        # extrapolate
        factors_hat[n]=tl.clip(factors[n]+beta*(factors[n]-factor_bf),a_min=0.0)
      ########################################
      #########      error update    #########
      ########################################
      if(exact_err==False):
          
          F_hat_new,_= err_rand(tensor,weights,factors,n_samples_err,indices_list=ind_bf)
      else : 
          F_hat_new = err(tensor,weights,factors)
      list_F_hat_bf.append(F_hat_new)

   
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
      if (exact_err==False):
          if it<filter:
              F_hat_bf=np.mean(list_F_hat_bf)
          else:
              F_hat_bf=np.mean(list_F_hat_bf[(len(list_F_hat_bf)-filter):(len(list_F_hat_bf)-1)])
      else : 
          F_hat_bf=F_hat_new



      if list_factors==True : list_fac.append(copy.deepcopy(factors))
      error.append(F_hat_new/norm_tensor)
      if (error[len(error)-1]<min_err) : min_err=error[len(error)-1] # err update
      else : err_it=err_it+1
      if time_rec==True : 
        toc=time.time()
        list_time.append(toc-tic)
    if list_factors==True and time_rec==True: return(weights,factors,it,error,cpt/it,list_fac,list_time)
    if list_factors==True : return(weights,factors,it,error,cpt/it,list_fac)
    if time_rec==True : return(weights,factors,it,error,cpt/it,list_time)
    return(weights,factors,it,error,cpt/it)




def her_CPRAND1(tensor,rank,n_samples,factors=None,exact_err=True,it_max=100,err_it_max=20,tol=1e-7,beta=0.1,eta=3,gamma=1.01,gamma_bar=1.005,list_factors=False,time_rec=False):
    """
      same err sample without taking mean value
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
    #
    # added
    #
    n_samples_err=400 # assuming that miu max = 1
    ########################################
    ######### error initialization #########
    ########################################
    #
    # added
    #
    F_hat_bf,ind_bf=err_rand(tensor,None,factors,n_samples_err)
    F_hat_bf_ex=err(tensor,None,factors) # exact cost
    rng = tl.check_random_state(None)
    error=[F_hat_bf/norm_tensor]
    error_ex=[F_hat_bf_ex/norm_tensor]
    min_err=error[len(error)-1]
  
    while (min_err>tol and it<it_max and err_it<err_it_max): 
      if time_rec==True :tic=time.time()
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
      #
      # added (F in fact, not F hat)
      #
      F_hat_new,_= err_rand(tensor,weights,factors,n_samples_err,indices_list=ind_bf)

   
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
      F_hat_bf=F_hat_new
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

def her_CPRAND2(tensor,rank,n_samples,factors=None,exact_err=True,it_max=100,err_it_max=20,tol=1e-7,beta=0.1,eta=3,gamma=1.01,gamma_bar=1.005,list_factors=False,time_rec=False):
    """
      different err sample without taking mean value
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

    n_samples_err=400 # assuming that miu max = 1
    ########################################
    ######### error initialization #########
    ########################################
    F_hat_bf,ind_bf=err_rand(tensor,None,factors,n_samples_err)
    F_hat_bf_ex=err(tensor,None,factors) # exact cost
    rng = tl.random.check_random_state(None)
    error=[F_hat_bf/norm_tensor]
    error_ex=[F_hat_bf_ex/norm_tensor]
    min_err=error[len(error)-1]
  
    while (min_err>tol and it<it_max and err_it<err_it_max): 
      if time_rec==True :tic=time.time()
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
      F_hat_new,_= err_rand(tensor,weights,factors,n_samples_err,indices_list=ind_bf)
      #
      # added
      # a new sample
      F_hat_bf_new,ind_bf= err_rand(tensor,weights,factors,n_samples_err)

   
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
      # changed
      F_hat_bf=F_hat_bf_new
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


def her_CPRAND3(tensor,rank,n_samples,n_samples_err=400,factors=None,exact_err=True,it_max=100,err_it_max=20,tol=1e-7,beta=0.1,eta=3,gamma=1.01,gamma_bar=1.005,list_factors=False,time_rec=False,filter=10):
    """
      same err sample taking mean value (selected version)
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
    #
    # added
    #
    list_F_hat_bf=[]
  
    weights=None
    it=0
    err_it=0
    cpt=0
    ########################################
    ######### error initialization #########
    ########################################
    F_hat_bf,ind_bf=err_rand(tensor,None,factors,n_samples_err)
    #
    # added
    #
    list_F_hat_bf.append(F_hat_bf)
    
    F_hat_bf_ex=err(tensor,None,factors) # exact cost
    rng = tl.check_random_state(None)
    error=[F_hat_bf/norm_tensor]
    error_ex=[F_hat_bf_ex/norm_tensor]
    min_err=error[len(error)-1]
  
    while (min_err>tol and it<it_max and err_it<err_it_max): 
      if time_rec==True :tic=time.time()
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
      F_hat_new,_= err_rand(tensor,weights,factors,n_samples_err,indices_list=ind_bf)
      #
      # added
      #
      list_F_hat_bf.append(F_hat_new)

   
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
      #
      # changed
      #
      if it<filter:
          F_hat_bf=np.mean(list_F_hat_bf)
      else:
          F_hat_bf=np.mean(list_F_hat_bf[(len(list_F_hat_bf)-filter):(len(list_F_hat_bf)-1)])



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

def her_CPRAND4(tensor,rank,n_samples,factors=None,exact_err=True,it_max=100,err_it_max=20,tol=1e-7,beta=0.1,eta=3,gamma=1.01,gamma_bar=1.005,list_factors=False,time_rec=False):
    """
      different err sample taking mean value, F_hat_new is evaluated with the same samples as F_hat_bf
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
    list_F_hat_bf=[]
  
    weights=None
    it=0
    err_it=0
    cpt=0
    n_samples_err=400 # assuming that miu max = 1
    ########################################
    ######### error initialization #########
    ########################################
    F_hat_bf,ind_bf=err_rand(tensor,None,factors,n_samples_err)
    list_F_hat_bf.append(F_hat_bf)
    
    F_hat_bf_ex=err(tensor,None,factors) # exact cost
    rng = tl.random.check_random_state(None)
    error=[F_hat_bf/norm_tensor]
    error_ex=[F_hat_bf_ex/norm_tensor]
    min_err=error[len(error)-1]
  
    while (min_err>tol and it<it_max and err_it<err_it_max): 
      if time_rec==True :tic=time.time()
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
      F_hat_new,_= err_rand(tensor,weights,factors,n_samples_err,indices_list=ind_bf)
      #
      # added
      # a new sample
      F_hat_bf_new,ind_bf= err_rand(tensor,weights,factors,n_samples_err)
      list_F_hat_bf.append(F_hat_bf_new)

   
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
      if it<10 :
          F_hat_bf=np.mean(list_F_hat_bf)
      else:
          F_hat_bf=np.mean(list_F_hat_bf[(len(list_F_hat_bf)-10):(len(list_F_hat_bf)-1)])

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

def her_CPRAND5(tensor,rank,n_samples,factors=None,exact_err=True,it_max=100,err_it_max=20,tol=1e-7,beta=0.1,eta=3,gamma=1.01,gamma_bar=1.005,list_factors=False,time_rec=False):
    """
      different err sample taking mean value
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
    list_F_hat_bf=[]
  
    weights=None
    it=0
    err_it=0
    cpt=0
    n_samples_err=400 # assuming that miu max = 1
    ########################################
    ######### error initialization #########
    ########################################
    F_hat_bf,ind_bf=err_rand(tensor,None,factors,n_samples_err)
    list_F_hat_bf.append(F_hat_bf)
    
    F_hat_bf_ex=err(tensor,None,factors) # exact cost
    rng = tl.check_random_state(None)
    error=[F_hat_bf/norm_tensor]
    error_ex=[F_hat_bf_ex/norm_tensor]
    min_err=error[len(error)-1]
  
    while (min_err>tol and it<it_max and err_it<err_it_max): 
      if time_rec==True :tic=time.time()
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
      F_hat_new,_= err_rand(tensor,weights,factors,n_samples_err,indices_list=ind_bf)
      #
      # added
      # a new sample
      ind_bf = [np.random.choice(tl.shape(m)[0], n_samples_err) for m in factors]   
      ind_bf = [i.tolist() for i in ind_bf]
      ind_bf = tuple(ind_bf)
      list_F_hat_bf.append(F_hat_new)

   
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

      if it<10 :
          F_hat_bf=np.mean(list_F_hat_bf)
      else:
          F_hat_bf=np.mean(list_F_hat_bf[(len(list_F_hat_bf)-10):(len(list_F_hat_bf)-1)])


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