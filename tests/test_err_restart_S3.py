import numpy as np
import tensorly as tl
from src._cprand import CPRAND,err_rand,err_rand_fast
from src._base import init_factors,random_init_fac,svd_init_fac,err
from tensorly.decomposition import sample_khatri_rao
import matplotlib.pyplot as plt
import copy
import time

def test_cprand_size():
    """
    For a noised I*J*K rank r random tensor, with random initialized factor matrices
    plot err_fast and exact err for complicated case, with n_sampled changed.
    """
    I=50
    J=50
    K=50
    r=10 # rank
    A,B,C,noise=init_factors(I,J,K,r,True)
    fac_true=[A,B,C]
    t=tl.cp_to_tensor((None,fac_true))+noise
    print(tl.norm(t))
    factors=random_init_fac(t,r)
    
    # n_samples used for CPRAND
    n_samples=int(10*r*np.log(r)+1) # nb of randomized samples
    weights2,factors2,it2,error2,error_es2=CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=500,err_it_max=400)
    plt.figure(0)
    plt.plot(range(len(error2)),error2,'b-',label="exact n_samples")
    plt.plot(range(len(error_es2)),error_es2,'r-',label="err fast n_samples")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('cprand for complicated case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    
    # 500*n_samples used for CPRAND
    weights2,factors2,it2,error2,error_es2=CPRAND(t,r,500*n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=500,err_it_max=400)
    plt.figure(0)
    plt.plot(range(len(error2)),error2,'b--',label="exact 500*n_samples")
    plt.plot(range(len(error_es2)),error_es2,'r--',label="err fast 500*n_samples")
    plt.yscale('log')
    plt.legend(loc='best')

def her_CPRAND_r(tensor,rank,n_samples,factors=None,exact_err=True,it_max=100,err_it_max=20,tol=1e-7,beta=0.1,eta=3,gamma=1.01,gamma_bar=1.005,list_factors=False,time_rec=False):
    """
      modified herCPRAND to plot restart criterion
      retrun a list of F_hat_bf
      
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
    #
    # added
    #
    l_F_hat_bf=[]
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
      #
      # added
      #
      l_F_hat_bf.append(F_hat_bf/norm_tensor)

   
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
    #
    # added
    #
    return(weights,factors,it,error_ex,error,l_F_hat_bf,cpt/it)

def test_plot_restart(c):
    """
    For a noised I*J*K rank r random tensor, with 4 random initialized factor matrices
    For each initialization, plot F_hat_bf, F_hat_new and exact F for complicated case.
    Need to change True/ False in init_factors to display simple / complicated case.
    
    Parameters
    ----------
    c : int
        c=1,
            F_hat_new = F_hat_it for it=0,...,nb_it
            F_hat_bf = F_hat_it for it=0,0,...,nb_it-1
            F_exact = F_it for it=0,...,nb_it
            For iteration k>0, F_hat_new[k]=F_hat_k, F_hat_bf[k]=F_hat_k-1, F_exact=F_k
        c=2,
            F_hat_new = F_hat_it for it=0,...,nb_it
            F_hat_bf = F_hat_it for it=0,...,nb_it-1
            F_exact = F_it for it=0,...,nb_it
    """
    I=50
    J=50
    K=50
    r=10 # rank
    A,B,C,noise=init_factors(I,J,K,r,False)
    fac_true=[A,B,C]
    t=tl.cp_to_tensor((None,fac_true))+noise
    print("norm_tensor",tl.norm(t))
    n_samples=int(10*r*np.log(r)+1)
    for i in range(4):
        factors=random_init_fac(t,r)
        w,f,it,error_ex,error,F_hat_bf,pct=her_CPRAND_r(t, r, n_samples,factors,it_max=500,err_it_max=300)
        print("Restart pct",pct)

        if c==1 :
            F_hat_bf.insert(0, error[0])
        plt.figure(i)
        plt.plot(range(len(error)),error,'r-',label="F_hat_new/norm_tensor")
        plt.plot(range(len(error_ex)),error_ex,'y-',label="F_exact/norm_tensor")
        plt.plot(range(len(F_hat_bf)),F_hat_bf,'g--',label="F_hat_bf/norm_tensor")
        plt.xlabel('it')
        plt.yscale('log')
        plt.title('Restart criterion')
        plt.ylabel('terminaison criterion')
        plt.legend(loc='best')
    

def her_CPRAND_e(tensor,rank,n_samples,factors=None,exact_err=True,it_max=100,err_it_max=20,tol=1e-7,beta=0.1,eta=3,gamma=1.01,gamma_bar=1.005,list_factors=False,time_rec=False):
    """
      modified herCPRAND to use another err computation
      
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

def test_herCPRAND_e():
    I=50
    J=50
    K=50
    r=10 # rank
    n_samples=int(10*r*np.log(r)+1) # nb of randomized samples
    A,B,C,noise=init_factors(I,J,K,r,True)
    fac_true=[A,B,C]
    t=tl.cp_to_tensor((None,fac_true))+noise
    factors=random_init_fac(t,r)
    weights2,factors2,it2,error2,error_es2,cpt2=her_CPRAND_e(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=500,err_it_max=400)
    plt.figure(0)
    plt.plot(range(len(error2)),error2,'b-',label="exact")
    plt.plot(range(len(error_es2)),error_es2,'r--',label="err fast")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand for complicated case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')
    plt.figure(1)
    A,B,C,noise=init_factors(I,J,K,r,False)
    fac_true=[A,B,C]
    t=tl.cp_to_tensor((None,fac_true))+noise
    factors=random_init_fac(t,r)
    weights2,factors2,it2,error2,error_es2,cpt2=her_CPRAND_e(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=500,err_it_max=400)
    plt.plot(range(len(error2)),error2,'b-',label="exact")
    plt.plot(range(len(error_es2)),error_es2,'r--',label="err fast")
    plt.xlabel('it')
    plt.yscale('log')
    plt.title('hercprand for simple case')
    plt.ylabel('terminaison criterion')
    plt.legend(loc='best')