
"""
exact error doesn't work, need to change her cprand function 
"""
import numpy as np
import tensorly as tl
from src._base import random_init_fac
from src._comparaison import init_factors
from src._hercprand import her_CPRAND
import copy


def test_nbrestart(i):
    """
    Compute the nb restart for 4 cases : 
    1 - simple case with exact error
    2 - simple case with estimated error
    3 - complicated case with exact error 
    4 - complicated case with estimated error
    
    In each case, we simulate nb_rand noised I*J*K rank r random tensors.
    For each tensor, we do 5 factor matrices initializations.

    Parameters
    ----------
    i : int
        choice of case.

    Returns
    -------
    float
        mean value of nbrestart

    """
    I=50
    J=50
    K=50
    r=10 # rank
    n_samples=int(10*r*np.log(r)+1) # nb of randomized samples
    nb_rand=10 # nb of random initialization

    list_pct=[]
    for i in range(nb_rand) : 
        # Random initialization of a noised cp_tensor
        np.random.seed(i)
        if i==1 or i==2:
            A,B,C,noise=init_factors(I,J,K,r,scale=False)
        else :
            A,B,C,noise=init_factors(I,J,K,r,scale=True)
        t=tl.cp_to_tensor((None,[A,B,C]))+noise
        for j in range(5):
            factors=random_init_fac(t,r)
            if i==1 :
                # simple case with exact error
                weights,factors,it,error,_,pct=her_CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=500,err_it_max=100) 
                list_pct.append(pct)
            if i==2:
                # simple case with estimated error
                weights,factors,it,_,error,pct=her_CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=False,it_max=500,err_it_max=100)
                list_pct.append(pct)
            if i==3:
                # complicated case with exact error 
                weights,factors,it,error,_,pct=her_CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=True,it_max=500,err_it_max=100) 
                list_pct.append(pct)
            if i==4:
                # complicated case with estimated error
                weights,factors,it,_,error,pct=her_CPRAND(t,r,n_samples,factors=copy.deepcopy(factors),exact_err=False,it_max=500,err_it_max=400)
                list_pct.append(pct)
    return(np.mean(list_pct))
    
    
