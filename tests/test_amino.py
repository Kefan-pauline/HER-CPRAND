import numpy as np
from src._amino import test_amino,amino_plot,err_time



def test_test_amino(): 
    r=3
    n_samples=int(10*r*np.log(r)+1)
    n_samples_err=100
    test_amino(r,n_samples,n_samples_err)
    
    
def test_aminoplot():
    r=3
    n_samples=int(10*r*np.log(r)+1)
    n_samples_err=400
    
    amino_plot(r,n_samples,n_samples_err,exact_err=False,tol=0.01)
    
def test_err_time():
    r=3
    n_samples=int(10*r*np.log(r)+1)
    #n_samples=10
    n_samples_err=400
    err_time(r,n_samples,n_samples_err,exact_err=False,tol=0.00002,list_factors=False)
