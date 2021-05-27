"""
Need to change herCPRAND to find the graphs in report because it doesn't return exact error for now
Test of param_research
"""
import numpy as np
from src._paramResearch import param_research

def test_paramresearch(i):
    """
    Test of paramresearch
    Need to change plot label in pram_research to run the test.
    Parameters
    ----------
    i : int
        i=1, plot for beta
        i=2, plot for eta
        i=3, plot for gamma

    Returns
    -------
    None.

    """
    
    I=100
    J=100
    K=100
    r=10 # rank
    n_samples=int(10*r*np.log(r)+1) # nb of randomized samples
    nb_rand=10 # nb of random initialization
    
    # beta
    if i ==1 :
        param_research(I,J,K,r,nb_rand,n_samples,exact_err=True)
    # eta
    if i ==2 :
        param_research(I,J,K,r,nb_rand,n_samples,beta=False,eta=True,exact_err=True)
    # gamma
    if i ==3 :
        param_research(I,J,K,r,nb_rand,n_samples,beta=False,gamma=True,exact_err=True)


