
import numpy as np
from src._comparaison import comparaison,compar_time,comparison,compar_filter,nn_comparison,nn_err_it


def test_comparaison(i): 
    """
    Test of comparaison
    Need to change plot x,y label and title in comparaison to run the test.

    Parameters
    ----------
    i : int
        i=1, plot data fitting error for simple case.
        i=2, plot factor error for simple case.
        i=3, plot data fitting error for complicated case.
        i=4, plot factor error for complicated case.
        
    Returns
    -------
    None.

    """
    I=100
    J=100
    K=100
    r=10 # rank
    n_samples=int(10*r*np.log(r)+1) # nb of randomized samples
    nb_rand=5 # nb of random initialization
    if i ==1 :
        # simple case data fitting error
        comparaison(I,J,K,r,nb_rand,n_samples)
    if i ==2 : 
        # simple case factors error
        comparaison(I,J,K,r,nb_rand,n_samples,False,list_factors=True)
    if i == 3 :
        # complicated case data fitting error
        comparaison(I,J,K,r,nb_rand,n_samples,scale=True)
    if i==4 :
        # complicated case factors error
        comparaison(I,J,K,r,nb_rand,n_samples,False,list_factors=True,scale=True)
        
def test_compar_time(i):
    """
    Test of compar_time
    Need to change plot x,y label and title in compar_time to run the test.

    Parameters
    ----------
    i : int
        i=1, plot data fitting error for simple case.
        i=2, plot factor error for simple case.
        i=3, plot data fitting error for complicated case.
        i=4, plot factor error for complicated case.
        
    Returns
    -------
    None.
    
    """
    I=50
    J=50
    K=50
    r=10 # rank
    n_samples=int(10*r*np.log(r)+1) # nb of randomized samples
    nb_rand=10 # nb of random initialization
    if i==1:
        # simple case data fitting error 
        compar_time(I,J,K,r,nb_rand,n_samples)
    if i==2:
        # simple case factors error 
        compar_time(I,J,K,r,nb_rand,n_samples,list_factors=True)
    if i==3:
        # complicated case data fitting error 
        compar_time(I,J,K,r,nb_rand,n_samples,scale=True)
    if i==4:
        # complicated case factors error 
        compar_time(I,J,K,r,nb_rand,n_samples,list_factors=True,scale=True)

def test_comparison():
    """
    test of comparison in a specific situation

    Returns
    -------
    None.

    """
    I=100
    J=100
    K=100
    r=10
    n_samples=int(10*r*np.log(r)+1)
    nb_rand=10
    n_samples_err=400
    comparison(I,J,K,r,nb_rand,n_samples,n_samples_err,exact_err=False,scale=True)
    
def test_compar_filter():
    """
    test of comparison in a specific situation

    Returns
    -------
    None.

    """
    I=100
    J=100
    K=100
    r=10
    n_samples=int(10*r*np.log(r)+1)
    nb_rand=10
    n_samples_err=400
    compar_filter(I,J,K,r,nb_rand,n_samples,n_samples_err,exact_err=False,scale=True)


def test_nn_comparison():
    """
    test of nn_comparison in a specific situation

    Returns
    -------
    None.

    """
    I=100
    J=100
    K=100
    r=10
    n_samples=int(10*r*np.log(r)+1)
    nb_rand=10
    n_samples_err=400
    noise_level=0.1
    nn_comparison(I,J,K,r,nb_rand,n_samples,n_samples_err,noise_level=noise_level,exact_err=False,scale=True)
    
def test_nn_err_it():
    """
    test of nn_comparison in a specific situation

    Returns
    -------
    None.

    """
    I=100
    J=100
    K=100
    r=10
    n_samples=int(10*r*np.log(r)+1)
    nb_rand=5
    n_samples_err=400
    noise_level=0.3
    nn_err_it(I,J,K,r,nb_rand,n_samples,n_samples_err,exact_err=False,list_factors=False,scale=True,noise_level=noise_level)