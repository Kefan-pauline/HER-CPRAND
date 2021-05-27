
import numpy as np
from src._comparaison import comparison,compar_filter,nn_comparison,nn_err_it


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
    test of compar_filter in a specific situation

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
    
