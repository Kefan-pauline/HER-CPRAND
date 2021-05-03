from src._speedup import speedup,speedup_old
import numpy as np

def test_speedup():
    r=10
    list_N=[60,70,80,90,100]

    list_S=[int(10*r*np.log(r)+1),500, 700,1000]
    list_P=[400,400,400,400]

    noise_level=0.1
    scale=True
    nn=True
    tol=0.11 # 10% error 
    speedup_old(list_N,r,list_S,list_P,tol=tol,noise_level=noise_level,scale=scale,nn=nn,nb_tensors=5)



