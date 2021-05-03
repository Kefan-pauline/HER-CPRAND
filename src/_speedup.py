import tensorly as tl
import numpy as np
from src._als import als,nn_als
from src._herals import her_Als,nn_her_Als
from src._cprand import CPRAND, nn_CPRAND
from src._hercprand import her_CPRAND,nn_her_CPRAND
from src._base import init_factors,random_init_fac
import copy
import matplotlib.pyplot as plt

def speedup_old(list_N,r,list_S,list_P,tol,noise_level=0.1,scale=True,nn=False,nb_tensors=5):
    """
    Calculate the speed up of her CPRAND vs ALS, her ALS and CPRAND

    Parameters
    ----------
    list_N : list
        list of dimensions (in the increasing order)
    r : int
        rank of the tensor
    list_S : list
        list of the sample sizes, same length as list_P
    list_P : list
        list of the err sample sizes, same length as list_P
    tol : double
        tolerance for the 4 algorithms
    noise_level : float, optional
        noise_level of the tensor. The default is 0.1.
    scale : boolean, optional
        whether to scale the condition number of factors or not. The default is True.
    nn : boolean, optional
        use nn methods or not. The default is False.

    Returns
    -------
    None.

    """
    vsals = np.zeros((len(list_N),len(list_S)))
    vsherals = np.zeros((len(list_N),len(list_S)))
    vscprand = np.zeros((len(list_N),len(list_S)))
    for i in range(len(list_N)) :
        time_als = 0
        time_herals = 0
        time_hercprand = np.zeros(len(list_S))
        time_cprand = np.zeros(len(list_S))
        for k in range(nb_tensors):
            fac_true,noise = init_factors(list_N[i], list_N[i], list_N[i], r,noise_level=noise_level,scale=scale,nn=nn)
            t=tl.cp_to_tensor((None,fac_true))+noise
            if k==0 :
                factors=random_init_fac(t,r)
            if nn==False :
                #weights2,factors2,it2,error2,time2=als(t,r,factors=copy.deepcopy(factors),it_max=10000,tol=tol,time_rec=True) 
                weights1,factors1,it1,error1,cpt1,time1=her_Als(t,r,factors=copy.deepcopy(factors),it_max=10000,tol=tol,time_rec=True)  
        
            else : 
                weights2,factors2,it2,error2,time2=nn_als(t,r,factors=copy.deepcopy(factors),it_max=10000,tol=tol,time_rec=True) 
                weights1,factors1,it1,error1,cpt1,time1=nn_her_Als(t,r,factors=copy.deepcopy(factors),it_max=10000,tol=tol,time_rec=True)  
                
            time_als += np.cumsum(time2)[it2-1]
            time_herals += np.cumsum(time1)[it1-1]
            for s in range(len(list_S)):
                if(nn==False):
                    weights3,factors3,it3,error3,time3=CPRAND(t,r,list_S[s],list_P[s],factors=copy.deepcopy(factors),exact_err=False,it_max=10000,err_it_max=10000,tol=tol,time_rec=True)
                    weights4,factors4,it4,error4,cpt4,time4=her_CPRAND(t,r,list_S[s],list_P[s],factors=copy.deepcopy(factors),exact_err=False,it_max=10000,err_it_max=10000,tol=tol,time_rec=True) 
                else :
                    weights3,factors3,it3,error3,time3=nn_CPRAND(t,r,list_S[s],list_P[s],factors=copy.deepcopy(factors),exact_err=False,it_max=10000,err_it_max=10000,tol=tol,time_rec=True)
                    weights4,factors4,it4,error4,cpt4,time4=nn_her_CPRAND(t,r,list_S[s],list_P[s],factors=copy.deepcopy(factors),exact_err=False,it_max=10000,err_it_max=10000,tol=tol,time_rec=True) 
                time_hercprand[s] += np.cumsum(time4)[it4-1]
                time_cprand[s] =+ np.cumsum(time3)[it3-1]
                
        # print("ALS",time_als)
        # print("Herals",time_herals)
        # for j in range(len(list_S)):
        #      print("cprand",time_cprand[j])
        #      print("Hercprand",time_hercprand[j])
        vsals[i,:] = time_als / copy.deepcopy(time_hercprand)
        vsherals[i,:] =time_herals/copy.deepcopy(time_hercprand)
        vscprand[i,:] =copy.deepcopy(time_cprand)/copy.deepcopy(time_hercprand)
        

    # plot
    plt.figure(0)
    for s in range(len(list_S)):
        legend = "S = " + str(list_S[s]) +" , P = " + str(list_P[s])
        plt.plot(list_N, vsals[:,s],label=legend)
    plt.axhline(y = 1,  color = 'k',linestyle = '--',label="speed up = 1")
    plt.xlabel('N')
    plt.ylabel('Speed up factor')
    plt.legend(loc='best')
    plt.title('Speed up vs als')
        
    plt.figure(1)
    for s in range(len(list_S)):
        legend = "S = " + str(list_S[s]) +" , P = " + str(list_P[s])
        plt.plot(list_N,vsherals[:,s],label=legend)
    plt.axhline(y = 1, color = 'k',linestyle = '--',label="speed up = 1")
    plt.xlabel('N')
    plt.ylabel('Speed up factor')
    plt.legend(loc='best')
    plt.title('Speed up vs herals')
                
    plt.figure(2)
    for s in range(len(list_S)):
        legend = "S = " + str(list_S[s]) +" , P = " + str(list_P[s])
        plt.plot(list_N,vscprand[:,s],label=legend)
    plt.axhline(y = 1, color = 'k',linestyle = '--',label="speed up = 1")
    plt.xlabel('N')
    plt.ylabel('Speed up factor')
    plt.legend(loc='best')
    plt.title('Speed up vs cprand')



def speedup(list_N,r,list_S,list_P,tol,noise_level=0.1,scale=True,nn=False,nb_tensors=5):
    """
    Calculate the speed up of her CPRAND vs ALS, her ALS and CPRAND

    Parameters
    ----------
    list_N : list
        list of dimensions (in the increasing order)
    r : int
        rank of the tensor
    list_S : list
        list of the sample sizes, same length as list_P
    list_P : list
        list of the err sample sizes, same length as list_P
    tol : double
        tolerance for the 4 algorithms
    noise_level : float, optional
        noise_level of the tensor. The default is 0.1.
    scale : boolean, optional
        whether to scale the condition number of factors or not. The default is True.
    nn : boolean, optional
        use nn methods or not. The default is False.

    Returns
    -------
    None.

    """

    vsals = np.zeros((len(list_N),len(list_S)))
    seals = np.zeros((len(list_N),len(list_S)))
    vsherals = np.zeros((len(list_N),len(list_S)))
    seherals = np.zeros((len(list_N),len(list_S)))
    vscprand = np.zeros((len(list_N),len(list_S)))
    secprand = np.zeros((len(list_N),len(list_S)))
    for i in range(len(list_N)) :
        time_als = np.zeros(nb_tensors)
        time_herals = np.zeros(nb_tensors)
        time_hercprand = np.zeros((nb_tensors,len(list_S)))
        time_cprand = np.zeros((nb_tensors,len(list_S)))
        for k in range(nb_tensors):
            fac_true,noise = init_factors(list_N[i], list_N[i], list_N[i], r,noise_level=noise_level,scale=scale,nn=nn)
            t=tl.cp_to_tensor((None,fac_true))+noise
            if k==0 :
                factors=random_init_fac(t,r)
            if nn==False :
                weights2,factors2,it2,error2,time2=als(t,r,factors=copy.deepcopy(factors),it_max=10000,tol=tol,time_rec=True) 
                weights1,factors1,it1,error1,cpt1,time1=her_Als(t,r,factors=copy.deepcopy(factors),it_max=10000,tol=tol,time_rec=True)  
        
            else : 
                weights2,factors2,it2,error2,time2=nn_als(t,r,factors=copy.deepcopy(factors),it_max=10000,tol=tol,time_rec=True) 
                weights1,factors1,it1,error1,cpt1,time1=nn_her_Als(t,r,factors=copy.deepcopy(factors),it_max=10000,tol=tol,time_rec=True)  
                
            time_als[k] = np.cumsum(time2)[it2-1]
            time_herals[k] = np.cumsum(time1)[it1-1]
            for s in range(len(list_S)):
                if(nn==False):
                    weights3,factors3,it3,error3,time3=CPRAND(t,r,list_S[s],list_P[s],factors=copy.deepcopy(factors),exact_err=False,it_max=10000,err_it_max=10000,tol=tol,time_rec=True)
                    weights4,factors4,it4,error4,cpt4,time4=her_CPRAND(t,r,list_S[s],list_P[s],factors=copy.deepcopy(factors),exact_err=False,it_max=10000,err_it_max=10000,tol=tol,time_rec=True) 
                else :
                    weights3,factors3,it3,error3,time3=nn_CPRAND(t,r,list_S[s],list_P[s],factors=copy.deepcopy(factors),exact_err=False,it_max=10000,err_it_max=10000,tol=tol,time_rec=True)
                    weights4,factors4,it4,error4,cpt4,time4=nn_her_CPRAND(t,r,list_S[s],list_P[s],factors=copy.deepcopy(factors),exact_err=False,it_max=10000,err_it_max=10000,tol=tol,time_rec=True) 
                time_hercprand[k,s] =np.cumsum(time4)[it4-1]
                time_cprand[k,s] = np.cumsum(time3)[it3-1]
                

        vsals[i,:] = np.mean(copy.deepcopy(time_als))/ np.mean(copy.deepcopy(time_hercprand),0)
        seals[i,:]= np.std(copy.deepcopy(time_als)) / np.mean(copy.deepcopy(time_hercprand),0)
        vsherals[i,:] =np.mean(copy.deepcopy(time_herals))/np.mean(copy.deepcopy(time_hercprand),0)
        seherals[i,:]= np.std(copy.deepcopy(time_herals))/ np.mean(copy.deepcopy(time_hercprand),0)
        vscprand[i,:] = np.mean(copy.deepcopy(time_cprand),0)/np.mean(copy.deepcopy(time_hercprand),0)
        secprand[i,:]=  np.std(copy.deepcopy(time_cprand),0)/ np.mean(copy.deepcopy(time_hercprand),0)
    # plot
    plt.figure(0)
    for s in range(len(list_S)):
        legend = "S = " + str(list_S[s]) +" , P = " + str(list_P[s])
        plt.errorbar(list_N, vsals[:,s], yerr=seals[:,s],label=legend)
        plt.xlabel('N')
        plt.ylabel('Speed up factor')
        plt.legend(loc='best')
        plt.title('Speed up vs als')
        
    plt.figure(1)
    for s in range(len(list_S)):
        legend = "S = " + str(list_S[s]) +" , P = " + str(list_P[s])
        #plt.plot(list_N,vsherals[:,s],label=legend)
        plt.errorbar(list_N, vsherals[:,s], yerr=seherals[:,s],label=legend)
        plt.xlabel('N')
        plt.ylabel('Speed up factor')
        plt.legend(loc='best')
        plt.title('Speed up vs herals')
                
    plt.figure(2)
    for s in range(len(list_S)):
        legend = "S = " + str(list_S[s]) +" , P = " + str(list_P[s])
        plt.errorbar(list_N, vscprand[:,s], yerr=secprand[:,s],label=legend)
        plt.xlabel('N')
        plt.ylabel('Speed up factor')
        plt.legend(loc='best')
        plt.title('Speed up vs cprand')
    return(vsals,seals)

