import numpy as np
from functools import wraps,partial

def _fun2D(fun):
    @wraps(fun)
    def result(kx,ky,*args,**kwargs):
        if np.shape(kx) != np.shape(ky):
            print("kx and ky shape mismatch")
            return None 
        else:
            sh_0 = np.shape(fun(0,0,*args,**kwargs))
            x = np.asarray(kx)
            y = np.asarray(ky)
            fun_temp = partial(fun,*args,**kwargs)
            en_temp = np.array(list(map(fun_temp,x.flatten(),y.flatten())))
            en = np.reshape(en_temp.T,sh_0+kx.shape)
        
            return en     
    
    return result

def _HSL2D(bravais_lattice,k_1,k_2,N=100,points='default'):
    
    if bravais_lattice == 't' or bravais_lattice == 'tp' or bravais_lattice == 'square':
        if points =='default' or points=='GXMG':
            #Gamma-X
            k1 = np.linspace(0,0.5,N,endpoint=False)
            k2 = np.zeros_like(k1)
            kx = k_1[0]*k1
            ky = k_1[1]*k1
            k = np.linspace(0,1,N,endpoint=False)
        
            #X-M
            k1 = 0.5*np.ones_like(k1)
            k2 = np.linspace(0,0.5,N,endpoint=False)
            kx = np.append(kx,k_1[0]*k1 + k_2[0]*k2)
            ky = np.append(ky,k_1[1]*k1 + k_2[1]*k2)
            k = np.append(k,np.linspace(1,2,N,endpoint=False))
        
            #M-Gamma
            k1 = np.linspace(0.5,0,N)
            k2 = np.linspace(0.5,0,N)
            kx = np.append(kx,k_1[0]*k1 + k_2[0]*k2)
            ky = np.append(ky,k_1[1]*k1 + k_2[1]*k2)
            k = np.append(k,np.linspace(2,2+np.sqrt(2),N))
            return kx,ky,k
    elif bravais_lattice == 'h' or bravais_lattice == 'hp' or bravais_lattice == 'hexagonal' or bravais_lattice == 'triangular':
        if points =='default' or points=='GKMG':
            #chosen k_1 reciprocal lattice vector and defined the line Gamma-M along this vector
            #to determine kx,ky on lines Gamma-K and K-M used vector perpendicular to k_1
            #Gamma-K
            k1 = np.linspace(0,0.5,N,endpoint=False)
            k2 = np.linspace(0,np.sqrt(3)/6,N,endpoint=False)
            kx = k_1[0]*k1 - k_1[1]*k2
            ky = k_1[1]*k1 + k_1[0]*k2
            k = np.linspace(0,2/np.sqrt(3),N,endpoint=False)
            
            #K-M
            k1 = 0.5*np.ones_like(k1)
            k2 = np.linspace(np.sqrt(3)/6,0,N,endpoint=False)
            kx = np.append(kx,k_1[0]*k1 - k_1[1]*k2)
            ky = np.append(ky,k_1[1]*k1 + k_1[0]*k2)
            k = np.append(k,np.linspace(2/np.sqrt(3),3/np.sqrt(3),N,endpoint=False))
            
            #M-Gamma
            k1 = np.linspace(0.5,0,N)
            k2 = np.zeros_like(k1)
            kx = np.append(kx,k_1[0]*k1 - k_1[1]*k2)
            ky = np.append(ky,k_1[1]*k1 + k_1[0]*k2)
            k = np.append(k,np.linspace(3/np.sqrt(3),3/np.sqrt(3)+1,N))
            return kx,ky,k
