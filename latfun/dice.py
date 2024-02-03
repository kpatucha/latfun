import numpy as np
from scipy import special
from . import _util

__name__ = 'dice'

bravais_lattice = 'hp'

n_bands = 3

e_min = -3*np.sqrt(3)
e_max = 3*np.sqrt(3)

bandwidth = e_max-e_min

#in units of bond length
d_1 = np.array([np.sqrt(3),0])
d_2 = np.array([np.sqrt(3)/2,3/2])

k_1 = 2*np.pi*np.linalg.inv([d_1,d_2]).T[0]  
k_2 = 2*np.pi*np.linalg.inv([d_1,d_2]).T[1]

@_util._fun2D
def ham(kx,ky):
    """Hamiltonian kernel for %s lattice (dimensionless)
    
    Returns Hamiltonian matrix
    kx,ky - wave vector in units of lattice constant a
    Sites order [hub,rim_1,rim_2]    
    """
    H = np.zeros((n_bands,n_bands))+ 1j*np.zeros((n_bands,n_bands))
    
    k = np.array((kx,ky))
    d1=d_1 @ k
    d2=d_2 @ k
    d3=(d_2-d_1) @ k
    
    H[0,1] = -np.exp(1j*(d1+d2)/3) - np.exp(1j*(d3-d1)/3) - np.exp(1j*(-d2-d3)/3) 
    H[0,2] = -np.exp(1j*(-d1-d2)/3) - np.exp(1j*(d1-d3)/3) - np.exp(1j*(d2+d3)/3)
    H[1,0] = -np.exp(-1j*(d1+d2)/3) - np.exp(-1j*(d3-d1)/3) - np.exp(-1j*(-d2-d3)/3)
    H[2,0] = -np.exp(-1j*(-d1-d2)/3) - np.exp(-1j*(d1-d3)/3) - np.exp(-1j*(d2+d3)/3)
    
    return H


@_util._fun2D    
def disp(kx,ky):
    """Dispersion relation for %s lattice (dimensionless).
    
    kx,ky - wave vector in units of lattice constant a
    """
    k = np.array([kx,ky])
    
    en=np.zeros((n_bands))
    en[0] = - np.sqrt(2)*np.sqrt(3 + 2*np.cos(d_1 @ k) + 2*np.cos(d_2 @ k) + 2*np.cos((d_2-d_1) @ k))
    en[2] = -en[0]
    return en


def HSL(N=100):
    """High symmetry lines for %s lattice.
    
    Returns - kx,ky,k 
        kx,ky- coordinates of points along HSL
        k - auxiliary variable for plotting with appropriatly scaled distances
    N - number of points along each high symmetry line    
    """    
    return _util._HSL2D(bravais_lattice,k_1,k_2,N=N)


def disp_disc(N_1=100,N_2=100):
    """Discretized dispersion relation for %s lattice (dimensionless).
    
    Returns kx,ky,disp - wave vectors and values of dispersion (meshgrid form)
    N_1,N_2 - number of points in respective directions of primitive reciprocal vectors (default 100)
    """
    
    k1 = np.linspace(-0.5,0.5,N_1)
    k2 = np.linspace(-0.5,0.5,N_2)
    k1,k2 = np.meshgrid(k1,k2)
    kx = k1*k_1[0] + k2*k_2[0]
    ky = k1*k_1[1] + k2*k_2[1]
    
    return kx,ky,disp(kx,ky)

ham.__doc__ %= __name__
disp.__doc__ %= __name__
HSL.__doc__ %= __name__
disp_disc.__doc__ %= __name__