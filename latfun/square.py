r"""2D square lattice.

Useful properties of infinite 2D square lattice with nearest neighbours 
hopping/coupling. Can be used in e.g. tight-binding type models.

Everything is expressed in units of nearest neighbours hopping. By 
convention, normalized dispersion has a minimum at :math: '\mathbf{k} = 0' 
and has the following form
.. math:: E(\mathbf{k}) = -2\cos(k_x) - 2\cos(k_y)

"""
import numpy as np
from scipy import special
from . import _util


BRAVAIS_LATTICE = 'tp'
"""Tetragonal primitive
"""

N_BANDS = 1
"""Number of bands
"""

E_MIN,E_MAX = -4,4
BANDWIDTH = E_MAX-E_MIN

#in units of bond length
d_1 = np.array([1,0])
d_2 = np.array([0,1])


k_1 = 2*np.pi*np.linalg.inv([d_1,d_2]).T[0]  
k_2 = 2*np.pi*np.linalg.inv([d_1,d_2]).T[1]
      
@_util._fun2D    
def disp(kx,ky):
    """Dispersion relation for 2D square lattice (dimensionless).
    
    .. math:: E(kx,ky) = -2\cos(kx) - 2\cos(ky)
    Dispersion relation for 2D square lattice with nearest neighbour
    hopping/coupling.
    Includes negative sign, has minimum at (`kx,ky`) = (0,0)
    
    Parameters
    ----------
    
    kx,ky : array_like
       wave vector, assuming lattice constant a=1
        
    Returns
    -------
    disp : array_like
        Dispersion for given wave vector (kx,ky), same size as kx (ky).
    """
    return -2*np.cos(kx) - 2*np.cos(ky)


def dos(E,sing=False):
    """Denisty of states for %s lattice.
    
    Returns value(s) of DoS
    E - normalized, dimensionless energy
    sing - if singularity should be included (near E=0)
        True - np.inf near E=0
        False - replaced with 18.927117802238033
    """
    
    E = np.asarray(E,dtype=float)
    rho = np.zeros_like(E)
    nonzero = np.abs(E) <=4
    
    rho[nonzero] = 1/(2*np.pi**2)*special.ellipkm1(E[nonzero]**2/16)
    
    #18.927117802238033 is a value returned near singularity
    #handling arrays and single numbers
    if sing:
        return rho    
    else:
        rho = np.asarray(rho)
        rho[rho==np.inf] = 18.927117802238033
        rho[rho<0] = 18.927117802238033
        return rho
    

def gdos(E):
    """Generalized density of states for %s lattice.
    Weighted by d^2 E(k)/d kx^2
    
    Returns value(s) of GDoS
    E - normalized, dimensionless energy     
    """
    #handling of improper limit near 0
 
    E=np.asarray(E,dtype=float)
    grho = np.zeros_like(E)
    nonzero = (np.abs(E) <= 4) & (E!=0) & (E**2!=0)
    grho[nonzero] = 4/(np.pi**2)*(special.ellipe(1-E[nonzero]**2/16)\
         -1/16.*E[nonzero]**2*special.ellipkm1(E[nonzero]**2/16))
    grho[E==0] = 0.4052847345693511
    grho[E**2==0] = 0.4052847345693511
    return grho 


def HSL(N=100):
    """High symmetry lines for %s lattice.
    
    Returns - kx,ky,k 
        kx,ky- coordinates of points along HSL
        k - auxiliary variable for plotting with appropriatly scaled distances
    N - number of points along each high symmetry line    
    """    
    return _util._HSL2D(BRAVAIS_LATTICE,k_1,k_2,N=N)


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


def dos_disc(N=10**5):
    """Discretized Denisty of States for %s lattice.
    
    Returns E,rho - arguments (normalized, dimensionless energy) and corresponding values of DoS
    N - number of points (default 10**5)
    """   
    
    E = np.linspace(-E_MIN,E_MAX,N)
    return E,dos(E)


def gdos_disc(N=10**5):
    """Discretized Generalized Denisty of States for %s lattice.
    
    Returns E,rho - arguments (normalized, dimensionless energy) and corresponding values of GDoS
    N - number of points (default 10**5)
    """   
    
    E = np.linspace(-E_MIN,E_MAX,N)
    return E,gdos(E)