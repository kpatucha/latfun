r"""2D square lattice.

Useful properties of infinite 2D square lattice with nearest neighbours 
hopping/coupling. Can be used in e.g. tight-binding type models.

Everything is expressed in units of nearest neighbours hopping. By 
convention, normalized dispersion has a minimum at k = 0 
and has the following form
.. math::
   E(k_x,k_y) = -2\cos(k_x) - 2\cos(k_y)

"""
import numpy as np
from scipy import special
from . import _util


BRAVAIS_LATTICE = 'tp'
r"""Tetragonal primitive
"""

N_BANDS = 1
r"""Number of bands
"""

E_MIN,E_MAX = -4,4
r"""Minimal and maximal values of dispersion
"""

BANDWIDTH = E_MAX-E_MIN
r"""Total bandwidth
"""

d_1,d_2 = np.array([1,0]),np.array([0,1])
r"""Primitive vectors
"""

k_1,k_2 = 2*np.pi*np.linalg.inv([d_1,d_2]).T[0] , 2*np.pi*np.linalg.inv([d_1,d_2]).T[1] 
r"""Primitive vectors in reciprocal space
"""

#@_util._fun2D    
def disp(kx,ky):
    r"""Dispersion relation for 2D square lattice (dimensionless).
    
    .. math::
       E(k_x,k_y) = -2\cos(k_x) - 2\cos(k_y)
        
    Dispersion relation for 2D square lattice with nearest neighbour
    hopping/coupling. Dimensionless - in units of NN hopping.
    
    Parameters
    ----------
    
    kx,ky : float np.ndarray or float
       wave vector (assuming lattice constant a=1)
        
    Returns
    -------
    disp : float np.ndarray or float
        Dispersion for given wave vector (kx,ky), same length as kx (ky).
    """
    return -2*np.cos(kx) - 2*np.cos(ky)


def dos(E,singularity=False):
    r"""Denisty of states for 2D square lattice.
    
    Parameters
    ----------
    
    E : float np.ndarray or float
       Energy (dimensionless - in units of NN hopping).
       
    singularity : bool, optional
       If `singularity`, return singularity near E=0,
       otherwise return machine limit value for the DoS (defualt = False)
   
    Returns
    -------
    
    rho : float np.ndarray or float
       Value(s) of DoS for given energies `E`
    """
    
    E = np.asarray(E,dtype=float)
    rho = np.zeros_like(E)
    nonzero = (E>=E_MIN) and (E<=E_MAX)
    
    rho[nonzero] = 1/(2*np.pi**2)*special.ellipkm1(E[nonzero]**2/16)
    
    
    if singularity:
        return rho    
    else:
        rho_at_singularity = 1/(2*np.pi**2)*special.ellipkm1(np.finfo(float).smallest_subnormal)
        rho[rho==np.inf] = rho_at_singularity
        rho[rho<0] = rho_at_singularity
        return rho


def HSL(N=100):
    r"""High symmetry lines for 2D quare lattice.
    
    Parameters
    ----------
    
    N : integer
        Number of points per line (for the total of N*number of lines)
    
    Returns
    -------
    
    kx,ky,k : float np.ndarray
       kx,ky - coordinates along HSL
       k - auxiliary variable with approriatly scalesd distances
    """    
    return _util._HSL2D(BRAVAIS_LATTICE,k_1,k_2,N=N)


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
