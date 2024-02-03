r"""2D square lattice

Useful properties of infinite 2D square lattice with nearest neighbours 
hopping/coupling. Can be used in e.g. tight-binding type models.

Everything is expressed in units of nearest neighbours hopping. By 
convention, normalized dispersion has a minimum at :math:`k = 0`
and has the following form

.. math:: E(k_x,k_y) = -2\cos(k_x) - 2\cos(k_y)

"""

import numpy as np
from scipy import special
from . import _util

BRAVAIS_LATTICE = 'tp'
r"""Tetragonal primitive"""

N_BANDS = 1
r"""Number of bands = 1"""

E_MIN = -4
r"""Minimal value of the dispersion = -4"""

E_MAX = 4
r"""Maximal value of dispersion = 4"""

BANDWIDTH = E_MAX-E_MIN
r"""Total bandwidth = 8"""

D_1 = np.array([1, 0])
r"""Primitive vector :math:`d_1 = (1,0)`"""

D_2 = np.array([0, 1])
r"""Primitive vector :math:`d_2 = (0,1)`"""

K_1 = 2*np.pi*np.linalg.inv([D_1, D_2]).T[0]
r"""Primitive vector in reciprocal space :math:`k_1 = (2\pi,0)`"""

K_2 = 2*np.pi*np.linalg.inv([D_1, D_2]).T[1]
r"""Primitive vector in reciprocal space :math:`k_2 = (0,2\pi)`"""


def disp(kx, ky):
    r"""Dispersion relation for 2D square lattice (dimensionless).
        
    Dispersion relation for wave vector (`kx`, `ky`). `kx` and `ky` should be
    the same size. They can be single numbers or arrays (e.g. in meshgrid format).
    
    Parameters
    ----------
    kx, ky : array_like
        Wave vector (assuming lattice constant a=1).
        
    Returns
    -------
    disp : array_like
        Dispersion for given wave vector (`kx`, `ky`), same length as `kx` (`ky`). Dimensionless (in units of NN hopping)

    Notes
    -----
    Dispersion is given by

    .. math::
       E(k_x,k_y) = -2\cos(k_x) - 2\cos(k_y)
    """
    return -2*np.cos(kx) - 2*np.cos(ky)


def dos(E, singularity=False):
    r"""Density of states for 2D square lattice.
    
    Parameters
    ----------
    E : array_like
        Energy (dimensionless - in units of NN hopping).
    singularity : bool, default = False
        If `singularity`, return singularity near `E`\=0,
        otherwise return machine limit value for the DoS.
   
    Returns
    -------
    rho : array_like
        Value(s) of DoS for given energies `E`.
    """
    E = np.asarray(E, dtype=float)
    rho = np.zeros_like(E)
    nonzero = (E >= E_MIN) & (E <= E_MAX)
    rho[nonzero] = 1/(2*np.pi**2)*special.ellipkm1(E[nonzero]**2/16)

    if singularity:
        return rho
    else:
        rho_at_singularity = 1/(2*np.pi**2)*special.ellipkm1(np.finfo(float).smallest_subnormal)
        rho[rho == np.inf] = rho_at_singularity
        return rho


def hsl(n=100, points='GXMG'):
    r"""High symmetry lines (HSL) for 2D quare lattice.
    
    Parameters
    ----------
    n : int, default = 100
        Number of points per line [for the total of `n`\*(len(`points`)-1)+1 points].
    points : str, default = 'GXMG'
        Order of High Symmetry Points (HSP) for HSL.
    
    Returns
    -------
    kx, ky : ndarray
        Coordinates along HSL.
    k : ndarray
        Auxiliary variable corresponding to coordinates (`kx`, `ky`) with appropriately scaled distances. Same size
        as `kx` (`ky`).
    """
    return _util.hsl_2d(BRAVAIS_LATTICE, K_1, K_2, n=n, points=points)
