r"""2D triangular lattice

Useful properties of infinite 2D triangular lattice with nearest neighbours
hopping/coupling. Can be used in e.g. tight-binding type models.

Everything is expressed in units of nearest neighbours hopping. By
convention, normalized dispersion has a minimum at :math:`k = 0`
and has the following form

.. math:: E(k_x,k_y) = -2\cos(k_x) - 2\cos\left(\frac{k_x + \sqrt{3}k_y}{2}\right)
   - 2\cos\left(\frac{k_x-\sqrt{3}k_y}{2}\right)

"""

import numpy as np
from scipy import special
from . import _util


SQRT_3 = np.sqrt(3)

BRAVAIS_LATTICE = 'hp'
r"""Hexagonal primitive"""

N_BANDS = 1
r"""Number of bands = 1"""

E_MIN = -6
r"""Minimal value of the dispersion = -6"""

E_MAX = 3
r"""Maximal value of dispersion = 3"""

BANDWIDTH = E_MAX-E_MIN
r"""Total bandwidth = 9"""

D_1 = np.array([1, 0])
r"""Primitive vector :math:`d_1 = (1,0)`"""

D_2 = np.array([0.5, SQRT_3/2])
r"""Primitive vector :math:`d_2 = (\frac{1}{2},\frac{\sqrt{3}}{2})`"""

K_1 = 2*np.pi*np.linalg.inv([D_1, D_2]).T[0]
r"""Primitive vector in reciprocal space :math:`k_1 = (2\pi,-\frac{2\sqrt{3}}{3}\pi)`"""

K_2 = 2*np.pi*np.linalg.inv([D_1, D_2]).T[1]
r"""Primitive vector in reciprocal space :math:`k_2 = (0,\frac{4\sqrt{3}}{3}\pi)`"""


def disp(kx, ky):
    r"""Dispersion relation for 2D triangular lattice (dimensionless)

    .. math:: E(k_x,k_y) = -2\cos(k_x) - 2\cos\left(\frac{k_x + \sqrt{3}k_y}{2}\right)
       - 2\cos\left(\frac{k_x-\sqrt{3}k_y}{2}\right)

    Dispersion relation for 2D triangular lattice with nearest neighbour
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
    return -2*np.cos(kx) - 2*np.cos(0.5*(kx + SQRT_3*ky)) - 2*np.cos(0.5*(kx - SQRT_3*ky))


def dos(E, singularity=False):
    r"""Density of states for 2D triangular lattice.

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
    r = np.sqrt(3 - nonzero)
    z0 = ((3 - r)*(r + 1)**3/4)*(nonzero < -2) + 4*r*(nonzero >= -2)
    z1 = 4*r*(nonzero < -2) + ((3 - r)*(r + 1)**3/4)*(nonzero >= -2)
    rho[nonzero] = 1 / (np.sqrt(z0) * np.pi**2) * special.ellipk(z1/z0)

    if not singularity:
        rho_at_singularity = 1 / (2 * np.pi**2) * special.ellipkm1(np.finfo(float).smallest_subnormal)
        rho[rho == np.inf] = rho_at_singularity
        return rho
    else:
        return rho


def hsl(n=100, points='GMKG'):
    r"""High symmetry lines (HSL) for 2D triangular lattice

    Parameters
    ----------
    n : integer, default = 100
        Number of points per line [for the total of n*(number of lines)]
    points : str, default = 'GMKG'
        String describing order of High Symmetry Points (HSP) for HSL

    Returns
    -------
    kx,ky : float np.ndarray
        kx,ky - coordinates along HSL
    k : float np.ndarray
        k - auxiliary variable with appropriately scaled distances
    """    
    return _util.hsl_2d(BRAVAIS_LATTICE, K_1, K_2, n=n, points=points)

#
# def disp_disc(N_1=100,N_2=100):
#     """Discretized dispersion relation for %s lattice (dimensionless).
#
#     Returns kx,ky,disp - wave vectors and values of dispersion (meshgrid form)
#     N_1,N_2 - number of points in respective directions of primitive reciprocal vectors (default 100)
#     """
#
#     k1 = np.linspace(-0.5,0.5,N_1)
#     k2 = np.linspace(-0.5,0.5,N_2)
#     k1,k2 = np.meshgrid(k1,k2)
#     kx = k1*k_1[0] + k2*k_2[0]
#     ky = k1*k_1[1] + k2*k_2[1]
#
#     return kx,ky,disp(kx,ky)
#
#
# disp.__doc__ %= __name__
# HSL.__doc__ %= __name__
# disp_disc.__doc__ %= __name__