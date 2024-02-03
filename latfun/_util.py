import numpy as np
from functools import wraps, partial
from time import time


def _timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' %
              (f.__name__, te - ts))
        return result

    return wrap


def _fun2D(fun):
    @wraps(fun)
    def result(kx, ky, *args, **kwargs):
        if np.shape(kx) != np.shape(ky):
            print("kx and ky shape mismatch")
            return None
        else:
            sh_0 = np.shape(fun(0, 0, *args, **kwargs))
            x = np.asarray(kx)
            y = np.asarray(ky)
            fun_temp = partial(fun, *args, **kwargs)
            en_temp = np.array(list(map(fun_temp, x.flatten(), y.flatten())))
            en = np.reshape(en_temp.T, sh_0 + kx.shape)

            return en

    return result


def hsl_2d(bravais_lattice, k_1, k_2, n=100, points='default'):
    r"""High Symmetry Lines for 2D lattices.

    Parameters
    ----------
    bravais_lattice : {'tp', 'hp'}
        Bravais lattice.
    k_1, k_2 : ndarray
        Primitive vectors in reciprocal space. Each with shape (2,).
    n : int, default = 100
        Number of points per line [for the total of `n`\*(len(`points`)-1)+1 points].
    points : str, default = 'default'
        Order of High Symmetry Points (HSP) for HSL.
        For each Bravais lattice this can assume some default values.

    Returns
    -------
    kx, ky : ndarray
        Coordinates along HSL.
    k : ndarray
        Auxiliary variable corresponding to coordinates (`kx`, `ky`) with appropriately scaled distances. Same size
        as `kx` (`ky`).
    """
    # Depending on Bravais lattice different set of high symmetry points is defined. They are expressed in units of
    # primitive vectors [`k_1`,`k_2`] in reciprocal space.
    if bravais_lattice in ['tp', 't', 'square']:
        if points == 'default':
            points = 'GXMG'

        hsp = {'G': np.array([0, 0]),
               'X': np.array([0.5, 0]),
               'M': np.array([0.5, 0.5])
               }

    elif bravais_lattice in ['hp', 'h', 'hexagonal']:
        if points == 'default':
            points = 'GMKG'

        hsp = {'G': np.array([0, 0]),
               'K': np.array([2 / 3, 1 / 3]),
               'M': np.array([0.5, 0])
               }

    bravais_matrix = np.array([k_1, k_2]).T
    points_list = [bravais_matrix @ hsp[point] for point in points]

    kx, ky, k = [], [], []
    length_prev = 0
    last_line_idx = len(points_list) - 2

    for idx, point in enumerate(points_list[:-1]):
        next_point = points_list[idx + 1]

        k_line = np.linspace(point, next_point, num=n + (idx == last_line_idx), endpoint=(idx == last_line_idx))

        kx = np.append(kx, k_line[:, 0])
        ky = np.append(ky, k_line[:, 1])

        length_total = length_prev + np.linalg.norm(next_point - point)

        k = np.append(k, np.linspace(length_prev, length_total,
                                     num=n + (idx == last_line_idx), endpoint=(idx == last_line_idx)))
        
        length_prev = length_total

    return kx, ky, k
