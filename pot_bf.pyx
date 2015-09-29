import cython

from libc.math cimport sqrt
import numpy as np

# don’t check for out-of-bounds indexing.
@cython.boundscheck(False)
# assume no negative indexing.
@cython.wraparound(False)
def potential_bf(float epsilon, float[::1] pmass, float[:,::1] pos):
    """
    Calculate potential by brute force in c speed.

    Parameters
    ----------
    epsilon: float
        Softening length.
    pmass: (N,) array of float
        Particle mass.
    pos: (N,3) array of float
        Particle positions.

    Returns
    -------
    pot_bf: (N,) array of float
        Gravitational potentials of each particle.

    """
    cdef:
        float grav_const = 43007.1, tot_dist, dist
        float[::1] pot_bf = np.zeros((pos.shape[0],), dtype=np.float32)
        int Numpart = pos.shape[0], i, j

    for i in xrange(Numpart):
        tot_dist = 0
        for j in xrange(Numpart):
            if i != j:
                dist = sqrt((pos[i,0] - pos[j,0]) * (pos[i,0] - pos[j,0]) +\
                            (pos[i,1] - pos[j,1]) * (pos[i,1] - pos[j,1]) +\
                            (pos[i,2] - pos[j,2]) * (pos[i,2] - pos[j,2]))
                tot_dist += dist
                pot_bf[i] -= grav_const * pmass[i] / (dist + epsilon)
        print i, tot_dist / (Numpart - 1)
    np.savetxt('pot_bf', pot_bf)
    return pot_bf
