import cython

from libc.math cimport sqrt, fabs
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float euclidean_periodic(float[::1] ppos, float[::1] hpos,\
        float blen) nogil:
    """
    Calculate Euclidean distances within a squared box.

    Parameters
    ----------
    ppos: (3,) array
        Particle positions.
    hpos: (3,) array
        Halo positions.
    blen: float
        Box length. If blen < 0, there is no boundary condition.
    Returns
    -------
    Eculidean distance: float
        Eculidean distance with periodic boundary conditions.

    """
    cdef float dx, dy, dz
    dx = fabs(ppos[0] - hpos[0])
    dy = fabs(ppos[1] - hpos[1])
    dz = fabs(ppos[2] - hpos[2])

    if blen > 0:
        if dx > blen / 2.0: dx = blen - dx;
        if dy > blen / 2.0: dy = blen - dy;
        if dz > blen / 2.0: dz = blen - dz;

    return sqrt(dx * dx + dy * dy + dz * dz)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def potential_bf(float epsilon, float[::1] pmass, float[:,::1] pos,\
        float blen):
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
    blen: float
        Box length. If blen < 0, there is no boundary condition.

    Returns
    -------
    pot_bf: (N,) array of float
        Gravitational potentials of each particle.

    """
    cdef:
        float grav_const = 43007.1, tot_dist, dist
        float[::1] pot_bf = np.zeros((pos.shape[0],), dtype=np.float32)
        int Numpart = pos.shape[0], i, j

    if blen > 0:
        for i in xrange(Numpart):
            tot_dist = 0
            for j in xrange(Numpart):
                if i != j:
                    dist = euclidean_periodic(pos[i,:], pos[j,:], blen)
                    tot_dist += dist
                    pot_bf[i] -= grav_const * pmass[i] / (dist + epsilon)
    else:

    # For better speed.

        for i in xrange(Numpart):
            tot_dist = 0
            for j in xrange(Numpart):
                if i != j:
                    dist = sqrt((pos[i,0] - pos[j,0]) * (pos[i,0] - pos[j,0]) +\
                                (pos[i,1] - pos[j,1]) * (pos[i,1] - pos[j,1]) +\
                                (pos[i,2] - pos[j,2]) * (pos[i,2] - pos[j,2]))
                    tot_dist += dist
                    pot_bf[i] -= grav_const * pmass[i] / (dist + epsilon)

    return pot_bf
