import cython
from matplotlib import pylab as plt

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
cpdef double deuclidean_periodic(double[::1] ppos, double[::1] hpos,\
        double blen) nogil:
    """
    Calculate Euclidean distances within a squared box with double arrays.

    Parameters
    ----------
    ppos: (3,) array
        Particle positions.
    hpos: (3,) array
        Halo positions.
    blen: double
        Box length. If blen < 0, there is no boundary condition.
    Returns
    -------
    Eculidean distance: double
        Eculidean distance with periodic boundary conditions.

    """
    cdef double dx, dy, dz
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


def profile_1d(nbins, dr_large, pfeat, pos, center, blen, islog=False):
    """
    Calculate the potential profile.

    Parameters
    ----------
    nbins : int
        Number of bins.
    dr_large : double
        xlim to plot.
    pfeat : (N,3) array of float
        Particle feature (i.e. potential).
    pos : (N,3) array of float
        Particle positions.
    center : (3,) array of float
        Halo/Cluster center.
    blen : float
        Box length.
    islog : boolean, optional
        If True, plot in log space. Default is False (linear space).

    Returns
    -------
    feat_profile : (N,4) array of double
        Stats of the feature, to be specific,
        Column 0: Mid point of feature each bin
        Column 1: Counts in each bin
        Column 2: Sum of feat in each bin
        Column 3: Poisson error of feat in each bin
    """
    Numpart = pos.shape[0]
    feat_profile = np.zeros((nbins, 4))
    dist_center = np.zeros((Numpart,))
    for i in xrange(Numpart):
        dist_center[i] = euclidean_periodic(pos[i], center, blen)
    dr_small = max(dist_center.min(), np.finfo(np.float32).eps)

    if islog:
        logdx = np.log(dr_large / dr_small) * 1. / nbins
        ind_array = np.floor(np.log(dist_center / dr_small) / logdx)
        a = np.linspace(np.log(dr_small), np.log(dr_large), nbins+1)
        feat_profile[:,0] = np.exp(a[:-1] + np.diff(a)/2)

    else:
        dx = (dr_large - dr_small) * 1. / nbins
        ind_array = np.floor((dist_center - dr_small) / dx)
        a = np.linspace(dr_small, dr_large, nbins+1)
        feat_profile[:,0] = a[:-1] + np.diff(a)/2

    for i in xrange(nbins):
        feat_profile[i,1] = np.sum(ind_array==i)
        feat_profile[i,2] = np.sum(pfeat[ind_array==i])
        feat_profile[i,3] = np.std(pfeat[ind_array==i], ddof=1) / np.sqrt(np.sum(ind_array==i))

    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.plot(feat_profile[:,0], feat_profile[:,1], color='b')
    ax1.errorbar(feat_profile[:,0], feat_profile[:,1], yerr=np.sqrt(feat_profile[:,0]), color='b')
    ax1.set_xlabel('$r\; \mathrm{kpc}/h$')
    ax1.set_ylabel(r'Number of paticles')

    ax2 = fig.add_subplot(122)
    ax2.plot(feat_profile[:,0], feat_profile[:,2]/feat_profile[:,1], color='b')
    ax2.errorbar(feat_profile[:,0], feat_profile[:,2]/feat_profile[:,1], yerr=feat_profile[:,3], color='b')
    ax2.set_xlabel('$r\; \mathrm{kpc}/h$')
    ax2.set_ylabel(r'$z_g\; \mathrm{km/s}$')

    if islog:
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')

    plt.show()

    return feat_profile


def density_profile_1d(nbins, dr_large, pfeat, pos, center, blen, islog=False):
    """
    Calculate the density profile.

    Parameters
    ----------
    nbins : int
        Number of bins.
    dr_large : double
        xlim to plot.
    pfeat : (N,3) array of float
        Particle feature (i.e. mass).
    pos : (N,3) array of float
        Particle positions.
    center : (3,) array of float
        Halo/Cluster center.
    blen : float
        Box length.
    islog : boolean, optional
        If True, plot in log space. Default is False (linear space).

    Returns
    -------
    feat_profile : (N,4) array of double
        Stats of the feature, to be specific,
        Column 0: Mid point of feature each bin
        Column 1: Counts in each bin
        Column 2: Sum of feat in each bin
        Column 3: Poisson error of feat in each bin

    """
    Numpart = pos.shape[0]
    feat_profile = np.zeros((nbins, 4))

    dist_center = np.zeros((Numpart,))
    for i in xrange(Numpart):
        dist_center[i] = euclidean_periodic(pos[i], center, blen)
    dr_small = max(dist_center.min(), np.finfo(np.float32).eps)
    print "dist_center.min = ", dist_center.min()
    print "dr_small = ", dr_small

    if islog:
        logdx = np.log(dr_large / dr_small) * 1. / nbins
        ind_array = np.floor(np.log(dist_center / dr_small) / logdx)
        a = np.linspace(np.log(dr_small), np.log(dr_large), nbins+1)
        feat_profile[:,0] = np.exp(a[:-1] + np.diff(a)/2)
        dr = np.exp(a)

    else:
        dx = (dr_large - dr_small) * 1. / nbins
        ind_array = np.floor((dist_center - dr_small) / dx)
        dr = np.linspace(dr_small, dr_large, nbins+1)
        feat_profile[:,0] = dr[:-1] + np.diff(dr)/2

    for i in xrange(nbins):
        feat_profile[i,1] = np.sum(ind_array==i)
        feat_profile[i,2] = np.sum(pfeat[ind_array==i]) / (4 * np.pi * (dr[i+1] ** 3 - dr[i] ** 3) / 3)
        feat_profile[i,3] = np.sqrt(np.sum(pfeat[ind_array==i])) / (4 * np.pi * (dr[i+1] ** 3 - dr[i] ** 3) / 3)

    fig = plt.figure()

    ax1 = fig.add_subplot(121)
    ax1.plot(feat_profile[:,0], feat_profile[:,1], color='b')
    ax1.errorbar(feat_profile[:,0], feat_profile[:,1], yerr=np.sqrt(feat_profile[:,0]), color='b')
    ax1.set_xlabel('$r\; \mathrm{kpc}/h$')
    ax1.set_ylabel(r'Number of paticles')

    ax2 = fig.add_subplot(122)
    ax2.plot(feat_profile[:,0], feat_profile[:,2], color='b')
    ax2.errorbar(feat_profile[:,0], feat_profile[:,2], yerr=feat_profile[:,3], color='b')
    ax2.set_xlabel('$r\; \mathrm{kpc}/h$')
    ax2.set_ylabel(r'$\rho\; 10^{10}h^2M_\odot\mathrm{kpc}^{-3}$')

    if islog:
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax2.set_yscale('log')

    plt.show()

    return feat_profile


def slicePlot(pos, plane, center, frac, feat_density, feat_grav, nbins):
    """
    Plot a slice along 'x' or 'y' or 'z' direction.

    Parameters
    ----------
    pos : (N,3) array of float or double
        Particle positions.
    plane : {'xy','yx','z','yz','zy','x','xz','zx','y'}, string
        The direction (plane) of the slice.
    center : (3,) array of float or double
        Make sure the slice go through the center to have more
        information, but could be any point.
    frac : float or double
        Slice width is a fraction of the box length.
    feat_density : (N,) array of float or double
        Particle mass.
    feat_grav: (N,) array of float or double
        Particle potentials.
    kwargs : optional
        See numpy.histogram2d bins.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot

    """
    if plane in ['xy', 'yx', 'z']:
        fraclen = np.fabs(pos[:,2] - center[2]).max() * frac
        slice_ind = np.fabs(pos[:,2] - center[2]) <= fraclen
        x, y = pos[slice_ind,0], pos[slice_ind,1]
        xlabel, ylabel = 'x', 'y'
    elif plane in ['yz', 'zy', 'x']:
        fraclen = np.fabs(pos[:,0] - center[0]).max() * frac
        slice_ind = np.fabs(pos[:,0] - center[0]) <= fraclen
        x, y = pos[slice_ind,1], pos[slice_ind,2]
        xlabel, ylabel = 'y', 'z'
    elif plane in ['xz', 'zx', 'y']:
        fraclen = np.fabs(pos[:,1] - center[1]).max() * frac
        slice_ind = np.fabs(pos[:,1] - center[1]) <= fraclen
        x, y = pos[slice_ind,0], pos[slice_ind,2]
        xlabel, ylabel = 'x', 'z'
    else: raise ValueError('A projection plane must correctly specified!')

    w_density = feat_density[slice_ind]
    w_grav = feat_grav[slice_ind]

    H_density, xedges, yedges = np.histogram2d(x, y, weights = w_density, bins=nbins)
    Hw, xedges, yedges = np.histogram2d(x, y, weights = w_grav, bins=nbins)
    Hn, xedges, yedges = np.histogram2d(x, y, bins=nbins)
    Hwmask = np.ma.masked_where(Hw==0 ,Hw)
    Hnmask = np.ma.masked_where(Hn==0 ,Hn)
    grav_profile = Hwmask / Hnmask

    H_density = H_density * 1. / ((xedges[1] - xedges[0]) * (yedges[1] - yedges[0]) * 2 * fraclen)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(H_density, interpolation='nearest', origin='lower',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    cs = ax.contour(grav_profile, origin='lower', colors='k',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    plt.clabel(cs, fontsize=9, inline=1)
    ax.set_xlabel(xlabel+r'$\;\mathrm{kpc}/h$')
    ax.set_ylabel(ylabel+r'$\;\mathrm{kpc}/h$')
    ax.set_aspect('equal')
    cbar = fig.colorbar(im)

    return ax
