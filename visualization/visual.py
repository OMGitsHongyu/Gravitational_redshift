import numpy as np

try:
    import matplotlib.pyplot as plt
    matplotlib = True
except ImportError:
    matplotlib = False


def phistgram(pfeat, **kwarg):
    """
    Plot histgrams on a given feature and return a matplotlib.axes._subplots.AxesSubplot
    for future labeling.

    Parameters
    ----------
    pfeat : (n,) array or sequence of (n,) arrays
        Particle feature(s), this takes either a single array or a sequency of arrays
        which are not required to be of the same length.
    kwargs : optional
        See matplotlib.pyplot.hist.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(pfeat, **kwarg)
    return ax
