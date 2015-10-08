import Gadget2Snapshot as gs
from analysis import calculator as calc
import numpy as np
from matplotlib import pylab as plt

cluster = gs.Gadget2Snapshot.open('cluster_littleendian.dat')
cluster.readSnapshot()

epsilon = 1.85

pot_cluster = calc.potential_bf(epsilon, cluster.mass, cluster.positions, -1)

np.savetxt('pot_cluster', pot_cluster)
