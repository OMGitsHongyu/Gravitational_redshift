import Gadget2Snapshot as gs
from analysis import calculator as calc
import numpy as np
from matplotlib import pylab as plt

galaxy = gs.Gadget2Snapshot.open('galaxy_littleendian.dat')
galaxy.readSnapshot()

epsilon = 1.85
center = np.array([100,50,0], dtype=np.float32)
pselect = []

for i in xrange(galaxy.mass.shape[0]):
    if calc.euclidean_periodic(galaxy.positions[i,:], center, 0) < 100:
        pselect.append(i)

print np.array(pselect).shape
plt.figure()
plt.scatter(galaxy.positions[pselect,0], galaxy.positions[pselect,1])
plt.show()

galaxy_select = gs.Gadget2Snapshot()
galaxy_select.setTypes(galaxy.types[pselect])
galaxy_select.setMass(galaxy.mass[pselect])
galaxy_select.setPositions(galaxy.positions[pselect,:])
galaxy_select.writeSnapshot('galaxy_select')

a = gs.Gadget2Snapshot.open('galaxy_select')
a.getHeader()
print a.header
