"""
This example is to generate a uniformly distributed sphere with radius = rlim
for further test on potetial profiles between Gadget2 and brute force method
"""

import numpy as np
import Gadget2Snapshot as gs

rlim = 100
uniform_square = np.random.uniform(-rlim, rlim, (100000,3))
uniform_sphere = []
for i in xrange(uniform_square.shape[0]):
    if uniform_square[i,0] * uniform_square[i,0] + uniform_square[i,1] * uniform_square[i,1]\
    + uniform_square[i,2] * uniform_square[i,2] <= rlim * rlim:
        uniform_sphere.append(uniform_square[i,:])
pos = np.vstack(uniform_sphere)
ptype = np.ones(pos.shape[0], dtype=np.int32)
pmass = np.ones(pos.shape[0], dtype=np.float32)

uniform_test = gs.Gadget2Snapshot()
uniform_test.setMass(pmass)
uniform_test.setTypes(ptype)
uniform_test.setPositions(np.float32(pos))
uniform_test.writeSnapshot('uniform_snapshot')

uniform_read = gs.Gadget2Snapshot.open('uniform_snapshot')
uniform_read.readSnapshot()

uniform_read.scatterXYZ()
