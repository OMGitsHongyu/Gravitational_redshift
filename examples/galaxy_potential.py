import Gadget2Snapshot
from analysis import calculator

galaxy = Gadget2Snapshot.Gadget2Snapshot.open('galaxy_littleendian.dat')
galaxy.readSnapshot()

epsilon = 1.85
calculator.potential_bf(epsilon, galaxy.mass, galaxy.positions, -1)
