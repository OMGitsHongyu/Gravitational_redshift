import Gadget2Snapshot as gs
from analysis import calculator_double as calc
import numpy as np
from matplotlib import pylab as plt
from mpi4py import MPI

precision = 1               # 1 from 061; 0 otherwise
dirno = '085'
dirname = '/physics/nkhandai/mb2/snapdir/'

epsilon = 1.85
rlim = 2000
center = np.array([37415.1875, 40487.3281, 60950.1797])

ns = 1024
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()
ns_rank = ns / nprocs       # if ns is not dividable by nprocs, need to consider the remains
miss = np.zeros((ns,))
pos_total = []
types_total = []
mass_total = []


for i in xrange(rank*ns_rank, (rank+1)*ns_rank):
    pselect = []
    print rank, i
    mb2sim = gs.Gadget2Snapshot.open(dirname + '/snapdir_' + dirno + '/snapshot_' +\
            dirno + '.' +str(1))
    mb2sim.preci = precision
    mb2sim.longids = 'True'
    mb2sim.readSnapshot()
    for j in xrange(mb2sim.mass.shape[0]):
        if calc.euclidean_periodic(mb2sim.positions[j,:], center, 100000) < rlim:
            pselect.append(j)
        else:
            miss[i] += 1
    if not pselect:
        pos_total.append(mb2sim.positions[pselect,:])
        types_total.append(mb2sim.types[pselect])
        mass_total.append(mb2sim.mass[pselect])
    del mb2sim


mb2sim_select = gs.Gadget2Snapshot()
mb2sim_select.setTypes(types_total)
mb2sim_select.setMass(mass_total)
mb2sim_select.setPositions(pos_total)
mb2sim_select.writeSnapshot('mb2sim_select')

a = gs.Gadget2Snapshot.open('mb2sim_select')
a.getHeader()
print a.header
