import numpy as np
import pdb, time
from calculate_distance import euclidean_periodic
from mpi4py import MPI

def groupcoord(fname, preci) :

    precb = (preci + 1) * 4
    header = np.dtype([
                       ('Npart',(np.uint32,6)),
                       ('Massarr',(np.double,6)),
                       ('Time',np.double),
                       ('Redshift',np.double),
                       ('FlagSfr',np.uint32),
                       ('FlagFeedback',np.uint32),
                       ('Nall',(np.uint32,6)),
                       ('FlagCooling',np.uint32),
                       ('NumFiles',np.uint32),
                       ('BoxSize',np.double),
                       ('Omega0',np.double),
                       ('OmegaLambda',np.double),
                       ('HubbleParam',np.double),
                       ('Na11HW',(np.uint32,6)),
                       ('fill',(np.int8,72))])
    assert header.itemsize == 256
    f = open(fname)
    int1 = np.fromfile(f, dtype=np.uint32, count=1)[0]
    h = np.fromfile(f, dtype=header, count = 1)[0]
    int2 = np.fromfile(f, dtype=np.uint32, count=1)[0]
    assert int1 == int2
    totNpart = h['Npart'].sum()
    int3 = np.fromfile(f, dtype=np.uint32, count=1)[0]
    pos = np.fromfile(f, dtype = ('f'+str(precb),3), count = totNpart)
    int4 = np.fromfile(f, dtype=np.uint32, count=1)[0]
    assert int3 == int4
    f.seek(totNpart*3*precb + 8,1)
    int5 = np.fromfile(f, dtype=np.uint32, count=1)[0]
    pid = np.fromfile(f, dtype = np.uint64, count = totNpart)
    int6 = np.fromfile(f, dtype=np.uint32, count=1)[0]
    assert int5 == int6
    Nm = 0
    for i in range(0,6):
          if (h['Massarr'][i] == 0):
             Nm = Nm + (h['Npart'][i])
    if (Nm > 0):
       int7 = np.fromfile(f,dtype=np.uint32,count=1)[0]
       vpmass = np.fromfile(f, dtype='f'+str(precb), count = Nm)
       int8 = np.fromfile(f,dtype=np.uint32,count=1)[0]
       assert int7 == int8
    ptype = np.zeros(totNpart,dtype = np.uint32)
    pmass = np.zeros(totNpart,dtype = 'f'+str(precb))
    count = 0
    vpcount = 0
    for i in range(len(h['Npart'])):
          ptype[count : count + h['Npart'][i]] = i
          if (h['Massarr'][i] == 0):
             pmass[count : count + h['Npart'][i]] = (vpmass[vpcount : vpcount + h['Npart'][i]]).astype('f'+str(precb))
             vpcount = vpcount + h['Npart'][i]
          if (h['Massarr'][i] > 0):
             pmass[count : count + h['Npart'][i]] = h['Massarr'][i]
          count = count + h['Npart'][i]
#    print count, totNpart, vpcount, Nm
#    print pid[0], pos[0], ptype[0:20], pmass[0]
    return pid, pos, ptype, h, totNpart, pmass

"""
def euclidean_periodic(x1, y1, z1, x2, y2, z2, blen):
    dx = np.fabs(x1 - x2)
    dy = np.fabs(y1 - y2)
    dz = np.fabs(z1 - z2)
    if dx > blen * 1. / 2:
        dx = blen - dx;
    if dy > blen * 1. / 2:
        dy = blen - dy;
    if dz > blen * 1. / 2:
        dz = blen - dz;
    return np.sqrt(dx * dx + dy * dy + dz * dz)
"""

# Subhalo selection
galaxy = np.loadtxt("/home/hongyuz/gravitation_z/mpi_c/subhalo/file_grnr_85.txt")
group = np.loadtxt("/home/hongyuz/gravitation_z/mpi_c/subhalo/group_mass_pos_085.txt")
# Cross reference
n = 10 # Number of subhalos
cutoff = np.logspace(np.log10(group[:,0].min()), np.log10(group[:,0].max()), n+1)
selected_subhalos = np.zeros((n, 5))
for i in range(n):
    ind = np.where((group[:,0] > cutoff[i]) & (group[:,0] <= cutoff[i+1]) == True)[0][0]
    ind_galaxy = np.where(galaxy[:,0] == galaxy[(galaxy[:,4] == ind),0].max())[0][0]
    selected_subhalos[i,:] = galaxy[ind_galaxy,:]

no_subhalo = 8

preci = 1
dirno = '085'
## preci = 1 from dirno = '061'; 0 otherwise

dirname = '/physics/nkhandai/mb2/snapdir/'
ns = 1024
#dirname1 = '/home/hongyuz/gravitation_z'

#fwrite = open(dirname1 + '/mmr.txt','w')

rlim = 10000
nbins = 20
epsilon = 1.85
logdx = np.log(rlim / epsilon) * 1. / nbins
xsubhalo, ysubhalo, zsubhalo = selected_subhalos[no_subhalo,1:]

NuminBin = np.zeros((nbins, 12))
"""
Want:     (0,1) -> total
          (2,3) -> darkmatter only -> Type 1
          (4,5) -> stars + darkmatter -> Type 1, 4
          (6,7) -> gas + stars + darkmatter -> Type 0, 1, 4
          (8,9) -> all particels -> Type 0, 1, 4, 5

Datatype: (0,1,2,3,4,5) -> Types
          (6,7,8,9,10,11) -> Potentials for each type
"""
t0 = time.time()

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

ns_rank = ns / nprocs

for i in range(rank*ns_rank, (rank+1)*ns_rank):
    pid, pos, ptype, h, totNpart, pmass = groupcoord(dirname + '/snapdir_' + dirno + '/snapshot_' + dirno+ '.'+str(i),preci)
    for k in range(pmass.shape[0]):
        dis = euclidean_periodic(pos[k,0], pos[k,1], pos[k,2], xsubhalo, ysubhalo, zsubhalo, 100000) + epsilon
        ktype = ptype[k]
        if dis < rlim:
            ind = np.floor(np.log(dis / epsilon) / logdx)
            NuminBin[ind,ktype] += 1
            NuminBin[ind,ktype+6] += pmass[k] / dis
    del dis, ktype, pid, pos, ptype, h, totNpart, pmass
    print rank, i, time.time() - t0

if rank == 0:
    totals = np.zeros_like(NuminBin)
else:
    totals = None

comm.Reduce([NuminBin, MPI.DOUBLE], [totals, MPI.DOUBLE], op = MPI.SUM, root = 0)

if rank == 0:
    np.savetxt('selected_subhalo_085_10mpc'+str(no_subhalo), totals)
