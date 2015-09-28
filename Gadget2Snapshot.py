import numpy as np

try:
    import matplotlib.pyplot as plt
    matplotlib = True
except ImportError:
    matplotlib = False

header = np.dtype([
                   ('Npart', (np.int32, 6)),
                   ('Massarr', (np.double, 6)),
                   ('Time', np.double),
                   ('Redshift', np.double),
                   ('FlagSfr', np.int32),
                   ('FlagFeedback', np.int32),
                   ('Nall', (np.int32, 6)),
                   ('FlagCooling', np.int32),
                   ('NumFiles', np.int32),
                   ('BoxSize', np.double),
                   ('Omega0', np.double),
                   ('OmegaLambda', np.double),
                   ('HubbleParam', np.double),
                   ('Na11HW', (np.int32, 6)),
                   ('fill', (np.int8, 72))
                  ])

class Gadget2Snapshot(object):

    """
    A class dealing with I/O, modifying data and visualization of Gadget 2 snapshot files.
    """

    def __init__(self, fp=None, preci=0, longid='False'):
        """

        Parameters
        ----------
        preci : {0, 1}, int, optional
            0 -> float, 1 -> double
            Single or double precision. In NKhandai's file, 1 from dirno = '061',
            0 otherwise. Default is 0.
        longid : boolean, optional
            The ID format for the input snapshot files. Int as the Gadget2 user
            guide suggests, long for large simulations (NKhandai). Default is 'False'.

        """
        assert (type(fp) == file) or (fp is None), "Call the open() method first!"

        if fp is not None:
            self.fp = fp
            self.preci = preci
            self.longid = longid

    @classmethod
    def open(cls, fname):
        """
        Open a gadget snapshot file.

        Parameters
        ----------
        fname : file-like object or string
            The file to read.

        Returns
        -------
        class object

        """
        if type(fname) == str:
            fp = open(fname, 'rb')
        else:
            raise TypeError("Filename must be string!")

        return cls(fp)


    def close(self):
        """
        Close the Gadget snapshot file.

        """
        self.fp.close()


    def getHeader(self, save=True):
        """
        Read in the header.

        Parameters
        ----------
        save : boolean, optional
            if True then save header as attribute. Default is True.

        Returns
        -------
        h : numpy void
            Header of the snapshot file.

        """
        assert not self.fp.closed

        self.fp.seek(0, 0)

        int1 = np.fromfile(self.fp, dtype=np.int32, count=1)[0]
        h = np.fromfile(self.fp, dtype=header, count=1)[0]
        int2 = np.fromfile(self.fp, dtype=np.int32, count=1)[0]
        assert int1 == int2, "Header is not read properly!"

        if save: self.header = h

        return h


    def getPositions(self, save=True):
        """
        Read in the positions.

        Parameters
        ----------
        save : boolean, optional
            if True then save the positions as attribute. Default is True.

        Returns
        -------
        pos : (N,3) array of float or double
            Particle positions.

        """
        precb = (self.preci + 1) * 4

        totNpart = self.header['Npart'].sum()

        self.fp.seek(256+8, 0)
        int3 = np.fromfile(self.fp, dtype=np.int32, count=1)[0]
        pos = np.fromfile(self.fp, dtype=('f'+str(precb), 3), count=totNpart)
        int4 = np.fromfile(self.fp, dtype=np.int32, count=1)[0]
        assert int3 == int4, "Positions are not read properly!"

        if save: self.positions = pos

        return pos


    def getVelocities(self, save=True):
        """
        Read in the velocities.

        Parameters
        ----------
        save : boolean, optional
            if True then save the velocities as attribute. Default is True.

        Returns
        -------
        vel : (N,3) array of float or double
            Particle velocities.

        """
        precb = (self.preci + 1) * 4

        totNpart = self.header['Npart'].sum()

        self.fp.seek(256+16+3*precb*totNpart, 0)
        int5 = np.fromfile(self.fp, dtype=np.int32, count=1)[0]
        vel = np.fromfile(self.fp, dtype=('f'+str(precb), 3), count=totNpart)
        int6 = np.fromfile(self.fp, dtype=np.int32, count=1)[0]
        assert int5 == int6, "Velocities are not read properly!"

        if save: self.velocities = vel

        return vel


    def getIDs(self, save=True):
        """
        Read in the particle IDs.

        Parameters
        ----------
        save : boolean, optional
            if True then save the IDs as attribute. Default is True.

        Returns
        -------
        pid : (N,) array of int or long
            Particle IDs.

        """
        precb = (self.preci + 1) * 4
        totNpart = self.header['Npart'].sum()

        self.fp.seek(256+24+2*3*precb*totNpart, 0)
        int7 = np.fromfile(self.fp, dtype=np.int32, count=1)[0]
        if self.longid == 'False':
            pid = np.fromfile(self.fp, dtype=np.int32, count=totNpart)
        else:
            if self.longid:
                pid = np.fromfile(self.fp, dtype=np.int64, count=totNpart)
            else:
                raise ValueError('longid must be either True or False')
        int8 = np.fromfile(self.fp, dtype=np.int32, count=1)[0]
        assert int7 == int8, "IDs are not read properly!"

        if save: self.ids = pid

        if not self.fp.read(1): print "Done reading"

        return pid


    def getTypeMass(self, save=True):
        """
        Read in the particle types and mass.

        Parameters
        ----------
        save : boolean, optional
            if True then save the mass as attribute. Default is True.

        Returns
        -------
        ptype : (N,) array of int
            Particle types in {0,1,2,3,4,5}.
        pmass : (N,) array of float or double
            Particle mass.

        """
        precb = (self.preci + 1) * 4
        totNpart = self.header['Npart'].sum()
        if self.longid == 'False':
            self.fp.seek(256+32+2*3*precb*totNpart+4*totNpart, 0)
        else:
            self.fp.seek(256+32+2*3*precb*totNpart+8*totNpart, 0)

        Nm = 0
        for i in range(6):
            if self.header['Massarr'][i] == 0:
                Nm += self.header['Npart'][i]
        if Nm > 0:
            int9 = np.fromfile(self.fp, dtype=np.int32, count=1)[0]
            vpmass = np.fromfile(self.fp, dtype='f'+str(precb), count=Nm)
            int10 = np.fromfile(self.fp, dtype=np.int32, count=1)[0]
            assert int9 == int10, "Variable particle mass is not read properly!"
        ptype = np.zeros(totNpart, dtype=np.int32)
        pmass = np.zeros(totNpart, dtype='f'+str(precb))
        count = 0
        vpcount = 0
        for i in range(6):
            ptype[count : count + self.header['Npart'][i]] = i
            if (self.header['Massarr'][i] == 0) & (Nm > 0):
                pmass[count : count + self.header['Npart'][i]] = (vpmass[vpcount                      : vpcount + self.header['Npart'][i]]).astype('f'+str(precb))
                vpcount += self.header['Npart'][i]
            if self.header['Massarr'][i] > 0:
                pmass[count : count + self.header['Npart'][i]] = self.header['Massarr'][i]
            count += self.header['Npart'][i]

        if save:
            self.types = ptype
            self.mass = pmass

        if not self.fp.read(1): print "Done reading (mass)"

        return ptype, pmass


    def getPotentials(self, save=True):
        """
        Read in the particle gravitational potentials.

        Parameters
        ----------
        save : boolean, optional
            if True then save the mass as attribute. Default is True.

        Returns
        -------
        pot : (N,) array of float or double
            Particle potentials.

        """
        precb = (self.preci + 1) * 4
        totNpart = self.header['Npart'].sum()
        Nm = 0
        for i in range(6):
            if self.header['Massarr'][i] == 0:
                Nm += self.header['Npart'][i]

        if self.longid == 'False':
            self.fp.seek(256+40+2*3*precb*totNpart+4*totNpart+Nm*precb, 0)
        else:
            self.fp.seek(256+40+2*3*precb*totNpart+8*totNpart+Nm*precb, 0)
        int11 = np.fromfile(self.fp, dtype=np.int32, count=1)[0]
        pot = np.fromfile(self.fp, dtype=('f'+str(precb)), count=totNpart)
        int12 = np.fromfile(self.fp, dtype=np.int32, count=1)[0]
        assert int11 == int12, "Potentials are not read properly!"

        if save: self.potentials = pot

        if not self.fp.read(1): print "Done reading (potentials)"

        return pot


    def readSnapshot(self, save=True, readvel=False):
        """
        Read in the snapshot file.

        Parameters
        ----------
        save : boolean, optional
            if True then save particle information as attribute. Default is True.
        readvel : boolean, optional
            if True then read in velocities. Default is False.

        Returns
        -------
        ptype : (N,) array of int
            Particle types in {0,1,2,3,4,5}.
        pid : (N,) array of int or long
            Particle IDs where N is the number of particles.
        pmass : (N,) array of float or double
            Particle mass.
        pos : (N,3) array of float or double
            Particle positions.
        vel : (N,3) array of float or double
            Particle velocities.
        pot : (N,) array of float or double
            Particle potentials.

        """
        self.getHeader(save)
        pos = self.getPositions(save)
        if readvel: vel = self.getVelocities(save)
        pid = self.getIDs(save)
        ptype, pmass = self.getTypeMass(save)

        if self.fp.read(1):
            pot = self.getPotentials(save)
            print "Done reading with potential"
        else:
            print "Done reading"


    def setPositions(self, pos):
        """
        Set the positions.

        Parameters
        ----------
        pos : (N,3) array of float or double
            Particle positions.

        """
        assert pos.shape[1] == 3

        self.positions = pos


    def setVelocities(self, vel):
        """
        Set the velocities.

        Parameters
        ----------
        vel : (N,3) array of float or double
            Particle velocities.

        """
        assert vel.shape[1] == 3

        self.velocities = vel


    def setTypes(self, ptype):
        """
        Set the particle types.

        Parameters
        ----------
        ptype : (N,) array of int
            Particle types in {0,1,2,3,4,5}.

        """

        self.types = ptype


    def setMass(self, pmass):
        """
        Set the particle mass.

        Parameters
        ----------
        pmass : (N,) array of float or double
            Particle mass.

        """

        self.mass = pmass


    def _setHeader(self, Time=0, Redshift=0, FlagSfr=0, FlagFeedback=0,
                   FlagCooling=0, NumFiles=1, BoxSize=0, Omega0=0.3,
                   OmegaLambda=0.7, HubbleParam=0.7):
        """
        Set the header.

        Parameters
        ----------
        cosmological params: optional

        """
        precb = (self.preci + 1) * 4

        assert self.types.shape[0] == self.mass.shape[0] == self.positions.shape[0]\
                == self.velocities.shape[0], "Number of particles are not the same!"

        type_ind = np.argsort(self.types, kind='mergesort')
        ptype_sorted = self.types[type_ind]
        pmass_sorted = self.mass[type_ind]
        Npart = np.zeros(6, dtype=np.int32)
        Massarr = np.zeros(6, dtype=('f'+str(precb)))
        Nall = np.zeros(6, dtype=np.int32)
        typetotal = len(pmass_sorted)
        count = 0
        vpcount = 0
        Nm = 0
        for i in range(6):
            Npart[i] = (ptype_sorted == i).sum()
            if Npart[i] > 0:
                if all(pmass_sorted[count : count + Npart[i]] == pmass_sorted[count]):
                    Massarr[i] += pmass_sorted[count]
                if any(pmass_sorted[count : count + Npart[i]] != pmass_sorted[count]):
                    Nm += Npart[i]
            count += Npart[i]
            Nall[i] += Npart[i]
            if count == typetotal:
                break

        header_node = np.array([(Npart, Massarr, Time, Redshift, FlagSfr, FlagFeedback,
                         Nall, FlagCooling, NumFiles, BoxSize, Omega0,
                         OmegaLambda, HubbleParam, np.zeros(6, dtype=np.int32),
                         np.zeros(72, dtype=np.int8))], dtype=header)

        self.header = header_node[0]


    def writeSnapshot(self, fname, **kwargs):
        """
        Write the snapshot file.

        Parameters
        ----------
        fname : file-like object or string
            The file to read.
        kwargs : optional
            Cosmological params. See the method '_setHeader'.

        """

        assert hasattr(self, "types"), "Types must be specified!"
        assert hasattr(self, "mass"), "Mass must be specified!"
        assert hasattr(self, "positions"), "Positions must be specified!"
        assert not hasattr(self, "velocities"), "Velocities are set to 0s!"

        precb = (self.preci + 1) * 4

        self._setHeader(**kwargs)
        type_ind = np.argsort(self.types, kind='mergesort')
        ptype_sorted = self.types[type_ind]
        pmass_sorted = self.mass[type_ind]
        pos_sorted = self.positions[type_ind]

        if not hasattr(self, "velocities"):
            vel_sorted = np.zeros((type_ind.shape[0], 3), dtype=('f'+str(precb)))
        else:
            vel_sorted = self.velocities[type_ind]

        Npart = np.zeros(6, dtype=np.int32)
        Massarr = np.zeros(6, dtype=('f'+str(precb)))
        Nall = np.zeros(6, dtype=np.int32)
        typetotal = len(pmass_sorted)

        count = 0
        vpcount = 0
        Nm = 0
        vpmass = []
        for i in range(6):
            Npart[i] = (ptype_sorted == i).sum()
            if Npart[i] > 0:
                if all(pmass_sorted[count : count + Npart[i]] == pmass_sorted[count]):
                    Massarr[i] += pmass_sorted[count]
                if any(pmass_sorted[count : count + Npart[i]] != pmass_sorted[count]):
                    vpmass.append(pmass_sorted[count : count + Npart[i]])
                    Nm += Npart[i]
            count += Npart[i]
            Nall[i] += Npart[i]
            if count == typetotal:
                break

        precb = (self.preci + 1) * 4
        header_size = np.array(256, dtype=np.int32)
        pos_size = np.array(typetotal * 3 * precb, dtype=np.int32)
        vel_size = np.array(typetotal * 3 * precb, dtype=np.int32)
        mass_size = np.array(typetotal * 3 * precb, dtype=np.int32)
        if self.longid:
            pid_size = np.array(typetotal * 8, dtype=np.int32)
        else:
            pid_size = np.array(typetotal * 4, dtype=np.int32)

        fw = open(fname,'wb')
        header_size.tofile(fw)
        header_node[0].tofile(fw)
        header_size.tofile(fw)
        pos_size.tofile(fw)
        pos_sorted.tofile(fw)
        pos_size.tofile(fw)
        vel_size.tofile(fw)
        vel_sorted.tofile(fw)
        vel_size.tofile(fw)
        pid_size.tofile(fw)
        pid_sorted.tofile(fw)
        pid_size.tofile(fw)

        if Nm > 0:
            if self.preci == 0:
                vpmass = np.float32(np.concatenate(vpmass))
            else:
                if self.preci == 1:
                    vpmass = np.double(np.concatenate(vpmass))
                else:
                    raise ValueError('Variable mass must be either 0(float) or 1(double)')
            vm_size = np.array(Nm * vpmass.itemsize, dtype=np.int32)

            vm_size.tofile(fw)
            vpmass.tofile(fw)
            vm_size.tofile(fw)

        fw.close()


    def euclidean_periodic(ppos, hpos, blen=None):
        """
        Calculate Eculidean distances within a squared box.

        Parameters
        ----------
        ppos: (3,) array
            Particle positions.
            hpos: (3,) array
            Halo positions.
            blen: None or double
            Box length. Default is None

        Returns
        -------
        Eculidean distance: double
            Eculidean distance with periodic boundary conditions.

        """
        assert ppos.shape[0] == 3
        assert hpos.shape[0] == 3
        dx, dy, dz = np.fabs(ppos - hpos)
        if blen is not None:
            if dx > blen * 1. / 2: dx = blen - dx;
            if dy > blen * 1. / 2: dy = blen - dy;
            if dz > blen * 1. / 2: dz = blen - dz;

        return np.sqrt(dx * dx + dy * dy + dz * dz)


    def scatterXYZ(self, fig=None, **kwargs):
        """
        Scatterplot the particles in XY, XZ, YZ projected plane.

        Parameters
        ----------

        fig: matplotlib.figure.Figure object

        """
        if not matplotlib:
            raise ImportError("matplotlib is not imported, install matplotlib first!")

        # Get the positions if you didn't do it before
        if not hasattr(self, "positions"):
            self.getPositions()

        # Instantiate figure
        if fig is None:
            self.fig = plt.figure()
        else:
            self.fig = fig

        ax1 = fig.add_subplot(1,3,1)

        if not hasattr(self, "types"):
            ax1.scatter(self.positions[:,0], self.positions[:,1], s=1, color='k')
        else:
            ax1.scatter(self.positions[ptype==0,0], self.positions[ptype==0,1], s=1, color='b')
            ax1.scatter(self.positions[ptype==1,0], self.positions[ptype==1,1], s=1, color='k')
            ax1.scatter(self.positions[ptype==2,0], self.positions[ptype==2,1], s=1, color='r')
            ax1.scatter(self.positions[ptype==3,0], self.positions[ptype==3,1], s=1, color='g')
            ax1.scatter(self.positions[ptype==4,0], self.positions[ptype==4,1], s=1, color='y')

        ax1.set_xlabel("$X\; \mathrm{kpc}/h$")
        ax1.set_ylabel("$Y\; \mathrm{kpc}/h$")

        ax2 = fig.add_subplot(1,3,2)

        if not hasattr(self, "types"):
            ax2.scatter(self.positions[:,0], self.positions[:,2], s=1, color='k')
        else:
            ax2.scatter(self.positions[ptype==0,0], self.positions[ptype==0,2], s=1, color='b')
            ax2.scatter(self.positions[ptype==1,0], self.positions[ptype==1,2], s=1, color='k')
            ax2.scatter(self.positions[ptype==2,0], self.positions[ptype==2,2], s=1, color='r')
            ax2.scatter(self.positions[ptype==3,0], self.positions[ptype==3,2], s=1, color='g')
            ax2.scatter(self.positions[ptype==4,0], self.positions[ptype==4,2], s=1, color='y')

        ax2.set_xlabel("$X\; \mathrm{kpc}/h$")
        ax2.set_ylabel("$Z\; \mathrm{kpc}/h$")

        ax3 = fig.add_subplot(1,3,3)

        if not hasattr(self, "types"):
            ax3.scatter(self.positions[:,1], self.positions[:,2], s=1, color='k')
        else:
            ax3.scatter(self.positions[ptype==0,1], self.positions[ptype==0,2], s=1, color='b')
            ax3.scatter(self.positions[ptype==1,1], self.positions[ptype==1,2], s=1, color='k')
            ax3.scatter(self.positions[ptype==2,1], self.positions[ptype==2,2], s=1, color='r')
            ax3.scatter(self.positions[ptype==3,1], self.positions[ptype==3,2], s=1, color='g')
            ax3.scatter(self.positions[ptype==4,1], self.positions[ptype==4,2], s=1, color='y')

        ax3.set_xlabel("$Y\; \mathrm{kpc}/h$")
        ax3.set_ylabel("$Z\; \mathrm{kpc}/h$")

        plt.show()
