#include "mpi.h"
#include <iostream>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include <sstream>
using namespace std;

#define xoff 0
#define yoff 0
#define zoff 0
#define tag1 123
#define tag2 1008
#define tag3 408

typedef struct Galaxy_props{
    double pos[3];  /* positions in Mpc/h */
    double vel[3];  /* peculiar velocities in km/s */
    double grav;    /* gravitational redshift in km/s (wrt the mean level) */
    double mass;    /* mass in gadget units (10^10 m_sun/h) */
} Galaxy;

struct ll_hoc {
    // This struct is used for chaining mesh
    long* ll;
    long*** hoc;
};

long read_Nbody(Galaxy* p, string filename, double peculiar, double gravfac, double* blen);
void write_xc(string filename, long** xc_bin, int nbins);
void quick_sort (Galaxy *p, long np);
template <typename TYPE>
TYPE cwarp(TYPE x, TYPE bound);
long** get2dlong(int nbins);
long*** get3dlong(int nho);
void free2dlong(long** p, int np);
void free3dlong(long*** p, int np);
double bwarp(double x, double box);
void init_mesh(long* ll, long*** hoc, Galaxy* p, long np, int nbins, double blen);
void corr2d(long **xc_bin, Galaxy *p1_rank, Galaxy *p2, struct ll_hoc init, double rlim,
              int nbins, int nhocells, double blen, double dis_f, long nrows,
              double tdfac, double rlcfac, double beamfac, double vlosmax);


int main(int argc, char **argv) {

    long i, ng1, ng2;
    long **xc_bin_rank, **xc_bin;
    double half_range = 200., rlim = 100., blen, tdfac, vlosmax, beamfac, rlcfac;
    int nhocells = 1000, nbins = 400;
    struct ll_hoc init;
    Galaxy *g1_rank, *g2_rank;

    // MPI defs
    MPI_Status status;
    int ierr, my_rank, rank_size, irank;
    long interval, start_row, end_row, nrows_to_send, nrows_to_receive;

    // Define a new MPI datatype, so that structures could be packed in distribute memories.
    MPI_Datatype galaxytype, oldtypes[1];
    int blockcounts[1];
    MPI_Aint offsets[1];

    // MPI initialization
    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &rank_size);

    // MPI new datatype -- memory allocation
    offsets[0] = 0;
    oldtypes[0] = MPI_DOUBLE;
    blockcounts[0] = 8;

    //MPI new datatype -- galaxytype
    MPI_Type_struct(1, blockcounts, offsets, oldtypes, &galaxytype);
    MPI_Type_commit(&galaxytype);

    if (my_rank == 0) {

        cout <<
            "Example: mpirun -np 32 ./a.out a.dat cpp.txt peculiar gravfac tdfac rlcfac beamfac vlosmax!"
            << endl;
        long ngalaxy;
        double masscut = 1000, peculiar = -1, gravfac = -1;

        tdfac = -1;
        vlosmax = 3000;
        beamfac = -1;
        rlcfac = -1;

        // Set parameters and print them out.
        if(argc != 9) {
            cout << "Please enter an input filename and an output filenam!" << endl;
        return(-1);
        }

        peculiar = atof(argv[3]);
        gravfac = atof(argv[4]);
        tdfac = atof(argv[5]);
        rlcfac = atof(argv[6]);
        beamfac = atof(argv[7]);
        vlosmax = atof(argv[8]);

        cout << "Parameters:" << endl;
        cout << "Transverse Doppler factor = " << tdfac << endl;
        cout << "Lightcone factor = " << rlcfac << endl;
        cout << "Beaming factor = " << beamfac << endl;
        cout << "Max line-of-sight velocity = " << vlosmax << " km/s" << endl;

        // Allocate memory for all galaxies.
        Galaxy* galaxy = new Galaxy[ngalaxy];

        // Read in N-body files and prepare on peculiarv/grvz effects.
        ngalaxy = read_Nbody(galaxy, argv[1], peculiar, gravfac, &blen);

        cout << "Number of galaxies = " << ngalaxy << ", box size = " << blen << endl;
        cout << galaxy[ngalaxy-1].grav << endl;

         // Sort galaxies by mass in a descending order.
        quick_sort(galaxy, ngalaxy);
        cout << galaxy[ngalaxy-1].mass << endl;

        for (i = 0; i<ngalaxy; i++)
            if (galaxy[i].mass <= masscut)
                break;

        // g1 denotes the galaxies with higher mass, ng2 denotes the galaxies with lower mass.
        // Note that we almost (odds, even) devide them into two even groups.
        // g1 is then devided into nproc parts sending to different processors.
        // g2 is used as pm grid.
        ng1 = floor((i-1)/2) + 1;
        ng2 = i - 1 - ng1;

        g2_rank = new Galaxy[ng2];
        for (i = 0; i < ng2; i++)  g2_rank[i] = galaxy[i+ng1];

        interval = floor(ng1 / rank_size);
        g1_rank = new Galaxy[interval];
        for (i = 0; i < interval; i++)  g1_rank[i] = galaxy[i];

        for (irank = 1; irank < rank_size; irank++) {
            start_row = irank * interval;
            end_row = (irank + 1) * interval;

            // In case ng1 % rank_size != 0
            if (irank == rank_size - 1)  end_row = ng1;
            nrows_to_send = end_row - start_row;
            cout << "rank = " << irank << ", length = " << nrows_to_send << ", starting from "
                << start_row << endl;

            ierr = MPI_Send(&ng2, 1, MPI_LONG, irank, tag3,
                            MPI_COMM_WORLD);
            ierr = MPI_Send(&nrows_to_send, 1, MPI_LONG, irank, tag1,
                            MPI_COMM_WORLD);
            ierr = MPI_Send((galaxy+start_row), nrows_to_send, galaxytype, irank, tag2,
                            MPI_COMM_WORLD);
         }

        nrows_to_receive = interval;
        delete [] galaxy;
    }

    else {

        ierr = MPI_Recv(&nrows_to_receive, 1, MPI_LONG, 0, tag1,
                        MPI_COMM_WORLD, &status);
        ierr = MPI_Recv(&ng2, 1, MPI_LONG, 0, tag3,
                        MPI_COMM_WORLD, &status);
        g1_rank = new Galaxy[nrows_to_receive];
        g2_rank = new Galaxy[ng2];

        printf("%ld, receive rank = %d\n", nrows_to_receive, my_rank);

        cout << "rank = " << my_rank << ", length = " << nrows_to_receive << endl;

        ierr = MPI_Recv(g1_rank, nrows_to_receive, galaxytype, 0, tag2,
                            MPI_COMM_WORLD, &status);
    }

    // Broadcast constants and g2_rank.
    MPI_Bcast(&blen, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tdfac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rlcfac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beamfac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vlosmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(g2_rank, ng2, galaxytype, 0, MPI_COMM_WORLD);

    init.ll = new long[ng2];
    init.hoc = get3dlong(nhocells);
    init_mesh(init.ll, init.hoc, g2_rank, ng2, nhocells, blen);


    xc_bin_rank = get2dlong(nbins);
    corr2d(xc_bin_rank, g1_rank, g2_rank, init, rlim, nbins, nhocells, blen,
            half_range, nrows_to_receive, tdfac, rlcfac, beamfac, vlosmax);

    xc_bin = get2dlong(nbins);
    ierr = MPI_Reduce(&xc_bin_rank[0][0], &xc_bin[0][0], nbins * nbins, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Type_free(&galaxytype);

    // Write 2d array (cross correlation) to file.
    if (my_rank == 0)  write_xc(argv[2], xc_bin, nbins);

    delete [] g1_rank;
    delete [] g2_rank;

    free2dlong(xc_bin_rank, nbins);
//    free2dlong(xc_bin, nbins);  //no idea why it causes an error

    delete [] init.ll;
    free3dlong(init.hoc, nhocells);
    ierr = MPI_Finalize();
    return 0;
}

long read_Nbody(Galaxy* p, string filename, double peculiar, double gravfac, double* blen) {

    ifstream filein;
    long np, i;
    double redshift;

    filein.open(filename.c_str(), ios::in);

    if (filein.fail()) {
        cerr << "Error opening the file!" << endl;
        exit(-1);
    }
    else {
        filein >> np;
        filein >> redshift;
        filein >> *blen;

        cout << "Redshift = " << redshift << ", Number of galaxies = " << np << ", box size = "
            << *blen << endl;

        for (i = 0; i < np; i++)
            filein >> p[i].pos[0] >> p[i].pos[1] >> p[i].pos[2] >> p[i].vel[0] >>
                p[i].vel[1] >> p[i].vel[2] >> p[i].grav >> p[i].mass;

// Try to understand this, 107413474 against 107413475
        if (filein.eof())
            cout << "EOF reached. Done reading!" << endl;
        else
            cout << filein.tellg() << endl << "Error!" << endl;

        filein.close();
    }

    // Include peculiar velocities.
    if (peculiar > 0) {
        for (i = 0; i < np; i++)
            p[i].pos[2] = cwarp(p[i].pos[2] + p[i].vel[2] / 100., *blen);
        cout << "Peculiar velocities are included!" << endl;
    } else {
        cout << "Peculiar velocities are not included!" << endl;
    }

    // Include gravitation redshifts.
    if (gravfac > 0) {
        for (i = 0; i < np; i++)
            p[i].pos[2] = cwarp(p[i].pos[2] + p[i].grav * gravfac / 100., *blen);
        cout << "Gravitational redshifts are included and magnified by factor" << gravfac << endl;
    } else {
        cout << "Gravitational redshifts are not included!" << endl;
    }

    return np;
}

void write_xc(string filename, long** xc_bin, int nbins) {

    ofstream fileout;
    fileout.open(filename.c_str(), ios::out);

    for (int p = 0; p < nbins; p++) {
            for (int q = 0; q < nbins; q++) {
                fileout << xc_bin[p][q] << " ";
            }
            fileout << endl;
        }
    fileout.close();
}

 void swap(Galaxy* p1, Galaxy* p2) {
    Galaxy temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}

void quick_sort (Galaxy* p, long n) {
    long i, j;
    double pmass;
    if (n < 2)  return;
    pmass = p[n / 2].mass;
    for (i = 0, j = n - 1;; i++, j--) {
        while (p[i].mass > pmass)
            i++;
        while (pmass > p[j].mass)
            j--;
        if (i >= j)  break;
        swap(&p[i], &p[j]);
    }
    quick_sort(p, i);
    quick_sort(p + i, n - i);
}

template <typename TYPE>
TYPE cwarp(TYPE x, TYPE bound) {
    // Cell warp according to periodic b. c
    TYPE modulo = x;
    if (x >= bound) modulo = x - bound;
    if (x < 0) modulo = x + bound;
    return modulo;
}

double bwarp(double x, double box) {
    // box warp according to periodic b. c
    double modulo = x;
    if (x >= box / 2) modulo = x - box;
    if (x < - box / 2) modulo = x + box;
    return modulo;
}

long** get2dlong(int nbins) {
    // This func is used for 2d correlation func
    int i;
    long** table;
    table = new long*[nbins];
    for (i = 0; i < nbins; i++) {
        table[i] = new long[nbins];
    }
    return table;
}

long*** get3dlong(int nho) {
    // This func is used for chaining mesh
    int i, j;
    long*** table;
    table = new long**[nho];
    for (i = 0; i < nho; i++) {
        table[i] = new long*[nho];
        for (j = 0; j < nho; j++) {
            table[i][j] = new long[nho];
        }
    }
    return table;
}

void init_mesh(long* ll, long*** hoc, Galaxy* p, long plength, int nho, double blen) {
    int ix, iy, iz, ii, jj, kk;
    long i;

    // Initiate hoc to -1 for not missing the last point
    for (ii = 0; ii < nho; ii++)
        for (jj = 0; jj < nho; jj++)
            for (kk = 0; kk < nho; kk++)
                hoc[ii][jj][kk] = -1;

    for (i = 0; i < plength; i++) {
        ix = floor(((p[i].pos[0] - xoff) / blen * nho));
        iy = floor(((p[i].pos[1] - yoff) / blen * nho));
        iz = floor(((p[i].pos[2] - zoff) / blen * nho));

        ll[i] = hoc[ix][iy][iz];
        hoc[ix][iy][iz] = i;
    }
}

void corr2d(long **xc_bin, Galaxy *p1_rank, Galaxy *p2, struct ll_hoc init, double rlim,
              int nbins, int nhocells, double blen, double dis_f, long nrows, double tdfac,
              double rlcfac, double beamfac, double vlosmax) {
    long iq, i;
    int pp, qq, rr, p, q, r, ix, iy, iz;
    double xp1, yp1, zp1, vxp1, vyp1, vzp1, dx, dy, dz, dr, dvx, dvy, dvz, beta2, tdcontrib,
           probtokeep_lc, probtokeep_beam;
    int k_perp, k_para;
    int ncb = floor((rlim / blen) * (double)(nhocells)) + 1;

    for (iq = 0; iq < nrows; iq++) {
        if (iq % 100 == 0)  cout << iq;
        xp1 = p1_rank[iq].pos[0];
        yp1 = p1_rank[iq].pos[1];
        zp1 = p1_rank[iq].pos[2];

        vxp1 = p1_rank[iq].vel[0];
        vyp1 = p1_rank[iq].vel[1];
        vzp1 = p1_rank[iq].vel[2];

        ix = floor(((xp1 - xoff) / blen * nhocells));
        iy = floor(((yp1 - yoff) / blen * nhocells));
        iz = floor(((zp1 - zoff) / blen * nhocells));

        for (pp = ix - ncb; pp < ix + ncb; pp++) {
            p = cwarp(pp, nhocells);
            for (qq = iy - ncb; qq < iy + ncb; qq++) {
                q = cwarp(qq, nhocells);
                for (rr = iz - ncb; rr < iz + ncb; rr++) {
                    r = cwarp(rr, nhocells);
                    if (init.hoc[p][q][r] != -1) {
                        i = init.hoc[p][q][r];
                        while (1) {

                            dx = bwarp(p2[i].pos[0] - xp1, blen);
                            dy = bwarp(p2[i].pos[1] - yp1, blen);
                            dz = bwarp(p2[i].pos[2] - zp1, blen);

                            // Transverse doppler effect
                            // propotional to beta
                            if (tdfac > 0) {
                                dvx = p2[i].vel[0] - vxp1;
                                dvy = p2[i].vel[1] - vyp1;
                                dvz = p2[i].vel[2] - vzp1;

                                beta2 = (dvx * dvx + dvy * dvy + dvz * dvz) / 299792.458;
                                tdcontrib = -beta2 * tdfac / 100;    // 1/100 from km/s to Mpc/h
                                dz += tdcontrib;
                            }

                            if (rlcfac > 0) {
                                probtokeep_lc = 1.+(-p2[i].vel[2]-vlosmax)*rlcfac/299792.458;
                                if (probtokeep_lc > 1) probtokeep_lc = 1;
                                if (probtokeep_lc < 0) probtokeep_lc = 0;
                                dz = dz * probtokeep_lc;
                            }

                            if (beamfac > 0) {
                                probtokeep_beam = 1.+(-p2[i].vel[2]-vlosmax)*4*beamfac/299792.458;
                                if (probtokeep_beam > 1) probtokeep_beam = 1;
                                if (probtokeep_beam < 0) probtokeep_beam = 0;
                                dz = dz * probtokeep_beam;
                            }

                            k_para = floor((dz + dis_f) * nbins / 2 / dis_f);
                            if (k_para >= nbins || k_para < 0) {
                                printf("\n%d", k_para);
                                continue;
                            }
                            dr = sqrt(dx * dx + dy * dy);
                            k_perp = floor(dr * nbins / 2 / dis_f);
                            xc_bin[k_perp][k_para] += 1;
                            if (init.ll[i] != -1) {
                                i = init.ll[i];
                            }
                            else break;
                        }
                    }
                }
            }
        }
    }
}

void free2dlong(long** p, int np) {
    int i;
    for (i = 0; i < np; i++)
        delete [] p[i];
    delete [] p;
}

void free3dlong(long*** p, int np) {
    int i, j;
    for (i = 0; i < np; i++) {
        for (j = 0; j < np; j++) {
            delete [] p[i][j];
        }
        delete [] p[i];
    }
    delete p;
}
