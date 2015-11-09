#include "corrdefs2.h"
using namespace std;

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
        ix = floor(((p[i].pos[0] - XOFF) / blen * nho));
        iy = floor(((p[i].pos[1] - YOFF) / blen * nho));
        iz = floor(((p[i].pos[2] - ZOFF) / blen * nho));

        ll[i] = hoc[ix][iy][iz];
        hoc[ix][iy][iz] = i;
    }
}

void corr2d(long **xc_bin, Galaxy *p1_rank, long nrows, Galaxy *p2, long plength, long* ll,
        long*** hoc, double rlim, int nbins, int nhocells, double blen, double dis_f, double tdfac,
              double rlcfac, double beamfac, double vlosmax) {
    long iq, i;
    int pp, qq, rr, p, q, r, ix, iy, iz;
    double xp1, yp1, zp1, vxp1, vyp1, vzp1, dx, dy, dz, dr, dvx, dvy, dvz, beta2, tdcontrib,
           probtokeep_lc, probtokeep_beam;
    int k_perp, k_para;
    int ncb = floor((rlim / blen) * (double)(nhocells)) + 1;

    init_mesh(ll, hoc, p2, plength, nhocells, blen);

    for (iq = 0; iq < nrows; iq++) {
        if (iq % 100 == 0)  cout << iq;
        xp1 = p1_rank[iq].pos[0];
        yp1 = p1_rank[iq].pos[1];
        zp1 = p1_rank[iq].pos[2];

        vxp1 = p1_rank[iq].vel[0];
        vyp1 = p1_rank[iq].vel[1];
        vzp1 = p1_rank[iq].vel[2];

        ix = floor(((xp1 - XOFF) / blen * nhocells));
        iy = floor(((yp1 - YOFF) / blen * nhocells));
        iz = floor(((zp1 - ZOFF) / blen * nhocells));

        for (pp = ix - ncb; pp < ix + ncb; pp++) {
            p = cwarp(pp, nhocells);
            for (qq = iy - ncb; qq < iy + ncb; qq++) {
                q = cwarp(qq, nhocells);
                for (rr = iz - ncb; rr < iz + ncb; rr++) {
                    r = cwarp(rr, nhocells);
                    if (hoc[p][q][r] != -1) {
                        i = hoc[p][q][r];
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
                            if (ll[i] != -1) {
                                i = ll[i];
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
