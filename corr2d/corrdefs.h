#include <iostream>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include <sstream>
using namespace std;

#define XOFF 0
#define YOFF 0
#define ZOFF 0
#define TAG1 123
#define TAG2 1008
#define TAG3 408

typedef struct Galaxy_props{
    double pos[3];  // positions in Mpc/h
    double vel[3];  // peculiar velocities in km/s
    double grav;    // gravitational redshift in km/s (wrt the mean level)
    double mass;    // mass in gadget units (10^10 m_sun/h)
} Galaxy;

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
void corr2d(long **xc_bin, Galaxy *p1_rank, long nrows, Galaxy *p2, long plength, long* ll,
        long*** hoc, double rlim, int nbins, int nhocells, double blen, double dis_f, double tdfac,
              double rlcfac, double beamfac, double vlosmax);
