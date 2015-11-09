#include "mpi.h"
#include "corrdefs2.h"

int main(int argc, char **argv) {

    long i, ng1, ng2;
    long **xc_bin_rank, **xc_bin;
    double half_range = 200., rlim = 100., blen, tdfac, vlosmax, beamfac, rlcfac;
    int nhocells = 1000, nbins = 400;
    long *ll, ***hoc;
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

            ierr = MPI_Send(&ng2, 1, MPI_LONG, irank, TAG3,
                            MPI_COMM_WORLD);
            ierr = MPI_Send(&nrows_to_send, 1, MPI_LONG, irank, TAG1,
                            MPI_COMM_WORLD);
            ierr = MPI_Send((galaxy+start_row), nrows_to_send, galaxytype, irank, TAG2,
                            MPI_COMM_WORLD);
         }

        nrows_to_receive = interval;
        delete [] galaxy;
    }

    else {

        ierr = MPI_Recv(&nrows_to_receive, 1, MPI_LONG, 0, TAG1,
                        MPI_COMM_WORLD, &status);
        ierr = MPI_Recv(&ng2, 1, MPI_LONG, 0, TAG3,
                        MPI_COMM_WORLD, &status);
        g1_rank = new Galaxy[nrows_to_receive];
        g2_rank = new Galaxy[ng2];

        printf("%ld, receive rank = %d\n", nrows_to_receive, my_rank);

        cout << "rank = " << my_rank << ", length = " << nrows_to_receive << endl;

        ierr = MPI_Recv(g1_rank, nrows_to_receive, galaxytype, 0, TAG2,
                            MPI_COMM_WORLD, &status);
    }

    // Broadcast constants and g2_rank.
    MPI_Bcast(&blen, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tdfac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rlcfac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beamfac, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vlosmax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(g2_rank, ng2, galaxytype, 0, MPI_COMM_WORLD);

    ll = new long[ng2];
    hoc = get3dlong(nhocells);

    xc_bin_rank = get2dlong(nbins);
    corr2d(xc_bin_rank, g1_rank, nrows_to_receive, g2_rank, ng2, ll, hoc, rlim, nbins, nhocells, blen,
            half_range, tdfac, rlcfac, beamfac, vlosmax);

    xc_bin = get2dlong(nbins);
    ierr = MPI_Reduce(&xc_bin_rank[0][0], &xc_bin[0][0], nbins * nbins, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Type_free(&galaxytype);

    // Write 2d array (cross correlation) to file.
    if (my_rank == 0)  write_xc(argv[2], xc_bin, nbins);

    delete [] g1_rank;
    delete [] g2_rank;

    free2dlong(xc_bin_rank, nbins);
    delete [] ll;
    free3dlong(hoc, nhocells);
    ierr = MPI_Finalize();
    return 0;
}
