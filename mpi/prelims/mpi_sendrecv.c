//mpi_sendrecv.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ST 11

int main(int argc, char **argv) {
	int nproc, id, a1=0, a2=0, left, right;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	
	a1=id;
	right = (id + 1) % nproc;
    left = id - 1;
    if (left < 0)
        left = nproc - 1;

	MPI_Sendrecv(&a1, 1, MPI_INT, left, ST, &a2, 1, MPI_INT, right, ST, MPI_COMM_WORLD, &status);
	printf("my id: %d my a1: %d a2: %d\n", id, a1, a2);  


	MPI_Finalize();
	return 0;
}