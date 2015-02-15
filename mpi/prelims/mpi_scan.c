//mpi_sendrecv_replace.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ST 11

int main(int argc, char **argv) {
	int nproc, id, a1=0, a2=0;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	
	a1=id;

	MPI_Scan(&a1, &a2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	printf("my id: %d my a1: %d my a2: %d\n", id, a1, a2);  


	MPI_Finalize();
	return 0;
}