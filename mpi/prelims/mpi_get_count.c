//mpi_getcount.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ST 11

int main(int argc, char **argv) {
	int nproc, id, i;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if(id==0) {
		for(i = 1; i < nproc; ++i) {
			int *a = (int*)malloc((i+1)*sizeof(int));
			MPI_Send(a, i+1, MPI_INT, i, ST, MPI_COMM_WORLD);
			free(a);
			a=NULL;
		}
	} else {
		int cnt;
		int *a = (int*)malloc(id*sizeof(int));
		MPI_Recv(a, id, MPI_INT, 0, ST, MPI_COMM_WORLD, &status);
		MPI_Get_count(&status, MPI_INT, &cnt);
		printf("my id: %d No. of items received: %d\n", id, cnt);
	}
	MPI_Finalize();
	return 0;
}