//mpi_status.c
#include <mpi.h>
#include <stdio.h>

#define ST 11

int main(int argc, char **argv) {
	int nproc, id, i;
	int a=0;
	MPI_Status status;
	if(MPI_Init(&argc, &argv) != MPI_SUCCESS) {
		printf("Err in init\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	} else {
		printf("Success in init\n");
	}
	if(MPI_Comm_size(MPI_COMM_WORLD, &nproc) != MPI_SUCCESS) {
		printf("Err in determining no. of proc\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	} else {
		printf("Success in determining no. of proc == %d\n", nproc);
	}

	if(MPI_Comm_rank(MPI_COMM_WORLD, &id) != MPI_SUCCESS) {
		printf("Error in determining rank\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	} else {
		printf("Success in determining rank, my rank == %d\n", id);
	}

	//checking if copy of above variables is created for each process
	printf("my address of a=%p\n", &a);

	if(id==0) {
		for(i = 1; i < nproc; ++i)
			if(MPI_Send(&i, 1, MPI_INT, i, ST, MPI_COMM_WORLD) != MPI_SUCCESS) {
				printf("Error in sending data from %d to %d\n", id, i);
				MPI_Abort(MPI_COMM_WORLD, 1);
			} else {
				printf("Success in sending data from %d to %d\n", id, i);
			}
	} else {
		if(MPI_Recv(&a, 1, MPI_INT, 0, ST, MPI_COMM_WORLD, &status) != MPI_SUCCESS) {
			printf("Error in receiving data from 0 to %d\n", id);
			MPI_Abort(MPI_COMM_WORLD, 1);
		} else {
			printf("Success in receiving data from 0 to %d\n", id);
			printf("Status: Source=%d Tag=%d Err=%d\n", status.MPI_SOURCE, status.MPI_TAG, status.MPI_ERROR);
		}
	}
	if(MPI_Finalize() != MPI_SUCCESS) {
		printf("Error in finalizing\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	} else {
		printf("Success in finalizing\n");
	}
	return 0;
}