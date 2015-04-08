//mpi_scatter_gather.c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MAX_VAL 10


int main(int argc, char **argv) {
	int nproc, id, i;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	if(argc!=2) {
		printf("Usage: %s num_elem_per_proc", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	int num_elem_per_proc = atoi(argv[1]);
	int total_elem = num_elem_per_proc*nproc;
	int *full_arr = (int*)malloc(total_elem*sizeof(int));
	int *sub_arr = (int*)malloc(num_elem_per_proc*sizeof(int));
	if(id == 0) {
		for(i = 0; i < total_elem; ++i) {
			full_arr[i] = rand()%MAX_VAL;
			printf("%d ", full_arr[i]);
		}
		printf("\n");
	}
	MPI_Scatter(full_arr, num_elem_per_proc, MPI_INT, sub_arr, num_elem_per_proc, MPI_INT, 0, MPI_COMM_WORLD);

	int my_val = 0;
	for(i = 0; i <num_elem_per_proc; ++i) {
		my_val += sub_arr[i];
	}
	int *glob_arr;
	glob_arr = (int*)malloc(nproc*sizeof(int));
	MPI_Allgather(&my_val, 1, MPI_INT, glob_arr, 1, MPI_INT, MPI_COMM_WORLD);

	int glob_val = 0;
	for(i = 0; i <nproc; ++i) {
		glob_val += glob_arr[i];
	}
	printf("my id: %d Global value: %d\n", id, glob_val);

	MPI_Finalize();
	return 0;
}