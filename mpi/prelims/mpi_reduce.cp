//mpi_reduce.c
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_VAL 10

int main(int argc, char **argv) {
	int nproc, id, i;
	int num_elem = atoi(argv[1]);
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	if(argc != 2) {
		printf("Usage: %s num_elem_per_process\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	int *rand_num = (int*)malloc(num_elem*sizeof(int));
	for(i = 0; i < num_elem; ++i) {
		rand_num[i] = rand()%MAX_VAL;
	}

	int my_val = 1;
	for(i = 0; i < num_elem; ++i) {
		my_val = my_val * rand_num[i];
	}

	printf("my value: %d\n", my_val);

	int glob_val;

	MPI_Reduce(&my_val, &glob_val, 1, MPI_INT, MPI_PROD, 0, MPI_COMM_WORLD);

	if(id==0) {
		printf("Global value: %d\n", glob_val);
	}
	MPI_Finalize();
	return 0;
}