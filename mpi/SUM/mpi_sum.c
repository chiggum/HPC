#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define ST 13
#define RC 14


int main(int argc, char **argv) {
	if(argc != 2) {
		printf("Usage: ./mpi_sum num_int  {to sum from 1 to num_int}\n");
		exit(EXIT_FAILURE);
	}
	MPI_Status status;
	int NUM = atoi(argv[1]);
	int nproc, myid;
	int *data = (int*)malloc(NUM*sizeof(int));
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	int seg_size = NUM/nproc;
	int rem = NUM - nproc*seg_size;
	int *seg = (int*)malloc(seg_size*sizeof(int));


	if(myid == 0) {
		//generating data
		int i;
		for(i = 0; i < NUM; ++i) {
			data[i] = i+1;
		}
		//assign segments of data to each processor
		for(i = 1; i < nproc; ++i) {
			MPI_Send(data + (i-1)*seg_size, seg_size, MPI_INT,i, ST, MPI_COMM_WORLD);
		}
		//sum my segment (parent to do: seg_size + remaining elem sum)
		int seg_sum = 0;
		for(i = 0; i < rem + seg_size; ++i) {
			seg_sum += data[(nproc-1)*seg_size + i];
		}
		//receive segments from other processes and sum them
		int recv_seg_sum;
		for(i = 1; i < nproc; ++i) {
			MPI_Recv(&recv_seg_sum, 1, MPI_INT, i, RC, MPI_COMM_WORLD, &status);
			seg_sum += recv_seg_sum;
		}
		printf("Sum: %d\n", seg_sum);

	} else {
		//receive my segment
		MPI_Recv(seg, seg_size, MPI_INT, 0, ST, MPI_COMM_WORLD, &status);
		int seg_sum = 0;
		int i;
		//sum my segment
		for(i = 0; i < seg_size; ++i) {
			seg_sum+=seg[i];
		}
		//send my segment
		MPI_Send(&seg_sum, 1, MPI_INT, 0, RC, MPI_COMM_WORLD);
	}
	MPI_Finalize();
	return 0;
}