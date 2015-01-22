/*
Lesson learnt:
-Use of broadcast: if it required for a process to pass same buffer to all the processors: use broadcast instead of send and receive
Floyd-Warshall Algorithm:
Time complexity: N^3/P + (MESSAGE PASSING TIME)
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ST 10
#define RC 11
#define INF 1000

#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))

int getBR(int q, int nprocs) {
	return (int)(q/nprocs);
}

int main(int argc, char **argv) {
	int numV = atoi(argv[1]), i, j, k, l, ret;
	int *dist;
	//ASSUMING numV = k * numprocs
	int nprocs, id;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	
	int part_rows = numV/nprocs;
	int part_size = (part_rows)*numV;
	int *part_dist, *part_row;
	
	if(id==0) {
		int *dist = (int*)malloc(numV*numV*sizeof(int));
		for(i = 0; i < numV; ++i) {
			for(j = 0; j < numV; ++j) {
				ret = scanf("%d", &dist[j+i*numV]);
			}
		}
	
		//send data 
		for(i = 1; i < nprocs; ++i) {
			MPI_Send(dist + part_size*i, part_size, MPI_INT, i, ST, MPI_COMM_WORLD);
		}

		//compute my part
		part_row = (int*)malloc(numV*sizeof(int));

		int local_i_start = id*part_rows;
		int local_i_end = (id+1)*part_rows;
		
		for(k = 0; k < numV; ++k) {

			if(k >= local_i_start && k < local_i_end) {
				for(l = 0; l < numV; ++l)
					part_row[l] = dist[l+k*numV];
			}
			MPI_Bcast(part_row, numV, MPI_INT, getBR(k, part_rows), MPI_COMM_WORLD);
			for(i = local_i_start; i < local_i_end; ++i) {
				for(j = 0; j < numV; ++j) {
					dist[j+i*numV] = min(dist[j+i*numV], dist[k+i*numV]+part_row[j]); 
				}
			}
		}


		//finally process 0 receive and merge data
		for(i = 1; i < nprocs; ++i) {
			MPI_Recv(dist + part_size*i, part_size, MPI_INT, i, RC, MPI_COMM_WORLD, &status);
		}

		//print distances
		for(i = 0; i < numV; ++i) {
			for(j = 0; j < numV; ++j) {
				printf("%d, ", dist[j+i*numV]);
			}
			printf("\n");
		}

	} else {
		part_dist  = (int*)malloc(numV*numV*sizeof(int));
		part_row = (int*)malloc(numV*sizeof(int));

		MPI_Recv(part_dist+id*part_size, part_size, MPI_INT, 0, ST, MPI_COMM_WORLD, &status);
		
		int local_i_start = id*part_rows;
		int local_i_end = (id+1)*part_rows;
		
		for(k = 0; k < numV; ++k) {

			if(k >= local_i_start && k < local_i_end) {
				for(l = 0; l < numV; ++l)
					part_row[l] = part_dist[l+k*numV];
			}
			
			MPI_Bcast(part_row, numV, MPI_INT, getBR(k, part_rows), MPI_COMM_WORLD);
			
			for(i = local_i_start; i < local_i_end; ++i) {
				for(j = 0; j < numV; ++j) {
					part_dist[j+i*numV] = min(part_dist[j+i*numV], part_dist[k+i*numV]+part_row[j]); 
				}
			}
		}
		MPI_Send(part_dist +id*part_size, part_size, MPI_INT, 0, RC, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	free(dist);
	free(part_row);
	free(part_dist);
	dist = NULL;
	part_row = NULL;
	part_dist = NULL;
	return 0;
}
