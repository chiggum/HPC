/*
Lesson learnt:
-Use of broadcast: if it is required for a process to pass same buffer to all the processors: use broadcast instead of send and receive
-Though both techniques are same but code is compact in case of broadcast
Floyd-Warshall Algorithm:
Time complexity: N^3/P + (MESSAGE PASSING TIME)
*/

#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>

#define ST 10
#define RC 11

#define min(X,Y) ((X) < (Y) ? (X) : (Y))

int getBR(int q, int nprocs) {
	return (int)(q/nprocs);
}

int main(int argc, char **argv) {
	if(argc != 4) {
		printf("./seq_FW num_vertices file_containing_input_data output_file_name\n");
		exit(EXIT_FAILURE);
	}

	clock_t start, end;
	double cpu_time_used;
	start = clock();

	int numV = atoi(argv[1]), i, j, k, l, ret;
	int *dist;
	int nprocs, id;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	int error = 0;

	if(numV%nprocs != 0) {
		error = 1;
	}
	
	int part_rows = numV/nprocs;
	int part_size = (part_rows)*numV;
	int *part_dist, *part_row;
	if(!error) {
		if(id==0) {
			int *dist = (int*)malloc(numV*numV*sizeof(int));

			//read input matrix
			FILE *fp = fopen(argv[2], "r");
			for(i = 0; i < numV; ++i) {
				for(j = 0; j < numV; ++j) {
					ret = fscanf(fp, "%d", &dist[j+i*numV]);
				}
			}
			fclose(fp);

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


			//finally process 0 receives and merges data
			for(i = 1; i < nprocs; ++i) {
				MPI_Recv(dist + part_size*i, part_size, MPI_INT, i, RC, MPI_COMM_WORLD, &status);
			}

			//output data in output file
			fp = fopen(argv[3], "w");
			for(i = 0; i < numV; ++i) {
				for(j = 0; j < numV; ++j) {
					fprintf(fp, "%d ", dist[j+i*numV]);
				}
				fprintf(fp, "\n");
			}
			fclose(fp);
			free(dist);
			free(part_row);
			dist = NULL;
			part_row = NULL;

			end = clock();
			cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
			printf("Time taken: %fs\n", cpu_time_used);

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
			free(part_row);
			free(part_dist);
			part_row = NULL;
			part_dist = NULL;
		}
	}
	if(id == 0 && error) {
		printf("number of vertices not a multiple of processors allotted\n");
	}

	MPI_Finalize();
	return 0;
}
