/*
input matrix must be square and num proc must be multiple of rows to decompose
ASSUMING SECOND MATRIX IS ALREADY TRANSPOSED
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int main(int argc, char **argv) {

	if(argc != 5) {
		printf("Usage: %s input_file mat_dim num_rows_decompose output_file\n", argv[0]);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	MPI_Status status;
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int i, j, k, ret;

	int dim = atoi(argv[2]);
	int nrow = atoi(argv[3]);
	assert(dim%nrow==0);
	assert(size == dim/nrow);

	int *mat, *mat_, *part_mat, *part_mat_;
	if(rank == 0) {
		FILE *fp = fopen(argv[1], "r");
		mat = (int*)malloc(dim*dim*sizeof(int));
		mat_ = (int*)malloc(dim*dim*sizeof(int));
		for(i = 0; i < dim; ++i) {
			for(j = 0; j < dim; ++j) {
				ret=fscanf(fp, "%d", &mat[j+i*dim]);
			}
		}
		for(i = 0; i < dim; ++i) {
			for(j = 0; j < dim; ++j) {
				ret=fscanf(fp, "%d", &mat_[j+i*dim]);
			}
		}
		fclose(fp);
	}

	part_mat = (int*)malloc(nrow*dim*sizeof(int));
	part_mat_ = (int*)malloc(nrow*dim*sizeof(int));

	MPI_Scatter(mat, nrow*dim, MPI_INT, part_mat, nrow*dim, MPI_INT, 0, MPI_COMM_WORLD);
	
	for(i = 0; i < nrow; ++i) {
		for(j = 0; j < dim; ++j) {
			int sum = 0;
			for(k = 0; k < dim; ++k) {
				sum += part_mat[k+i*dim]*mat_[j+dim*k];
			}
			part_mat_[j+i*dim] = sum;
		}
	}

	MPI_Gather(part_mat_, nrow*dim, MPI_INT, mat, nrow*dim, MPI_INT, 0, MPI_COMM_WORLD);

	if(rank == 0) {
		FILE *fp_ = fopen(argv[4], "w");
		mat = (int*)malloc(dim*dim*sizeof(int));
		mat_ = (int*)malloc(dim*dim*sizeof(int));
		for(i = 0; i < dim; ++i) {
			for(j = 0; j < dim; ++j) {
				fprintf(fp_, "%d ", mat[j+i*dim]);
			}
			fprintf(fp_, "\n");
		}
		fclose(fp_);
	}

	MPI_Finalize();

	return 0;
}