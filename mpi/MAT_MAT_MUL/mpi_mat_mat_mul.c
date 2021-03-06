#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>

#define ST 11
#define ST2 13
#define RC 12

int main(int argc, char **argv) {

	if(argc != 7) {
		printf("./mpi_mat_vec_mul orig_rows orig_cols part_rows part_cols file_containing_input_data output_file_name\n");
		exit(EXIT_FAILURE);
	}
	clock_t start, end;
	double cpu_time_used;
	start = clock();


	MPI_Status status;
	int nproc, id;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	int m, n, m_p, n_p, i, j, k, ret, l;
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	m_p = atoi(argv[3]);
	n_p = atoi(argv[4]);
	int r = m / m_p;
	int c = n / n_p;
	assert(nproc == r*c);
	assert(m == n);
	assert(m_p == n_p);

	if(id == 0) {
		FILE *fp = fopen(argv[5], "r");

		int *mat = (int*)malloc(m*n*sizeof(int));
		int *mat_ = (int*)malloc(m*n*sizeof(int));

		for(i = 0; i < m; ++i) {
			for(j = 0; j < n; ++j) {
				ret=fscanf(fp, "%d", &mat[j+i*n]);
			}
		}

		for(i = 0; i < m; ++i) {
			for(j = 0; j < n; ++j) {
				ret=fscanf(fp, "%d", &mat_[j+i*n]);
			}
		}

		int *part_mat = (int*)malloc(n_p*m_p*sizeof(int));
		int *part_mat_ = (int*)malloc(n_p*m_p*sizeof(int));

		for(i = 0; i < m; i+=m_p) {
			for(j = 0; j < n; j+=n_p) {
				if(i==0&&j==0)
					continue;
				for(k = 0; k < m_p; ++k) {
					for(l = 0; l < n_p; ++l) {
						part_mat[l + k*n_p] = mat[j+l + (i+k)*n];
					}
				}
				for(k = 0; k < m_p; ++k) {
					for(l = 0; l < n_p; ++l) {
						part_mat_[l + k*n_p] = mat_[i+l + (j+k)*n];
					}
				}
				MPI_Send(part_mat, m_p*n_p, MPI_INT, j/n_p + c*(i/m_p), ST, MPI_COMM_WORLD);
				MPI_Send(part_mat_, m_p*n_p, MPI_INT, j/n_p + c*(i/m_p), ST, MPI_COMM_WORLD);
			}
		}

		fclose(fp);

		int *my_mat_part = (int*)malloc(m_p*n_p*sizeof(int));
		int *my_mat_part_ = (int*)malloc(m_p*n_p*sizeof(int));
		for(i = 0; i < m_p; ++i) {
			for(j = 0; j < n_p; ++j) {
				my_mat_part[j+i*n_p] = mat[j+i*n];
			}
		}
		for(i = 0; i < m_p; ++i) {
			for(j = 0; j < n_p; ++j) {
				my_mat_part_[j+i*n_p] = mat_[j+i*n];
			}
		}

		int *my_mat_res = (int*)malloc(m_p*n_p*sizeof(int));
		for(i = 0; i < m_p; ++i) {
			for(j = 0; j < n_p; ++j) {
				int sum = 0;
				for(k = 0; k < n_p; ++k)
					sum += my_mat_part[k+i*n_p]*my_mat_part_[j+k*n_p];
				my_mat_res[j+i*n_p] = sum;
			}
		}



		int *result = (int*)malloc(m*n*sizeof(int));
		for(i = 0; i < m*n; ++i)
			result[i] = 0;
		for(k = 0; k < m_p; ++k) {
			for(l = 0; l < n_p; ++l) {
				result[l+k*n] += my_mat_res[l+k*n_p];
			}
		}

		for(i = 0; i < m; i+=m_p) {
			for(j = 0; j < n; j+=n_p) {
				if(i==0&&j==0) {
					continue;
				}
				MPI_Recv(my_mat_res, m_p*n_p, MPI_INT, j/n_p + c*(i/m_p), RC, MPI_COMM_WORLD, &status);
				for(k = 0; k < m_p; ++k) {
					for(l = 0; l < n_p; ++l) {
						result[j+l+(i+k)*n] += my_mat_res[l+k*n_p];
					}
				}
			}
		}

		fp = fopen(argv[6], "w");
		for(k = 0; k < m; ++k) {
			for(l = 0; l < n; ++l) {
				fprintf(fp, "%d ", result[l+k*n]);
			}
			fprintf(fp, "\n");
		}
		fclose(fp);
		free(my_mat_part_);
		free(my_mat_res);
		free(result);
		free(mat);
		free(mat_);
		free(my_mat_part);
		free(part_mat);
		free(part_mat_);
		end = clock();
		cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
		printf("Time taken: %fs\n", cpu_time_used);

	} else {
		int *my_mat_part = (int*)malloc(m_p*n_p*sizeof(int));
		int *my_mat_part_ = (int*)malloc(m_p*n_p*sizeof(int));
		MPI_Recv(my_mat_part, m_p*n_p, MPI_INT, 0, ST, MPI_COMM_WORLD, &status);
		MPI_Recv(my_mat_part_, m_p*n_p, MPI_INT, 0, ST, MPI_COMM_WORLD, &status);
		
		int *my_mat_res = (int*)malloc(m_p*n_p*sizeof(int));
		for(i = 0; i < m_p; ++i) {
			for(j = 0; j < n_p; ++j) {
				int sum = 0;
				for(k = 0; k < n_p; ++k)
					sum += my_mat_part[k+i*n_p]*my_mat_part_[j+k*n_p];
				my_mat_res[j+i*n_p] = sum;
			}
		}
		MPI_Send(my_mat_res, m_p*n_p, MPI_INT, 0, RC, MPI_COMM_WORLD);
		free(my_mat_res);
		free(my_mat_part);
		free(my_mat_part_);
	}
	MPI_Finalize();
	return 0;
}