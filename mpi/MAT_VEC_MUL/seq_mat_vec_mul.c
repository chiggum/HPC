#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main(int argc, char **argv) {
	if(argc != 7) {
		printf("./seq_mat_vec_mul orig_rows orig_cols part_rows part_cols file_containing_input_data output_file_name\n");
		exit(EXIT_FAILURE);
	}
	clock_t start, end;
	double cpu_time_used;
	start = clock();

	int m, n, m_p, n_p, i, j, k, l, ret;
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	m_p = atoi(argv[3]);
	n_p = atoi(argv[4]);

	FILE *fp = fopen(argv[5], "r");

	int *mat = (int*)malloc(m*n*sizeof(int));
	int *vec = (int*)malloc(n*sizeof(int));
	int *res = (int*)malloc(m*sizeof(int));

	for(i = 0; i < m; ++i) {
		for(j = 0; j < n; ++j) {
			ret=fscanf(fp, "%d", &mat[j+i*n]);
		}
	}

	for(i = 0; i < n; ++i) {
		ret=fscanf(fp, "%d", &vec[i]);
	}

	fclose(fp);

	for(i = 0; i < m; ++i)
		res[i] = 0;


	for(i = 0; i < m; i+=m_p) {
		for(j = 0; j < n; j+=n_p) {
			int *my_part_vec = (int*)malloc(n_p*sizeof(int));
			int *my_part_mat = (int*)malloc(n_p*m_p*sizeof(int));

			for(k = 0; k < m_p; ++k) {
				for(l = 0; l < n_p; ++l) {
					my_part_mat[l + k*n_p] = mat[j+l + (i+k)*n];
				}
			}
			for(k = 0; k < n_p; ++k)
				my_part_vec[k] = vec[j+k];

			int *res_part = (int*)malloc(m_p*sizeof(int));

			for(k = 0; k < m_p; ++k) {
				int sum = 0;
				for(l = 0; l < n_p; ++l) {
					sum += my_part_mat[l+k*n_p]*my_part_vec[l];
				}
				res_part[k] = sum;
			}

			for(k = 0; k < m_p; ++k)
				res[i+k] += res_part[k];

			free(res_part);
			free(my_part_mat);
			free(my_part_vec);
		}
	}


	fp = fopen(argv[6], "w");
	for(i = 0; i < m; ++i) {
		fprintf(fp, "%d\n", res[i]);
	}
	fclose(fp);
	free(mat);
	free(vec);
	free(res);

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Time taken: %fs\n", cpu_time_used);

	return 0;
}