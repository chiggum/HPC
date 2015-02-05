#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define min(X,Y) ((X) < (Y) ? (X) : (Y))

int main(int argc, char **argv) {
	if(argc != 4) {
		printf("./seq_FW num_vertices file_containing_input_data output_file_name\n");
		exit(EXIT_FAILURE);
	}
	clock_t start, end;
	double cpu_time_used;
	start = clock();

	int numV = atoi(argv[1]);
	int **dist = (int**)malloc(numV*sizeof(int*));
	int i, j, k;
	//reading matrix input data
	FILE *fp = fopen(argv[2], "r");
	for(i = 0; i < numV; ++i) {
		dist[i] = (int*)malloc(numV*sizeof(int));
		for(j = 0; j < numV; ++j) {
			fscanf(fp, "%d", &dist[i][j]);
		}
	}
	fclose(fp);
	//floyd warshall algorithm
	for(i = 0; i < numV; ++i) {
		for(j = 0; j < numV; ++j) {
			for(k = 0; k < numV; ++k) {
				dist[j][k] = min(dist[j][k], dist[j][i]+dist[i][k]);
			}
		}
	}
	//write to output file
	fp = fopen(argv[3], "w");
	for(i = 0; i < numV; ++i) {
		for(j = 0; j < numV; ++j) {
			fprintf(fp, "%d ", dist[i][j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	for(i = 0; i < numV; ++i) {
		free(dist[i]);
	}
	free(dist);

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Time taken: %fs\n", cpu_time_used);

	return 0;
}