#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
	if(argc != 3) {
		printf("./seq_FW num_vertices output_file_name\n");
		exit(EXIT_FAILURE);
	}
	clock_t start, end;
	double cpu_time_used;
	start = clock();

	int numV = atoi(argv[1]);
	int **dist = (int**)malloc(numV*sizeof(int*));
	int i, j, k;

	//write to output file
	FILE *fp = fopen(argv[2], "w");
	for(i = 0; i < numV; ++i) {
		for(j = 0; j < numV; ++j) {
			fprintf(fp, "%d ", rand()%numV + 1);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Time taken: %fs\n", cpu_time_used);

	return 0;
}