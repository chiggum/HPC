#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_VAL 2

int main(int argc, char **argv) {
	if(argc != 4) {
		printf("./mat_gen orig_rows orig_cols output_file_name\n");
		exit(EXIT_FAILURE);
	}
	clock_t start, end;
	double cpu_time_used;
	start = clock();

	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int i, j;

	//write to output file
	FILE *fp = fopen(argv[3], "w");
	for(i = 0; i < m; ++i) {
		for(j = 0; j < n; ++j) {
			fprintf(fp, "%d ", rand()%MAX_VAL);
		}
		fprintf(fp, "\n");
	}
	for(i = 0; i < m; ++i) {
		for(j = 0; j < n; ++j) {
			fprintf(fp, "%d ", rand()%MAX_VAL);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Time taken: %fs\n", cpu_time_used);

	return 0;
}