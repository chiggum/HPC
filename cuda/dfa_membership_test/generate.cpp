#include <iostream>
#include <cstdlib>
#include <fstream>

int main(int argc, char **argv) {
	int m, n, len;
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	len = atoi(argv[3]);

	std::ofstream output1("data1");
	std::ofstream output2("data2");

	output1 << m << std::endl;
	output1 << n << std::endl;
	for(int i = 0; i < m; ++i) {
		for(int j = 0; j < n; ++j) {
			output1 << rand()%m << " ";
		}
		output1 << std::endl;
	}
	for(int i = 0; i < m; ++i) {
		if((1.0*rand())/RAND_MAX > 0.5)
			output1 << 1 << std::endl;
		else
			output1 << 0 << std::endl;
	}
	output1.close();

	output2 << rand()%m << std::endl;
	output2 << len << std::endl;
	for(int i = 0; i < len; ++i) {
		output2 << rand()%n << std::endl;
	}
	output2.close();
	return 0;
}