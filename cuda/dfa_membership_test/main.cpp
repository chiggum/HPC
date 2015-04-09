#include <iostream>
#include <string>
#include <fstream>

#define cudaMemcpyHTD(dest, src, sizeInBytes) cudaMemcpy(dest, src, sizeInBytes, cudaMemcpyHostToDevice)
#define cudaMemcpyDTH(dest, src, sizeInBytes) cudaMemcpy(dest, src, sizeInBytes, cudaMemcpyDeviceToHost)
#define NUM_BLOCKS 512

//delta is an mXn matrix where m is the number of states and n is the 
//size of set of legal inputs
//returns final state achieved
int seqMembershipTestUtil(int *delta, int *input, int q0, int s, int e, int m, int n) {
	for(int i = s; i < e; ++i) {
		if(q0 >= m || input[i] >= n)
			return -1;
		q0 = delta[input[i] + n*q0];
	}
	return q0;
}

bool seqMembershipTest(int *delta, bool *finalStates, int *input, int q0, int len, int m, int n) {
	int fState = seqMembershipTestUtil(delta, input, q0, 0, len, m, n);
	if(fState == -1)
		return false;
	return finalStates[fState];
}




int main(int argc, char **argv) {

	/********************************
	Fetching Data
	********************************/
	int *delta, M, N;
	int q0, *inString, len;
	bool *finalStates;
	
	std::ifstream input(argv[1]);
	input >> M >> N;
	delta = new int[M*N];
	finalStates = new bool[M];
	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < N; ++j) {
			input >> delta[j+i*N];
		}
	}
	for(int i = 0; i < M; ++i) {
		input >> finalStates[i];
	}
	input.close();

	input.open(argv[2]);
	input >> q0;
	input >> len;
	inString = new int[N];
	for(int i = 0; i < len; ++i) {
		input >> inString[i];
	}
	input.close();
	/********************************
	Fetched Data
	********************************/
	/********************************
	Sequential Membership Test
	********************************/
	bool result = seqMembershipTest(delta, finalStates, inString, q0, len, M, N);
	if(result)
		std::cout << "Member" << std::endl;
	else
		std::cout << "Not a member" << std::endl;
	/********************************
	Sequential Membership Test Ends Here
	********************************/
	/********************************
	Parallel Membership Test
	********************************/

	return 0;
}