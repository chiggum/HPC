#include <iostream>
#include <string>
#include <cstdlib>
#include <fstream>

#define cudaMemcpyHTD(dest, src, sizeInBytes) cudaMemcpy(dest, src, sizeInBytes, cudaMemcpyHostToDevice)
#define cudaMemcpyDTH(dest, src, sizeInBytes) cudaMemcpy(dest, src, sizeInBytes, cudaMemcpyDeviceToHost)

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
	std::cout << "Final State: " <<  fState << std::endl;
	return finalStates[fState];
}

__global__ void assignValues(int *fStates, int m) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx >= m)
		return;
	fStates[idx] = idx;
}

__global__ void match(int *delta, int *input, int *fStates, int start_, int end_, int m, int n, int p) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx >= m)
		return;
	for(int i = start_; i < end_; ++i) {
		fStates[idx+p*m] = delta[input[i] + fStates[idx+p*m]*n];
	}
}

__global__ void specDFAMatching(int *delta, int *input, int *fStates, int q0, int len, int m, int n) {
	long long int idx = threadIdx.x + blockIdx.x*blockDim.x;
	long long int totalThreads = blockDim.x*gridDim.x;
	/*
	Dynamic parallelism
	dim3 threadsPerBlock(512);
	dim3 numBlocks((m-1)/threadsPerBlock.x + 1);
	assignValues<<<numBlocks, threadsPerBlock>>>(fStates, m);
	*/

	for(int i = 0; i < m; ++i) {
		fStates[i + idx*m] = i;
	}
	double L0 = (1.0*m*len)/(m+totalThreads-1);
	long long int start_=0, end_=len;
	if(idx!=0)
		start_ = (L0 + ((idx-1.0)*L0)/m);
	end_ = (L0 + (1.0*idx*L0)/m);
	if(end_ > len) {
		end_ = len;
	}
	
	//printf("idx: %d start %d end %d\n", idx, start_, end_);
	if(start_ > end_ || end_ < 0 || start_ < 0)
		return;
	if(idx == 0) {
		for(long long int i = start_; i < end_; ++i) {
			fStates[idx*m] = delta[input[i] + fStates[idx*m]*n];
		}
	} else {
		/*
		Dynamic parallelism
		match<<<numBlocks, threadsPerBlock>>>(delta, input, fStates, start_, end_, m, n, idx);
		*/
		for(int i = 0; i < m; ++i) {
			for(long long int j = start_; j < end_; ++j) {
				fStates[i+idx*m] = delta[input[j] + fStates[i+idx*m]*n];
			}
		}
	}
}

bool parMembershipTest(int *delta, bool *finalStates, int *input, int q0, int len, int m, int n) {
	int *d_delta, *d_input, *d_fStates, *h_fStates;
	int NUMBLOCKS = 1024*32;		// this value can be tuned
	int BLOCKSIZE = 1024;	// this value can be tuned
	dim3 threadsPerBlock(BLOCKSIZE);
	dim3 numBlocks(NUMBLOCKS);
	int totalThreads = BLOCKSIZE*NUMBLOCKS;
	
	clock_t start, end;
	start = clock();
	
	/*********************
	Allocate memory on device
	**********************/
	cudaMalloc((void**)&d_delta, m*n*sizeof(int));
	cudaMalloc((void**)&d_input, len*sizeof(int));
	cudaMalloc((void**)&d_fStates, totalThreads*m*sizeof(int));
	//host
	h_fStates = new int[totalThreads*m*sizeof(int)];

	/*********************
	Transfer data from host to device
	**********************/
	cudaMemcpyHTD(d_delta, delta, m*n*sizeof(int));
	cudaMemcpyHTD(d_input, input, len*sizeof(int));
	cudaMemset(d_fStates, 0, totalThreads*m*sizeof(int));

	/********************
	Launch kernel
	*********************/
	specDFAMatching<<<numBlocks, threadsPerBlock>>>(d_delta, d_input, d_fStates, q0, len, m, n);

	/*******************
	Transfer back data from device to host
	*******************/
	cudaMemcpyDTH(h_fStates, d_fStates, totalThreads*m*sizeof(int));
	/*******************
	Free Memory
	********************/
	cudaFree(d_delta);
	cudaFree(d_input);
	cudaFree(d_fStates);


	std::cout << "Processing done!" << std::endl;
	end = clock();
	double cpuTime = ((double) (end - start)) / CLOCKS_PER_SEC;
	std::cout << "Time by parallel algo: " <<  cpuTime << std::endl;
	
	int fState = q0;
	for(int i = 0; i < totalThreads; ++i) {
		fState = h_fStates[fState + i*m];
	}
	std::cout << "Finale State: " << fState << std::endl;
	return finalStates[fState];
}



int main(int argc, char **argv) {

	/********************************
	Fetching Data
	********************************/
	srand(time(NULL));
	int *delta, M=atoi(argv[1]), N=atoi(argv[2]);
	int q0=0, *inString, len=atoi(argv[3]);
	bool *finalStates;
	std::cout << "Length of input string " << len << std::endl;
	
	delta = new int[M*N];
	finalStates = new bool[M];
	for(int i = 0; i < M; ++i) {
		for(int j = 0; j < N; ++j) {
			delta[j+i*N] = rand()%M;
		}
	}
	for(int i = 0; i < M; ++i) {
		if((1.0*rand())/RAND_MAX > 0.5)
			finalStates[i] = true;
		else
			finalStates[i] = false;
	}

	inString = new int[len];
	for(int i = 0; i < len; ++i) {
		inString[i] = (rand()%N);
	}
	/********************************
	Fetched Data
	********************************/
	/********************************
	Sequential Membership Test
	********************************/
	clock_t start, end;
	start = clock();
	bool result = seqMembershipTest(delta, finalStates, inString, q0, len, M, N);
	end = clock();
	double cpuTime = ((double) (end - start)) / CLOCKS_PER_SEC;
	std::cout << "Time by sequential algo: " <<  cpuTime << std::endl;
	if(result)
		std::cout << "Member" << std::endl;
	else
		std::cout << "Not a member" << std::endl;

	/********************************
	Parallel Membership Test
	********************************/
	result = parMembershipTest(delta, finalStates, inString, q0, len, M, N);
	if(result)
		std::cout << "Member" << std::endl;
	else
		std::cout << "Not a member" << std::endl;

	return 0;
}
