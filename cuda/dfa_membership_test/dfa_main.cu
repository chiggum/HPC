/*
DFA Membership test in SIMT Environment
Author: Dhruv Kohli
Mentor: Prof. Kalpesh Kapoor

Reference:
"A Speculative Parallel DFA Membership Test for Multicore, SIMD and Cloud Computing Environments" Yousun Ko, Minyoung Jung, Yo-Sub Han, Bernd Burgstaller
*/

#include <cstdlib>
#include <cstdio>

#define cudaMemcpyHTD(dest, src, sizeInBytes) cudaMemcpy(dest, src, sizeInBytes, cudaMemcpyHostToDevice)
#define cudaMemcpyDTH(dest, src, sizeInBytes) cudaMemcpy(dest, src, sizeInBytes, cudaMemcpyDeviceToHost)

#define BLOCKSIZE 1024

__global__ void specDFAMatching(int M, int N, int len, int *delta, char *input, int *fStates, int q0, int maxThreads) {
	unsigned int idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=maxThreads)
		return;
	long long unsigned int i, j;
	if(idx==0) {
		int fst=q0;
		long long unsigned int l=0;
		long long unsigned int r=(M)*(len/(M+maxThreads-1))-1;
		for(i=l; i<=r; ++i) {
			fst=delta[(int)(input[i]-'0')+fst*N];
		}
		fStates[q0+idx*M]=fst;
	} else {
		long long unsigned int l=(M)*(len/(M+maxThreads-1)) + (idx-1)*(len/(M+maxThreads-1));
		long long unsigned int r=l+(len/(M+maxThreads-1))-1;
		for(j=0; j<M; ++j) {
			int fst=j;
			for(i=l; i<=r; ++i) {
				fst=delta[(int)(input[i]-'0')+fst*N];
			}
			fStates[j+idx*M]=fst;
		}
	}
}

__global__ void finalStateUsingRedn(int M, int *fStates, int q0, unsigned int maxThreads, int *d_check) {
	unsigned int idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=maxThreads)
		return;
	for(unsigned int s=1; s<maxThreads; s*=2) {
		if(idx%(2*s)==0) {
			if((idx+s)<maxThreads) {
				for(int i=0; i<M; ++i) {
					int x=fStates[i+idx*M];
					fStates[i+idx*M]=fStates[x+(idx+s)*M];
				}
				d_check[idx+s]++;
			}
		}
		__syncthreads();
	}
}

int SIMTMembershipTest(int M, int N, int len, int *h_delta, bool *h_finalState, char *h_input, int q0, int K=1) {
	int *d_delta;
	char *d_input;
	int *d_fStates;
	int *d_check;
	int maxThreads=(len/K)-M+1;
	if(len%K!=0) {
		maxThreads=(len-M+1);
	}

	cudaMalloc((void**)&d_delta, M*N*sizeof(int));
	cudaMalloc((void**)&d_input, len*sizeof(char));
	cudaMalloc((void**)&d_fStates, (M*maxThreads)*sizeof(int));
	cudaMalloc((void**)&d_check, (maxThreads)*sizeof(int));
	cudaMemset(d_fStates, 0, M*maxThreads*sizeof(int));
	cudaMemset(d_check, 0, maxThreads*sizeof(int));

	cudaMemcpyHTD(d_delta, h_delta, M*N*sizeof(int));
	cudaMemcpyHTD(d_input, h_input, len*sizeof(char));

	dim3 threadsPerBlock(BLOCKSIZE);
	int nBlocks=(maxThreads-1)/BLOCKSIZE+1;
	dim3 numBlocks(nBlocks);
	printf("Max threads: %d\nNum blocks: %d\nThreads per block: %d\nVirtual Max threads: %d\n", maxThreads, nBlocks, BLOCKSIZE, BLOCKSIZE*nBlocks);

	specDFAMatching<<<numBlocks, threadsPerBlock>>>(M, N, len, d_delta, d_input, d_fStates, q0, maxThreads);

	/*
	checking
	*/
	int *h_fStates=(int*)malloc(M*maxThreads*sizeof(int));
	cudaMemcpyDTH(h_fStates, d_fStates, M*maxThreads*sizeof(int));
	int fst=q0;
	for(int i=0; i<maxThreads; ++i) {
		fst=h_fStates[fst+i*M];
	}
	printf("Parallel CHECKING says final state is: %d and %s a member\n", fst, (h_finalState[fst]?"YES":"NOT"));
	/*********
	**********/

	finalStateUsingRedn<<<numBlocks, threadsPerBlock>>>(M, d_fStates, q0, maxThreads, d_check);
	cudaMemcpyDTH(h_fStates, d_check, maxThreads*sizeof(int));
	for(int i=0; i<maxThreads; ++i) {
		if(false && h_fStates[i]!=1) {
			printf("Problem: %d :: %d\n", i, h_fStates[i]);
		}
	}

	int *finalStReached=(int*)malloc(maxThreads*M*sizeof(int));
	cudaMemcpyDTH(finalStReached, d_fStates, M*maxThreads*sizeof(int));

	int ret=finalStReached[q0];

	cudaFree(d_delta);
	cudaFree(d_check);
	cudaFree(d_input);
	cudaFree(d_fStates);
	delete[] h_fStates;
	delete[] finalStReached;

	return ret;
}

int main(int argc, char **argv) {
	if(argc<4) {
		printf("Usage: %s NUM_STATES NUM_ALPHABETS LEN_OF_INPUT\n", argv[0]);
		exit(1);
	}
	int M, N, len;
	M=atoi(argv[1]);
	N=atoi(argv[2]);
	len=atoi(argv[3]);

	if(M<2) {
		printf("Min number of states allowed is 2\n");
		exit(1);
	}
	if(N<2 || N>10) {
		printf("Number of alphabets should lie b/w 2 and 10 (inclusive)\n");
		exit(1);
	}

	printf("Number of states: %d\nNumber of alphabets: %d\nLength of input string: %d\n",M, N, len);

	int i, j;
	int *delta;
	bool *finalState;
	char *input;
	int q0=0;
	
	delta=(int*)malloc(M*N*sizeof(int));
	for(i=0; i<M; ++i) {
		for(j=0; j<N; ++j) {
			delta[j+i*N]=(rand()%M);
		}
	}

	finalState=(bool*)malloc(M*sizeof(bool));
	for(i=0; i<M-1; ++i) {
		if((1.0*rand())/RAND_MAX > 0.8)
			finalState[i]=true;
		else
			finalState[i]=false;
	}
	finalState[M-1]=true;

	input=(char*)malloc(len*sizeof(char));
	for(i=0; i<len; ++i) {
		input[i]=(char)('0'+(rand()%N));
	}

	//input string
	if(len<20)
		printf("INPUT: %s\n", input);

	//delta function
	printf("====DELTA===\n");
	for(i=0; i<M; ++i) {
		for(j=0; j<N; ++j) {
			printf("%d %d -> %d,  ", i, j, delta[j+i*N]);
		}
		printf("\n");
	}

	//Accepting and non accepting states
	printf("====ACCEPTING=STATES===\n");
	for(i=0; i<M; ++i) {
		printf("State %d -> %s\n", i, (finalState[i]?"ACCEPT":"REJECT"));
	}

	//Sequential
	int fst=q0;
	for(i=0; i<len; ++i) {
		fst=delta[(int)(input[i]-'0')+fst*N];
	}
	printf("Sequential says final state is: %d and %s a member\n", fst, (finalState[fst]?"YES":"NOT"));

	//parallel
	int res=SIMTMembershipTest(M, N, len, delta, finalState, input, q0);
	printf("Parallel says final state is: %d and %s a member\n", res, (finalState[res]?"YES":"NOT"));

	return 0;
}