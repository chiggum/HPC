/*
DFA Membership test in SIMT Environment
Author: Dhruv Kohli
Mentor: Kalpesh Kapoor

Reference:
"A Speculative Parallel DFA Membership Test for Multicore, SIMD and Cloud Computing Environments" Yousun Ko, Minyoung Jung, Yo-Sub Han, Bernd Burgstaller

For now: length of input must be a multiple of size of chunk assigned to each thread.
*/

#include <cstdlib>
#include <cstdio>

#define cudaMemcpyHTD(dest, src, sizeInBytes) cudaMemcpy(dest, src, sizeInBytes, cudaMemcpyHostToDevice)
#define cudaMemcpyDTH(dest, src, sizeInBytes) cudaMemcpy(dest, src, sizeInBytes, cudaMemcpyDeviceToHost)

#define BLOCKSIZE 1024

typedef unsigned long long int ull;

__global__ void specDFAMatching(ull M, ull N, ull len, ull *delta, char *input, ull *fStates, ull q0, ull maxThreads) {
	ull idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx>=maxThreads)
		return;
	ull i, j;
	if(idx==0) {
		ull fst=q0;
		ull l=0;
		ull r=(M)*(len/(M+maxThreads-1))-1;
		for(i=l; i<=r; ++i) {
			fst=delta[(ull)(input[i]-'0')+fst*N];
		}
		fStates[q0+idx*M]=fst;
	} else {
		ull l=(M)*(len/(M+maxThreads-1)) + (idx-1)*(len/(M+maxThreads-1));
		ull r=l+(len/(M+maxThreads-1))-1;
		for(j=0; j<M; ++j) {
			ull fst=j;
			for(i=l; i<=r; ++i) {
				fst=delta[(ull)(input[i]-'0')+fst*N];
			}
			fStates[j+idx*M]=fst;
		}
	}
}

__global__ void finalStateUsingRedn1(ull M, ull *fStates, ull q0, ull maxThreads, ull s) {
	ull idx=threadIdx.x+blockIdx.x*blockDim.x;
	if(idx+s>=maxThreads)
		return;
	if(idx%(2*s)==0) {
		for(ull i=0; i<M; ++i) {
			ull x=fStates[i+idx*M];
			fStates[i+idx*M]=fStates[x+(idx+s)*M];
		}
	}
}

ull SIMTMembershipTest(ull M, ull N, ull len, ull *h_delta, bool *h_finalState, char *h_input, ull q0, ull K=1) {
	ull *d_delta;
	char *d_input;
	ull *d_fStates;
	ull maxThreads=(len/K)-M+1;
	if(len%K!=0) {
		//maxThreads=(len-M+1);
		printf("Current impl. constrains length of input string to be a multiple of K\n");
		exit(EXIT_FAILURE);
	}

	cudaMalloc((void**)&d_delta, M*N*sizeof(ull));
	cudaMalloc((void**)&d_input, len*sizeof(char));
	cudaMalloc((void**)&d_fStates, (M*maxThreads)*sizeof(ull));
	cudaMemset(d_fStates, 0, M*maxThreads*sizeof(ull));

	cudaMemcpyHTD(d_delta, h_delta, M*N*sizeof(ull));
	cudaMemcpyHTD(d_input, h_input, len*sizeof(char));

	dim3 threadsPerBlock(BLOCKSIZE);
	ull nBlocks=(maxThreads-1)/BLOCKSIZE+1;
	dim3 numBlocks(nBlocks);
	#ifdef DEB
	printf("Max threads: %llu\nNum blocks: %llu\nThreads per block: %llu\nVirtual Max threads: %llu\nK: %llu\n\n", maxThreads, nBlocks, (ull)BLOCKSIZE, (ull)BLOCKSIZE*nBlocks, K);
	#else
	printf("%llu ", maxThreads);
	#endif
	specDFAMatching<<<numBlocks, threadsPerBlock>>>(M, N, len, d_delta, d_input, d_fStates, q0, maxThreads);

	for(ull s=1; s<maxThreads; s*=2) {
		finalStateUsingRedn1<<<numBlocks, threadsPerBlock>>>(M, d_fStates, q0, maxThreads, s);
	}

	ull ret;
	cudaMemcpyDTH(&ret, d_fStates+q0, sizeof(ull));

	cudaFree(d_delta);
	cudaFree(d_input);
	cudaFree(d_fStates);

	return ret;
}

int main(int argc, char **argv) {
	if(argc<4) {
		printf("Usage: %s NUM_STATES NUM_ALPHABETS LEN_OF_INPUT [CHUNK_SIZE_PER_PROC] [FLAG_TO_NOT_RUN_SEQ]\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	ull M, N, len;
	M=strtoull(argv[1], NULL, 10);
	N=strtoull(argv[2], NULL, 10);
	len=strtoull(argv[3], NULL, 10);
	ull K=1;
	bool toRunSeq=true;
	if(argc>=5) {
		K=strtoull(argv[4], NULL, 10);
	}
	if(argc>=6) {
		toRunSeq=false;
	}
	if(M<2) {
		printf("Min number of states allowed is 2\n");
		exit(1);
	}
	if(N<2 || N>10) {
		printf("Number of alphabets should lie b/w 2 and 10 (inclusive)\n");
		exit(1);
	}

	#ifdef DEB
	printf("Number of states: %llu\nNumber of alphabets: %llu\nLength of input string: %llu\n\n",M, N, len);
	#else
	printf("%llu %llu ", M, len);
	#endif

	ull i, j;
	ull *delta;
	bool *finalState;
	char *input;
	ull q0=0;
	
	delta=(ull*)malloc(M*N*sizeof(ull));
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

	clock_t start, end;
	start = clock();
	double cpuTime;

	if(toRunSeq) {
		//Sequential
		ull fst=q0;
		for(i=0; i<len; ++i) {
			fst=delta[(ull)(input[i]-'0')+fst*N];
		}
		#ifdef DEB
		printf("Sequential says final state is: %llu and %s a member\n", fst, (finalState[fst]?"YES":"NOT"));
		#endif

		end = clock();
		cpuTime = ((double) (end - start)) / CLOCKS_PER_SEC;
		#ifdef DEB
		printf("Time by sequential algo: %lf\n", cpuTime);
		#else
		printf("%lf ", cpuTime);
		#endif
	}

	start=clock();
	//parallel
	ull res=SIMTMembershipTest(M, N, len, delta, finalState, input, q0, K);
	#ifdef DEB
	printf("Parallel says final state is: %llu and %s a member\n", res, (finalState[res]?"YES":"NOT"));
	#endif

	end = clock();
	cpuTime = ((double) (end - start)) / CLOCKS_PER_SEC;
	#ifdef DEB
	printf("Time by Parallel algo: %lf\n", cpuTime);
	#else
	printf("%lf\n", cpuTime);
	#endif

	delete[] delta;
	delete[] finalState;
	delete[] input;

	return 0;
}