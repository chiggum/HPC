#define _USE_MATH_DEFINES
#include <cmath>

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <cmath>
#include <complex>
#include <thread>
#include <atomic>
#include <cassert>
#include "omp.h"
#include <sys/time.h>

#define INTERPOLATE -1
#define EXTRAPOLATE 1

#define value_type double

unsigned long nearPo2(unsigned long x) {
	return (std::pow(2, std::ceil(std::log(x)/std::log(2.0))));
}

inline double my_clock(void) {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (1.0e-6*t.tv_usec + t.tv_sec);
}


void printCompexVecInFile(std::vector< std::complex<value_type> > A, char *str) {
	std::ofstream outputFile(str);
	for(int i = 0; i < A.size(); ++i) {
		outputFile << A[i] << std::endl;
	}
	outputFile.close();
	return;
}

std::vector<value_type> naive(std::vector<value_type> A, std::vector<value_type> B) {
	int degC = A.size() + B.size() - 1;
	std::vector<value_type> C(degC, 0);
	for(int i = 0; i < C.size(); ++i) {
		value_type sum = 0;
		for(int j = 0; j <= i; ++j) {
			if(i-j >= A.size() || j >= B.size())
				continue;
			sum += A[i-j]*B[j];
		}
		C[i] = sum;
	}
	return C;
}

std::vector< std::complex<value_type> > ex_o_in_polate(std::vector< std::complex<value_type> > A, int TYPE) {
	if(A.size() == 1) {
		return A;
	}
	unsigned long N = A.size();		//will always be a power of 2
	std::complex<value_type> w_N(std::cos(TYPE*2.0*M_PI/N), std::sin(TYPE*2.0*M_PI/N));
	std::complex<value_type> w(1.0, 0);
	std::vector< std::complex<value_type> > A_o, A_e, A_o_c, A_e_c, A_p(N, std::complex<value_type>(0,0));
	for(int i = 0; i < A.size(); ++i) {
		if(i%2==0)
			A_e.push_back(A[i]);
		else
			A_o.push_back(A[i]);
	}
	A_e_c = ex_o_in_polate(A_e, TYPE);
	A_o_c = ex_o_in_polate(A_o, TYPE);

	for(int i = 0; i < N/2; ++i) {
		A_p[i] = A_e_c[i] + w*A_o_c[i];
		A_p[i+N/2] = A_e_c[i] - w*A_o_c[i];
		w = w*w_N;
	}
	return A_p;
}

std::vector< std::complex<value_type> > multiply_ptwise(std::vector< std::complex<value_type> > A, std::vector< std::complex<value_type> > B) {
	std::vector< std::complex<value_type> > C(A.size(), std::complex<value_type>(0,0));
	for(int i = 0; i < A.size(); ++i) {
		C[i] += A[i]*B[i];
	}
	return C;
}

std::vector<value_type> seq_dft(std::vector<value_type> A, std::vector<value_type> B) {
	std::complex<value_type> zero (0,0);
	int degC = A.size() + B.size() - 1;
	int npo2_degC = nearPo2(degC);

	std::vector< std::complex<value_type> > A_c(npo2_degC, zero), B_c(npo2_degC, zero), A_ex_c, B_ex_c, C_c, C_in_c;
	for(int i = 0; i < A.size(); ++i) {
		A_c[i] += A[i];
	}
	for(int i = 0; i < B.size(); ++i) {
		B_c[i] += B[i];
	}

	A_ex_c = ex_o_in_polate(A_c, EXTRAPOLATE);
	B_ex_c = ex_o_in_polate(B_c, EXTRAPOLATE);

	C_c = multiply_ptwise(A_ex_c, B_ex_c);

	C_in_c = ex_o_in_polate(C_c, INTERPOLATE);

	std::vector<value_type> res(degC, 0);
	for(int i =0; i < res.size(); ++i)
		res[i] = (C_in_c[i].real())/(1.0*npo2_degC);

	return res;
}

//A.size() = degA + 1 == num of elemnets in the vector == a0,a1,a2,....an-1

void addElem(std::complex<value_type> *A, value_type A_) {
	(*A) += (A_);
}

void addElem_v(value_type *A, value_type A_) {
	(*A) += (A_);
}

void addElem_c(std::complex<value_type> *A, std::complex<value_type> A_) {
	(*A) += (A_);
}

void addElem2(std::complex<value_type> A, std::complex<value_type> B, std::complex<value_type>  *C) {
	*C += A + B;
}

void multiply_kernel(std::complex<value_type> A, std::complex<value_type> B, std::complex<value_type> *C) {
	*C = A*B;
}

void ex_o_in_polate_parallel(std::vector< std::complex<value_type> > *A, 
							int TYPE, 
							std::vector< std::complex<value_type> > *A_res) {
	if(A->size() == 1) {
		(*A_res)[0] += (*A)[0];
		return;
	}
	unsigned long N = A->size();		//will always be a power of 2
	std::complex<value_type> w_N(std::cos(TYPE*2.0*M_PI/N), std::sin(TYPE*2.0*M_PI/N));
	std::complex<value_type> w(1.0, 0);
	std::complex<value_type> zero(0,0);
	std::vector< std::complex<value_type> > A_o(N/2, zero), A_e(N/2, zero);

	int cntO = 0, cntE = 0;
	if(omp_get_thread_num() > 1) {
		#pragma omp parallel for
		for(int i = 0; i < N/2; ++i) {
			addElem_c(&A_e[cntE++], (*A)[i*2]);
			addElem_c(&A_o[cntO++], (*A)[i*2+1]);
		}
	} else {
		for(int i = 0; i < N/2; ++i) {
			addElem_c(&A_e[cntE++], (*A)[i*2]);
			addElem_c(&A_o[cntO++], (*A)[i*2+1]);
		}
	}

	std::vector< std::complex<value_type> > A_o_c(A_o.size(), zero), A_e_c(A_e.size(), zero);
	/*
	#pragma omp parallel for
	for(int i = 0; i < 2; ++i) {
		if(i==0)
			ex_o_in_polate_parallel(&A_e, TYPE, &A_e_c);
		else
			ex_o_in_polate_parallel(&A_o, TYPE, &A_o_c);
	}
	*/
	#pragma omp parallel num_threads(2)
    {
        int i = omp_get_thread_num();

        if (i == 0){
            ex_o_in_polate_parallel(&A_e, TYPE, &A_e_c);
        }
        if (i == 1 || omp_get_num_threads() != 2){
            ex_o_in_polate_parallel(&A_o, TYPE, &A_o_c);
        }
    }

    if(omp_get_num_threads() > 1) {
		#pragma omp parallel for
		for(int i = 0; i < N/2; ++i) {
			addElem_c(&((*A_res)[i]), A_e_c[i]+w*A_o_c[i]);
			addElem_c(&((*A_res)[i+N/2]), A_e_c[i]-w*A_o_c[i]);
			w = std::pow(w_N,i+1);
		}
	} else {
		for(int i = 0; i < N/2; ++i) {
			addElem_c(&((*A_res)[i]), A_e_c[i]+w*A_o_c[i]);
			addElem_c(&((*A_res)[i+N/2]), A_e_c[i]-w*A_o_c[i]);
			w = w*w_N;
		}
	}
}


std::vector<value_type> par_dft(std::vector<value_type> A, std::vector<value_type> B) {
	std::complex<value_type> zero (0,0);
	int degC = A.size() + B.size() - 1;
	int npo2_degC = nearPo2(degC);

	std::vector< std::complex<value_type> > A_c(npo2_degC, zero), B_c(npo2_degC, zero),
	A_ex_c(npo2_degC, zero), B_ex_c(npo2_degC, zero), C_c(npo2_degC, zero), 
	C_in_c(npo2_degC, zero);
	
	
	#pragma omp parallel for
	for(int i = 0; i < (A.size()); ++i) {
		addElem(&A_c[i], A[i]);
	}
	#pragma omp parallel for
    for(int i = 0; i < (B.size()); ++i) {
		addElem(&B_c[i], B[i]);
	}
	#pragma omp parallel num_threads(2)
    {
        int i = omp_get_thread_num();

        if (i == 0){
            ex_o_in_polate_parallel(&A_c, EXTRAPOLATE, &A_ex_c);
        }
        if (i == 1 || omp_get_num_threads() != 2){
            ex_o_in_polate_parallel(&B_c, EXTRAPOLATE, &B_ex_c);
        }
    }
/*
	#pragma omp parallel for
	for(int i = 0; i < 2; ++i) {
		if(i==0)
			ex_o_in_polate_parallel(&A_c, EXTRAPOLATE, &A_ex_c);
		else
			ex_o_in_polate_parallel(&B_c, EXTRAPOLATE, &B_ex_c);
	}
	*/
	//multiply pointwise

	#pragma omp parallel for
	for(int i = 0; i < (A_ex_c.size()); ++i) {
		multiply_kernel(A_ex_c[i], B_ex_c[i], &C_c[i]);
	}

    ex_o_in_polate_parallel(&C_c, INTERPOLATE, &C_in_c);

	std::vector<value_type> res(degC, 0);

	#pragma omp parallel for
	for(int i = 0; i < (degC); ++i) {
		addElem_v(&res[i], C_in_c[i].real()/(1.0*npo2_degC));
	}

	return res;
}

int main(int argc, char **argv) {
	if(argc != 6) {
		std::cout << "Usage: " << argv[0] << " input_file output_file deg_A deg_B NAIVE" << std::endl;
		exit(EXIT_FAILURE);
	}


	double cpuTime;
	int degA = atoi(argv[3]);
	int degB = atoi(argv[4]);
	int toRunNaive = atoi(argv[5]);
	std::vector<value_type> A, B, C;
	std::ifstream input(argv[1]);

	for(int i = 0; i < degA+1; ++i) {
		value_type x;
		input >> x;
		A.push_back(x);
	}

	for(int i = 0; i < degB+1; ++i) {
		value_type x;
		input >> x;
		B.push_back(x);
	}

	input.close();

	clock_t start, end;
	
	start = clock();
	C = naive(A, B);
	end = clock();
	cpuTime = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("%fs nnnn\n", cpuTime);
	

	start = clock();
	C = seq_dft(A, B);
	end = clock();
	cpuTime = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("%fs ssss\n", cpuTime);
	C.clear();

	double start_ = my_clock();
	C = par_dft(A, B);
	double end_ = my_clock();
	double cpuTime_ = ((double) (end_ - start_));
	printf("%fs pppp\n", cpuTime_);
	C.clear();

	return 0;
}