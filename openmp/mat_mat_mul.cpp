#include <cstdlib>
#include <time.h>
#include <iostream>
#include <omp.h>
#include <sys/time.h>

inline double my_clock(void) {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (1.0e-6*t.tv_usec + t.tv_sec);
}


int main(int argc, char **argv) {
	int p=atoi(argv[1]);
	int q=atoi(argv[2]);
	int r=atoi(argv[3]);

	bool p1=false, p2=false, p3=true, p4=true;

	int **A, **B, **C;
	A=new int*[p];
	B=new int*[q];
	C=new int*[p];

	for(int i=0; i<p; ++i) {
		A[i]=new int[q];
		C[i]=new int[r];
		for(int j=0; j<q; ++j) {
			A[i][j]=rand();
		}
	}
	for(int i=0; i<q; ++i) {
		B[i]=new int[r];
		for(int j=0; j<r; ++j) {
			B[i][j]=rand();
		}
	}

	clock_t t1, t2;
	double tm;
	t1=clock();

	//naive seq
	for(int i=0; i<p && p1; ++i) {
		for(int j=0; j<r; ++j) {
			C[i][j]=0;
			for(int k=0; k<q; ++k) {
				C[i][j]+=A[i][k]*B[k][j];
			}
		}
	}

	t2=clock();
	tm=((double)(t2 - t1))/CLOCKS_PER_SEC;
	std::cout << "Naive: " << tm << "s" << std::endl;

	t1=clock();

	//opt seq
	for(int i=0; i<p && p2; ++i) {
		for(int j=0; j<r; ++j) {
			int sum=0;
			for(int k=0; k<q; ++k) {
				sum+=A[i][k]*B[k][j];
			}
			C[i][j]=sum;
		}
	}

	t2=clock();
	tm=((double)(t2 - t1))/CLOCKS_PER_SEC;
	std::cout << "opt seq: " << tm << "s" << std::endl;

	int *A_r=new int[q];
	t1=clock();

	//opt seq
	for(int i=0; i<p && p3; ++i) {
		for(int k=0; k<q; ++k) {
			A_r[k]=A[i][k];
		}
		for(int j=0; j<r; ++j) {
			int sum=0;
			for(int k=0; k<q; ++k) {
				sum+=A_r[k]*B[k][j];
			}
			C[i][j]=sum;
		}
	}

	t2=clock();
	tm=((double)(t2 - t1))/CLOCKS_PER_SEC;
	std::cout << "opt seq 2: " << tm << "s" << std::endl;

	double t1_, t2_;
	t1_=my_clock();

	//naive par
	#pragma omp parallel for
	for(int i=0; i<p; ++i) {
		#pragma omp parallel for
		for(int j=0; j<r; ++j) {
			int sum=0;
			for(int k=0; k<q; ++k) {
				sum+=A[i][k]*B[k][j];
			}
			C[i][j]=sum;
		}
	}

	t2_=my_clock();
	tm=((double)(t2_ - t1_));
	std::cout << "par : " << tm << "s" << std::endl;



	return 0;
}