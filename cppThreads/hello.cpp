#include <thread>
#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include <fstream>

#define MAX_THREADS 8


inline double my_clock(void) {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (1.0e-6*t.tv_usec + t.tv_sec);
}

void twice(long int *arr, long int s, long int e) {
	for(long int i = s; i < e; ++i) {
		arr[i] = arr[i]*2;
	}
}



int main(int argc, char **argv) {
	long int MAX_ELEM = atol(argv[1]);
	if(MAX_ELEM < MAX_THREADS) {
		std::cout << "argv[1] must be greater than " << MAX_THREADS << std::endl;
		exit(EXIT_FAILURE);
	}
	long int *arr = new long int[MAX_ELEM];
	for(long int i = 0; i < MAX_ELEM; ++i) {
		arr[i] = i;
	}
	std::ofstream seq("seq"), par("par");

	double s, e;
	s = my_clock();
	twice(arr, 0, MAX_ELEM);
	e = my_clock();
	//prlong intData(2, arr, MAX_ELEM, seq);
	std::cout << "Seq: " << e-s << std::endl;

	s = my_clock();
	std::thread t[MAX_THREADS];
	long int gap = (MAX_ELEM-1)/MAX_THREADS+1;
	for(long int i = 0; i < MAX_THREADS; ++i) {
		long int s_, e_;
		s_ = i*gap;
		e_ = std::min(s_+gap, MAX_ELEM);
		t[i] = std::thread(twice, arr, s_, e_);
	}
	for(long int i = 0; i < MAX_THREADS; ++i)
		t[i].join();
	e = my_clock();
	//prlong intData(4, arr, MAX_ELEM, par);
	std::cout << "Par: " << e-s << std::endl;

	return 0;
}