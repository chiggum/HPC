/*
COMPILE WITH
nvcc fractal.cu -lopencv_highgui -lopencv_core -lopencv_imgproc -o fractal
*/
#include <iostream>
#include <cstdlib>
#include <cuComplex.h>
#include "opencv2/core/core.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/highgui/highgui.hpp"

#define cudaMemcpyHTD(dest, src, nBytes) cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice)
#define cudaMemcpyDTH(dest, src, nBytes) cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost)

#define BLOCKSIZE 512


/**********************************************
Configuration:
***********************************************/
double interceptRealMin = -1;
double interceptRealMax = 1;
double interceptImgMin = -1;
double interceptImgMax = 1;
double realOffset = 0.01;
double imgOffset = 0.01;
double maxZAbs = 10;
int maxN = 255;
int minN = 1;
int decayN = 1;
double offset = 4;
double realMin = -2;
double imgMin = -2;	
double offMul = 1;
/*********************************************
*********************************************/
	
	

__device__ __forceinline__ cuDoubleComplex my_cexpf(cuDoubleComplex z) {
    cuFloatComplex res;
    float t = expf (z.x);
    sincosf(z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return cuComplexFloatToDouble(res);
}

__device__ cuDoubleComplex getFuncVal(cuDoubleComplex z, cuDoubleComplex c, int categ) {
	switch(categ) {
		case 1:
			return cuCadd(cuCmul(z,z), c);
		case 2:
			return cuCadd(cuCmul(cuCmul(z, z),z), c);
		case 3:
			return cuCadd(cuCmul(cuCmul(cuCmul(z, z), z),z), c);
		case 4:
			return cuCadd(cuCmul(cuCmul(cuCmul(cuCmul(z, z), z), z),z), c);
		case 5:
			return cuCadd(my_cexpf(z), c);
		case 6:
			return cuCadd(my_cexpf(cuCmul(cuCmul(cuCmul(z, z), z),z)), c);
		case 7:
			return cuCadd(cuCmul(my_cexpf(z), z),z);
		case 8:
			return cuCadd(cuCmul(cuCmul(my_cexpf(z), z), z),z);
		case 9:
			return cuCadd(cuCmul(cuCmul(cuCmul(my_cexpf(z), z), z), z),z);
		default:
			return cuCadd(cuCmul(z,z), c);
	}
}

__global__ void fractalForm(int *mat, int maxZAbs, int maxN, int minN, int decayN, double iReal,
							double iImg, int categ, double rMin, double rMax, double iMin, double iMax,
							int H, int W) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx >= H*W)
		return;
	int i_ = idx/W;
	int j_ = idx%W;
	double re = rMin + (i_*(rMax-rMin))/(1.0*H);
	double im = iMin + (j_*(iMax-iMin))/(1.0*W);
	cuDoubleComplex z = make_cuDoubleComplex(re, im);
	cuDoubleComplex c = make_cuDoubleComplex(iReal, iImg);
	size_t n;
	for(n = maxN; n >= minN && cuCabs(z) < maxZAbs; n-=decayN) {
		z = getFuncVal(z, c, categ);
	}
	
	mat[j_ + i_*W] = n;
}

int main(int argc, char **argv) {
	
	if(argc != 3) {
		std::cout << "Usage: " << argv[0] << " " << "height=width" << " " << "category in 1 to 10" << std::endl;
		exit(EXIT_FAILURE);
	} else {
		std::cout << "press\nq/z for quit\nf,g\nv,b\no,p\nk,l\nn,m\nu,i" << std::endl;
	}
	int H = atoi(argv[1]);
	int W = H;
	int category = atoi(argv[2]);
	
	
	int *h_mat, *d_mat;
	h_mat = new int[H*W];
	cudaMalloc((void**)&d_mat, H*W*sizeof(int));
	
	dim3 threadsPerBlock(BLOCKSIZE);
	dim3 numBlocks((H*W-1)/threadsPerBlock.x + 1);
	
	cv::Mat finalImg(H, W, CV_8UC3);
	
	for(double iReal = (interceptRealMin+interceptRealMax)/2, iImg = (interceptImgMin+interceptImgMax)/2;;) {
		fractalForm<<<numBlocks, threadsPerBlock>>>(d_mat, maxZAbs, maxN, minN, decayN, iReal,
													iImg, category, realMin, realMin + offset, 
													imgMin, imgMin + offset, H, W);
	
		cudaMemcpyDTH(h_mat, d_mat, H*W*sizeof(int));

		for(size_t i = 0; i < H; ++i) {
		     for(size_t j = 0; j < W; ++j) {
				//finalImg.at<uchar>(i,j) = h_mat[j+i*W];
				finalImg.at<cv::Vec3b>(i, j)[2] = h_mat[j+i*W];
		     }
		}
		cv::imshow("fractal", finalImg);
		char c = cv::waitKey(0);
		if(c == 'f') {
			iReal-=realOffset;
		} else if(c == 'g') {
			iReal+=realOffset;
		} else if(c == 'v') {
			iImg -= imgOffset;
		} else if(c == 'b') {
			iImg += imgOffset;
		} else if(c == 'z' || c == 'q') {
			break;
		} else if(c == 's') {
			cv::imwrite("fractal.png", finalImg);
		} else if(c == 'o') {
			offset -= 0.05*offMul;
		} else if(c == 'p') {
			offset += 0.05*offMul;
		} else if(c == 'k') {
			realMin -= 0.05*offMul;
		} else if(c == 'l') {
			realMin += 0.05*offMul;
		} else if(c == 'n') {
			imgMin -= 0.05*offMul;
		} else if(c == 'm') {
			imgMin += 0.05*offMul;
		} else if(c == 'u') {
			offMul/=10;
		} else if(c == 'i') {
			offMul*=10;
		}
		
	}
	return 0;
}