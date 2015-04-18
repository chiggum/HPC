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
#include "complex.hpp"

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
double maxZAbs = 100;
int maxN = 255;
int minN = 5;
int decayN = 5;
double offset = 4;
double realMin = -2;
double imgMin = -2;	
double offMul = 1;
int numColors = 52;
/*********************************************
*********************************************/
	

/*********************************************
HSV TO RGB
*********************************************/
typedef struct {
    double r;       // percent
    double g;       // percent
    double b;       // percent
} rgb;

__device__ rgb hsl2rgb(double h, double sl, double l)
{
    double v;
    double r,g,b;

    r = l;   // default to gray
    g = l;
    b = l;
    v = (l <= 0.5) ? (l * (1.0 + sl)) : (l + sl - l * sl);
    if (v > 0) {
          double m;
          double sv;
          int sextant;
          double fract, vsf, mid1, mid2;
          m = l + l - v;
          sv = (v - m ) / v;
          h *= 6.0;
          sextant = (int)h;
          fract = h - sextant;
          vsf = v * sv * fract;
          mid1 = m + vsf;
          mid2 = v - vsf;
          switch (sextant) {
                case 0:
                      r = v;
                      g = mid1;
                      b = m;
                      break;
                case 1:
                      r = mid2;
                      g = v;
                      b = m;
                      break;
                case 2:
                      r = m;
                      g = v;
                      b = mid1;
                      break;
                case 3:
                      r = m;
                      g = mid2;
                      b = v;
                      break;
                case 4:
                      r = mid1;
                      g = m;
                      b = v;
                      break;
                case 5:
                      r = v;
                      g = m;
                      b = mid2;
                      break;
          }
    }
    rgb rgb;
    rgb.r = (r * 255.0f);
    rgb.g = (g * 255.0f);
    rgb.b = (b * 255.0f);
    return rgb;
}

/**************************************************
*************************************************/

__device__ static unsigned long next = 1;

__device__ double my_rand() {
	next = next * 1103515245 + 12345;
    return((double)((unsigned)(next/65536) % 32768))/32768.0;
}

__device__ rgb getColor(int n, int MAXCOLORS) {
	double val = (360.0*n)/MAXCOLORS;
	double hue = val/360.0;
	double lightness = (50.0 + (10.0*my_rand()))/100.0;
	double saturation = (90.0 + (10.0*my_rand()))/100.0;
	return hsl2rgb(hue, saturation, lightness);
}



__device__ __forceinline__ cuDoubleComplex my_cexpf(cuDoubleComplex z) {
    cuFloatComplex res;
    float t = expf (z.x);
    sincosf(z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return cuComplexFloatToDouble(res);
}

__device__ complex<double> getFuncVal(complex<double> z, complex<double> c, int categ) {
	switch(categ) {
		case 1:
			return z*z+c;
		case 2:
			return pow(z,complex<double>(3,0))+c;
		case 3:
			return pow(z,complex<double>(4,0))+c;
		case 4:
			return pow(z,complex<double>(5,0))+c;
		case 5:
			return exp(z)+c;
		case 6:
			return exp(pow(z,complex<double>(3,0)))+c;
		case 7:
			return z*exp(z)+c;
		case 8:
			return z*z*exp(z)+c;
		case 9:
			return pow(z,complex<double>(3,0))*exp(z)+c;
		case 10:
			complex<double> temp=sinh(z*z);
			return temp*temp+c;
		case 11:
			return (complex<double>(1,0)-z*z + (pow(z,complex<double>(5,0)))/(complex<double>(2,0)+complex<double>(4,0)*z)) + c;
		case 12:
			return (cos(exp(z))*sin(exp(z))+c);
		case 13:
			return cos(z)+c;
		case 14:
			return sin(z)+c;
		case 15:
			return log(z)*cos(z) + c;
		case 16:
			return cos(z)/(z)+c;
		case 17:
			return log(z)/z + c;
		case 18:
			return sinh(z)*z+c;
		case 19:
			return sinh(z)*cosh(z)*sin(z)*cos(z)+c;
		case 20:
			return exp(exp(z))+c;
		default:
			return z*z+c;
	}
}

__global__ void fractalForm(unsigned char *mat, int maxZAbs, int maxN, int minN, int decayN, double iReal,
							double iImg, int categ, double rMin, double rMax, double iMin, double iMax,
							int H, int W, int numCols) {
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if(idx >= H*W)
		return;
	int i_ = idx/W;
	int j_ = idx%W;
	double re = rMin + (i_*(rMax-rMin))/(1.0*H);
	double im = iMin + (j_*(iMax-iMin))/(1.0*W);
	complex<double> z(re,im);
	complex<double> c(iReal, iImg);
	size_t n;
	for(n = maxN; n >= minN && abs(z) < maxZAbs; n-=decayN) {
		z = getFuncVal(z, c, categ);
	}
	rgb col = getColor(n/decayN, numCols);
	mat[3*(j_ + i_*W)]=(unsigned char)(int)col.g;
	mat[3*(j_ + i_*W)+1]=(unsigned char)(int)col.r;
	mat[3*(j_ + i_*W)+2]=(unsigned char)(int)col.b; 
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
	
	
	unsigned char *d_mat;
	cudaMalloc((void**)&d_mat, 3*H*W*sizeof(unsigned char));
	
	dim3 threadsPerBlock(BLOCKSIZE);
	dim3 numBlocks((H*W-1)/threadsPerBlock.x + 1);
	
	cv::Mat finalImg(H, W, CV_8UC3);
	unsigned char *fImgData = (unsigned char*)(finalImg.data);
	
	for(double iReal = (interceptRealMin+interceptRealMax)/2, iImg = (interceptImgMin+interceptImgMax)/2;;) {
		fractalForm<<<numBlocks, threadsPerBlock>>>(d_mat, maxZAbs, maxN, minN, decayN, iReal,
													iImg, category, realMin, realMin + offset, 
													imgMin, imgMin + offset, H, W, numColors);
	
		cudaMemcpyDTH(fImgData, d_mat, 3*H*W*sizeof(unsigned char));
		
		
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
	cudaFree(d_mat);
	return 0;
}