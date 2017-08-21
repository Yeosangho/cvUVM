#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/core/cuda.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "memoryAccessor.h"
using namespace cv;
using namespace std;
using namespace std;
int main(int argc, char** argv){
	cuSetDeviceFlags();
	float* h_img;
	int rows = 1280;
	int cols = 720;

	cuMallocManaged(&h_img, rows, cols);

	Mat hmat;
	hmat = Mat(cvSize(cols, rows), CV_32F,  h_img);
	Mat result;
	hmat = cv::imread("../720p.jpg");
	cout << hmat << endl;
	cuda::GpuMat dmat(cvSize(rows, cols), CV_32F, h_img);
	cuDeviceSynchronize();
	double timeSec = 0;
	int64 startUVM = getTickCount();
	for(int i=0; i<100; i++){
		h_img++;
		//cout << *h_img	<< endl;
		//cout << dmat.channels() << endl;	
		timeSec += (getTickCount() - startUVM) / getTickFrequency();
	}
	dmat.download(hmat);
	cuDeviceSynchronize();
	cv::imshow("test", hmat);
	cout << timeSec << endl;
	
	float* h_img2 = (float*)malloc(sizeof(float)*rows*cols);
	float* d_img2;
	
	cuMalloc((void **)&d_img2, rows, cols);
	
	Mat hmat2(cvSize(rows, cols), CV_32F, h_img2);
	
	cuda::GpuMat dmat2(cvSize(rows, cols), CV_32F, d_img2);
	timeSec = 0;
	startUVM = getTickCount();
	for(int i=0; i<100; i++)
	{
		dmat2.upload(hmat2);
		dmat2.download(hmat2);
		timeSec += (getTickCount() - startUVM) / getTickFrequency();
	}
	cout << timeSec << endl;


	waitKey(0);
	//cout << dmat << endl;
	return 0;
}
